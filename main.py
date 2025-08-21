from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import asyncio
import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from agents import Agent, Runner, handoff
from agents.run_context import RunContextWrapper
from agents.mcp import create_static_tool_filter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """Represents a user session with cached MCP servers and agents."""
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    mcp_servers: List[Any]
    agents: Dict[str, Agent]
    conversation_history: List[dict]
    cached_tools: Optional[Dict[str, List]] = None
    
    def is_expired(self, timeout_minutes: int = 60) -> bool:
        """Check if session has expired."""
        return datetime.now() - self.last_accessed > timedelta(minutes=timeout_minutes)
    
    def update_access_time(self):
        """Update last accessed time."""
        self.last_accessed = datetime.now()

class OptimizedMCPManager:
    """Optimized MCP Manager with session support and performance improvements."""
    
    def __init__(self, session_timeout_minutes: int = 60):
        self.sessions: Dict[str, UserSession] = {}
        self.session_timeout_minutes = session_timeout_minutes
        self._cleanup_task = None
        self._server_configs = {}
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task to cleanup expired sessions."""
        async def cleanup_expired_sessions():
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    await self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    
    async def _cleanup_expired_sessions(self):
        """Remove expired sessions and close their resources."""
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired(self.session_timeout_minutes)
        ]
        
        for session_id in expired_sessions:
            session = self.sessions.pop(session_id)
            await self._close_session_resources(session)
            logger.info(f"Cleaned up expired session: {session_id}")
    
    async def _close_session_resources(self, session: UserSession):
        """Close resources associated with a session."""
        for server in session.mcp_servers:
            try:
                if hasattr(server, 'close'):
                    await server.close()
            except Exception as e:
                logger.error(f"Error closing MCP server: {e}")
    
    def register_server_config(self, name: str, config: Dict[str, Any]):
        """Register a reusable MCP server configuration."""
        self._server_configs[name] = config
    
    async def create_optimized_server(self, config: Dict[str, Any]) -> Any:
        """Create an optimized MCP server with caching enabled."""
        server_type = config.get('type', 'stdio')
        params = config.get('params', {})
        tool_filter_config = config.get('tool_filter')
        cache_tools = config.get('cache_tools', True)
        
        # Create tool filter if specified
        tool_filter = None
        if tool_filter_config:
            if tool_filter_config.get('type') == 'static':
                tool_filter = create_static_tool_filter(
                    allowed_tool_names=tool_filter_config.get('allowed_tools'),
                    blocked_tool_names=tool_filter_config.get('blocked_tools')
                )
            elif tool_filter_config.get('type') == 'dynamic':
                tool_filter = tool_filter_config.get('function')
        
        # Create server based on type
        if server_type == 'stdio':
            server = MCPServerStdio(
                params=params,
                tool_filter=tool_filter,
                cache_tools_list=cache_tools
            )
        elif server_type == 'sse':
            server = MCPServerSse(
                params=params,
                tool_filter=tool_filter,
                cache_tools_list=cache_tools
            )
        elif server_type == 'streamable_http':
            server = MCPServerStreamableHttp(
                params=params,
                tool_filter=tool_filter,
                cache_tools_list=cache_tools
            )
        else:
            raise ValueError(f"Unsupported server type: {server_type}")
        
        return server
    
    async def create_session(self, user_id: str, server_configs: List[str] = None) -> str:
        """Create a new user session with optimized MCP servers."""
        session_id = str(uuid.uuid4())
        
        # Create MCP servers for this session if configs provided
        mcp_servers = []
        if server_configs:
            for config_name in server_configs:
                if config_name not in self._server_configs:
                    raise ValueError(f"Unknown server configuration: {config_name}")
                
                config = self._server_configs[config_name]
                server = await self.create_optimized_server(config)
                mcp_servers.append(server)
        
        # Create session
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            mcp_servers=mcp_servers,
            agents={},
            conversation_history=[]
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {user_id}")
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID and update access time."""
        session = self.sessions.get(session_id)
        if session and not session.is_expired(self.session_timeout_minutes):
            session.update_access_time()
            return session
        elif session:
            # Session expired, clean it up
            await self._close_session_resources(session)
            del self.sessions[session_id]
        return None
    
    def create_agents_with_mcp(self, mcp_servers: List[Any]) -> Dict[str, Agent]:
        """Create all interview agents with MCP servers."""
        
        transition_agent = Agent(
            name="Transition Agent",
            instructions=(
                "Analyze the political conversation and decide when to transition topics."
                "If the conversation is ready for summarization of their political views, hand off to summary_agent. "
                "If it's time for a final reflection question about their political beliefs, hand off to final_question_agent. "
                "If the political interview should end, hand off to end_interview_agent. "
                "Otherwise, provide guidance on how to continue exploring their political views and correlations."
            ),
            model="gpt-4o",
            mcp_servers=mcp_servers,
            handoffs=[]
        )

        summary_agent = Agent(
            name="Summary Agent", 
            instructions="Summarize the user's political views, key positions, and the correlations between different political issues they discussed. Provide a clear, comprehensive summary of their political beliefs and how they connect their various positions.",
            model="gpt-4o",
            mcp_servers=mcp_servers
        )

        final_question_agent = Agent(
            name="Final Question Agent",
            instructions="Ask one final thoughtful question about their political views to help them reflect on their overall political philosophy or how their beliefs have evolved. Make it open-ended and deeply introspective about their political journey.",
            model="gpt-4o",
            mcp_servers=mcp_servers
        )

        end_interview_agent = Agent(
            name="End Interview Agent", 
            instructions=(
                "Provide a thoughtful conclusion to the political interview. Thank them for sharing their political views and offer any final reflections on the complexity of political beliefs. "
                "Always end your response with 'CONCLUDE_INTERVIEW' to signal the interview is complete."
            ),
            model="gpt-4o",
            mcp_servers=mcp_servers
        )

        guardrail_agent = Agent(
            name="Guardrail Agent",
            instructions="Monitor political conversation for safety and appropriateness. Ensure the discussion remains respectful and focused on understanding views rather than debate. If the conversation becomes inappropriate, hostile, or unsafe, respond with 'FLAG: [reason]'. Otherwise, respond with 'CLEAR'. After flagging, ask a clarifying question to redirect the conversation constructively.",
            model="gpt-4o",
            mcp_servers=mcp_servers
        )

        context_agent = Agent(
            name="Context Agent",
            instructions="Given a political conversation history and a new user message, rewrite the message with useful context from the previous political discussion. Output only the rewritten message, making it clearer and more specific based on the political topics that have been discussed.",
            model="gpt-4o",
            mcp_servers=mcp_servers
        )

        interview_agent = Agent(
            name="Political Interview Agent",
            instructions=(
                "You are conducting a thoughtful interview about political views and beliefs. Your goal is to deeply understand their political perspectives and the connections between their various positions. "
                "ALWAYS lead with questions - never wait for the user to start. "
                "Start by asking what political issues they are most passionate about and why those issues matter to them personally. "
                "Once they share their passionate issues, dive deep into the underlying reasons and values that drive these beliefs. "
                "Explore correlations between different political positions - ask how their views on one issue relate to their views on others. "
                "Ask about the personal experiences, values, or principles that shaped their political beliefs. "
                "Probe into how they see connections between seemingly different political topics. "
                "Explore potential tensions or consistencies in their political worldview. "
                "Ask about how they prioritize different political issues when they might conflict. "
                "Guide the conversation to understand the deeper philosophical or moral foundations of their political views. "
                "Remain neutral and curious - your goal is understanding, not debating or challenging. "
                "After 5-7 substantial exchanges about their political views, consider handing off to the transition agents. "
                "Always end your responses with a question unless concluding the interview."
            ),
            model="gpt-4o",
            mcp_servers=mcp_servers,
            handoffs=[
                handoff(transition_agent),
                handoff(summary_agent), 
                handoff(final_question_agent),
                handoff(end_interview_agent)
            ]
        )

        # Update transition agent handoffs
        transition_agent = Agent(
            name="Transition Agent",
            instructions=transition_agent.instructions,
            model="gpt-4o",
            mcp_servers=mcp_servers,
            handoffs=[
                handoff(summary_agent),
                handoff(final_question_agent), 
                handoff(end_interview_agent)
            ]
        )
        
        return {
            'interview_agent': interview_agent,
            'transition_agent': transition_agent,
            'summary_agent': summary_agent,
            'final_question_agent': final_question_agent,
            'end_interview_agent': end_interview_agent,
            'guardrail_agent': guardrail_agent,
            'context_agent': context_agent
        }
    
    async def setup_session_agents(self, session_id: str) -> bool:
        """Setup agents for a session."""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        # Create agents with MCP servers
        session.agents = self.create_agents_with_mcp(session.mcp_servers)
        return True
    
    async def close_session(self, session_id: str):
        """Manually close a session."""
        session = self.sessions.pop(session_id, None)
        if session:
            await self._close_session_resources(session)
            logger.info(f"Closed session: {session_id}")
    
    async def close(self):
        """Close the manager and all sessions."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        for session in self.sessions.values():
            await self._close_session_resources(session)
        
        self.sessions.clear()

# Global MCP manager
mcp_manager: Optional[OptimizedMCPManager] = None

# Pydantic models
class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None
    server_configs: Optional[List[str]] = []

class SessionCreateResponse(BaseModel):
    session_id: str
    user_id: str

class InterviewRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[dict]] = []

class InterviewResponse(BaseModel):
    reply: str
    session_id: str
    end_signal: Optional[str] = None

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global mcp_manager
    
    # Startup
    mcp_manager = OptimizedMCPManager(session_timeout_minutes=60)  # 2 hour sessions
    
    # Register example MCP server configurations
    # You can add your own MCP servers here
    mcp_manager.register_server_config('research_tools', {
        'type': 'stdio',
        'params': {
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-filesystem', '/tmp/research']
        },
        'tool_filter': {
            'type': 'static',
            'allowed_tools': ['read_file', 'write_file', 'list_directory'],
            'blocked_tools': ['delete_file']
        },
        'cache_tools': True
    })
    
    # Custom filter for political content analysis
    def political_content_filter(context, tool) -> bool:
        """Allow tools that help with political content analysis."""
        allowed_tools = [
            'analyze_sentiment', 'extract_topics', 'summarize_text',
            'read_file', 'write_file', 'list_directory'
        ]
        return tool.name in allowed_tools
    
    mcp_manager.register_server_config('political_analysis', {
        'type': 'stdio',
        'params': {
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-analysis', '/data/political']
        },
        'tool_filter': {
            'type': 'dynamic',
            'function': political_content_filter
        },
        'cache_tools': True
    })
    
    logger.info("MCP Manager initialized")
    
    yield
    
    # Shutdown
    if mcp_manager:
        await mcp_manager.close()
        logger.info("MCP Manager closed")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Political Views Interview Agent API with MCP", 
    version="2.0.0",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://duke.yul1.qualtrics.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_conversation_history(history: List[dict]) -> str:
    """Format conversation history for agents."""
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)

async def get_or_create_session(session_id: Optional[str] = None, user_id: Optional[str] = None) -> UserSession:
    """Get existing session or create new one."""
    if session_id:
        session = await mcp_manager.get_session(session_id)
        if session:
            return session
        else:
            raise HTTPException(status_code=404, detail="Session not found or expired")
    
    # Create new session
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
    
    new_session_id = await mcp_manager.create_session(
        user_id=user_id,
        server_configs=[]  # Can be extended to include MCP servers
    )
    
    # Setup agents for the session
    await mcp_manager.setup_session_agents(new_session_id)
    
    session = await mcp_manager.get_session(new_session_id)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to create session")
    
    return session

@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new interview session."""
    try:
        user_id = request.user_id or f"user_{uuid.uuid4().hex[:8]}"
        
        session_id = await mcp_manager.create_session(
            user_id=user_id,
            server_configs=request.server_configs or []
        )
        
        # Setup agents for the session
        await mcp_manager.setup_session_agents(session_id)
        
        return SessionCreateResponse(
            session_id=session_id,
            user_id=user_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.post("/chat", response_model=InterviewResponse)
async def chat_endpoint(request: InterviewRequest):
    """Main chat endpoint with session management."""
    try:
        # Get or create session
        session = await get_or_create_session(request.session_id)
        
        # Use session conversation history if not provided in request
        if not request.conversation_history and session.conversation_history:
            conversation_history = session.conversation_history
        else:
            conversation_history = request.conversation_history or []
        
        # Handle opening message
        if not conversation_history and request.message.lower() in ["hello", "hi", "start", "begin"]:
            opening_prompt = "Start the political interview by asking what political issues they are most passionate about and why those issues matter to them personally. Be warm and curious in your approach."
            result = await Runner.run(session.agents['interview_agent'], opening_prompt)
            
            response_content = result.final_output
            
            # Update session conversation history
            session.conversation_history.append({"role": "user", "content": request.message})
            session.conversation_history.append({"role": "assistant", "content": response_content})
            
            return InterviewResponse(
                reply=response_content,
                session_id=session.session_id,
                end_signal=None
            )
        
        # Format conversation history
        convo_history = ""
        if conversation_history:
            convo_history = format_conversation_history(conversation_history)
        
        # Run guardrail check
        guardrail_input = f"Political conversation:\n{convo_history}\n\nLatest input: {request.message}"
        guardrail_result = await Runner.run(session.agents['guardrail_agent'], guardrail_input)
        
        if "flag" in guardrail_result.final_output.lower():
            response_content = f"I appreciate your input, but let's keep our political discussion respectful and focused on understanding your views. {guardrail_result.final_output}"
            
            # Update session conversation history
            session.conversation_history.append({"role": "user", "content": request.message})
            session.conversation_history.append({"role": "assistant", "content": response_content})
            
            return InterviewResponse(
                reply=response_content,
                session_id=session.session_id,
                end_signal=None
            )
        
        # Process message with context if needed
        final_input = request.message
        if conversation_history:
            context_prompt = (
                f"Political conversation history:\n{convo_history}\n\n"
                f"User's new message: {request.message}\n\n"
                f"Rewrite this message with helpful context from the political discussion:"
            )
            context_result = await Runner.run(session.agents['context_agent'], context_prompt)
            final_input = context_result.final_output
        
        # Run main interview agent
        agent_input = (
            f"Political conversation so far:\n{convo_history}\n\n"
            f"User's latest response: {final_input}\n\n"
            f"As the political interviewer, respond to their answer and ask a thoughtful follow-up question that explores their political views deeper. "
            f"Focus on understanding the correlations between their different political positions and the underlying values that drive their beliefs. "
            f"Remember you are leading this political interview - always end with a question unless handing off to another agent."
        )
        
        result = await Runner.run(session.agents['interview_agent'], agent_input)
        response_content = result.final_output
        
        # Check for end signal
        end_signal = None
        if "CONCLUDE_INTERVIEW" in response_content:
            end_signal = "conclude"
        
        # Update session conversation history
        session.conversation_history.append({"role": "user", "content": request.message})
        session.conversation_history.append({"role": "assistant", "content": response_content})
        
        return InterviewResponse(
            reply=response_content,
            session_id=session.session_id,
            end_signal=end_signal
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    session = await mcp_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "conversation_history": session.conversation_history,
        "created_at": session.created_at.isoformat(),
        "last_accessed": session.last_accessed.isoformat()
    }

@app.delete("/sessions/{session_id}")
async def close_session_endpoint(session_id: str):
    """Close a specific session."""
    session = await mcp_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await mcp_manager.close_session(session_id)
    return {"message": f"Session {session_id} closed successfully"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions (for debugging)."""
    sessions = []
    for session_id, session in mcp_manager.sessions.items():
        sessions.append({
            "session_id": session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "conversation_count": len(session.conversation_history)
        })
    
    return {"active_sessions": sessions}

@app.post("/sessions/{session_id}/invalidate_cache")
async def invalidate_session_cache(session_id: str):
    """Invalidate MCP tools cache for a session."""
    session = await mcp_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    for server in session.mcp_servers:
        try:
            if hasattr(server, 'invalidate_tools_cache'):
                server.invalidate_tools_cache()
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    return {"message": f"Cache invalidated for session {session_id}"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "message": "Political Views Interview Agent API with MCP is running",
        "active_sessions": len(mcp_manager.sessions) if mcp_manager else 0,
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    uvicorn.run(app, host="localhost", port=8000)
