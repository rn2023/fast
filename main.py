from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import asyncio
import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from agents import Agent, Runner, handoff


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    #holds session data for each user
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    agents: Dict[str, Agent]
    conversation_history: List[dict]
    user_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, timeout_minutes: int = 60) -> bool:
        #checks if session timed out
        return datetime.now() - self.last_accessed > timedelta(minutes=timeout_minutes)
    
    def update_access_time(self):
        #bumps the last accessed timestamp
        self.last_accessed = datetime.now()

class OptimizedSessionManager:
    #manages all active user sessions
    
    def __init__(self, session_timeout_minutes: int = 60):
        self.sessions: Dict[str, UserSession] = {}
        self.session_timeout_minutes = session_timeout_minutes
        self._cleanup_task = None
        
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        #spins up background task to clean old sessions
        async def cleanup_expired_sessions():
            while True:
                try:
                    await asyncio.sleep(300)  #check every 5 minutes
                    await self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    
    async def _cleanup_expired_sessions(self):
        #removes sessions that timed out
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired(self.session_timeout_minutes)
        ]
        
        for session_id in expired_sessions:
            session = self.sessions.pop(session_id)
            logger.info(f"Cleaned up expired session: {session_id}")
    
    async def create_session(
        self, 
        user_id: str,
        user_metadata: Dict[str, Any] = None
    ) -> str:
        #creates new session with unique id
        session_id = str(uuid.uuid4())
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            agents={},
            conversation_history=[],
            user_metadata=user_metadata or {}
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {user_id} with metadata: {user_metadata}")
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        #fetches session and refreshes timeout
        session = self.sessions.get(session_id)
        if session and not session.is_expired(self.session_timeout_minutes):
            session.update_access_time()
            return session
        elif session:
            del self.sessions[session_id]
        return None
    
    def create_agents(self, user_metadata: Dict[str, Any] = None) -> Dict[str, Agent]:
        #builds the full agent network for interviewing
        
        #formats user political data for agent context
        metadata_context = ""
        if user_metadata:
            metadata_context = "\n\nUser Information:\n" + "\n".join(
                f"- {key}: {value}" for key, value in user_metadata.items()
            )
        
        #agent that wraps up the interview with summary
        summary_agent = Agent(
            name="Summary Agent", 
            instructions=(
                "Summarize the user's political views, key positions, and the correlations between different political ideologies and issue stances they discussed. "
                "Provide a clear, comprehensive summary of their political beliefs and how they connect their various positions. "
                "After providing the summary, hand off to the final_question_agent to ask one last reflective question."
            ),
            model="gpt-4o"
        )

        #asks one deep final question
        final_question_agent = Agent(
            name="Final Question Agent",
            instructions=(
                "Ask one final thoughtful question about their political views to help them reflect on their overall political ideology or how their beliefs fall into the traditional 2 party ideology system. "
                "Make it open-ended and deeply introspective about their political stances. "
                "After they respond to this final question, hand off to the end_interview_agent to conclude the interview."
            ),
            model="gpt-4o"
        )

        #closes out the conversation
        end_interview_agent = Agent(
            name="End Interview Agent", 
            instructions=(
                "Provide a thoughtful conclusion to the political interview. Thank them for sharing their political views and offer any final reflections on the complexity of political beliefs. "
                "If the user want to end the interview, acknowledge that and formally close out the session."
                "Always end your response with 'CONCLUDE_INTERVIEW' to signal the interview is complete."
            ),
            model="gpt-4o"
        )

        #monitors for inappropriate content
        guardrail_agent = Agent(
            name="Guardrail Agent",
            instructions=(
                "Monitor political conversation for safety and appropriateness. "
                "Ensure the discussion remains respectful and focused on understanding views rather than debate. "
                "If the conversation becomes hostile or unsafe, respond with 'FLAG: [reason]' and ask a clarifying question to redirect the conversation constructively. "
                "Otherwise, respond with 'CLEAR' to indicate the conversation is appropriate."
                "Do not flag content that is related to political opinions, no matter how controversial as long as the conversation remains relevant and somewhat respectful."
            ),
            model="gpt-4o"
        )

        #adds context from previous messages to new user input
        context_agent = Agent(
            name="Context Agent",
            instructions=(
                "Given a political conversation history and a new user message, rewrite the message with useful context from the previous political discussion. "
                "Output only the rewritten message, making it clearer and more specific based on the political topics that have been discussed."
                f"The user was gauged on their political stances in relation to liberals, conservatives, and moderates with the percentages of agreement, and they are as follows: {metadata_context} The ideology is their self-described ideology 1-5 very liberal to very conservative. This is infromation to inform the interview.\n"
            ),
            model="gpt-4o"
        )

        #decides when to switch interview phases
        transition_agent = Agent(
            name="Transition Agent",
            instructions=(
                "Analyze the political conversation and decide when to transition topics. "
                "If the conversation is ready for summarization of their political views, hand off to summary_agent. "
                "If it's time for a final reflection question about their political beliefs, hand off to final_question_agent. "
                "If the political interview should end, hand off to end_interview_agent. "
                "If the conversation should continue with the main interview, hand off back to interview_agent with guidance on what to explore next."
            ),
            model="gpt-4o",
            handoffs=[
                handoff(summary_agent),
                handoff(final_question_agent),
                handoff(end_interview_agent)
            ]
        )

        #main interviewer agent that asks questions
        interview_agent = Agent(
            name="Political Interview Agent",
            instructions=(
                "You are conducting a thoughtful interview about political ideologies and the participants stances on various political issues. "
                "Your goal is to deeply understand their political ideology, especially those of political moderates by questioning them on their issue stances. "
                f"The user was gauged on their political stances in relation to liberals, conservatives, and moderates with the percentages of agreement, and they are as follows: {metadata_context} The ideology is their self-described ideology 1-5 very liberal to very conservative. This is infromation to inform the interview, especially the first line of questioning.\n"
                "ALWAYS lead with questions - never wait for the user to start. "
                "Start the interview by introducing yourself as a curious political interviewer interested in understanding their political issue stances and ideologies as well as the issues that they care most about that inform their political ideology classification. "
                "Move on to by asking why they identify with a specific ideology or as a political moderate and what political issues contribute to that identity. Say that we want to explore multiple political issues to understand their views better. "
                "Dig into the issues that they believe in most strongly impact their political party and ideology and what those issues and their solutions are. "
                "Once they share their passionate issues, dive deep into if they are moderates due to varying issue stance that align with both parties or moderate stances on most or all issues. "
                "Conduct this interview in a curious, non-judgmental manner - your goal is to understand their political views, not to debate or challenge them. "
                "The interview should focus on exploring the correlation between their stances on political issues and the political ideology they identify with. "
                "You should ask follow-up questions to dig deeper into their reasoning and the values that underpin their political beliefs. "
                "The questions should be concise yet thought-provoking, aiming to uncover the nuances of their political ideology with short and understandable language. "
                "Try to avoid yes/no questions and tackle one issue at a time to fully explore their views. MAKE THE INTERVIEW SIMPLE FOR THE INTERVIEWEE TO UNDERSTAND. ONE QUESTION AT A TIME! "
                "ASK ONLY ONE SIMPLE QUESTION AT A TIME. "
                "After the first couple questions, start to explore their views on different but specific political issues and what that reveals about their overall political ideology. "
                "After exploring a political issue stance thoroughly (usually 2-3 follow-ups), hand off to the transition_agent to decide whether to continue exploring or move toward conclusion. "
                "Explore potential tensions or consistencies in their political worldview especially when it comes to issues that break with their political ideologies. Make the questions simple and easy to understand. "
                "Ask about how they prioritize different political issues when they might conflict with the stances of the party or ideology that they most align with. "
                "Guide the conversation to understand the true nature of their political stance as a moderate - is it a mix of issue stances, a centrist approach, or something else? "
                "Remain neutral and curious - your goal is understanding, not debating or challenging. "
                "Always end your responses with a question unless concluding the interview. "
                "After 4-6 substantial exchanges about their political views, hand off to the transition_agent to assess progress. "
                "After 15-20 questions total, hand off to transition_agent to move toward conclusion."
            ),
            model="gpt-4o",
            handoffs=[
                handoff(transition_agent)
            ]
        )

        #wire up the handoff chain between agents
        transition_agent.handoffs.append(handoff(interview_agent))
        summary_agent.handoffs = [handoff(final_question_agent)]
        final_question_agent.handoffs = [handoff(end_interview_agent)]
        
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
        #initializes agents for a specific session
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.agents = self.create_agents(session.user_metadata)
        return True
    
    async def close_session(self, session_id: str):
        #manually terminates a session
        session = self.sessions.pop(session_id, None)
        if session:
            logger.info(f"Closed session: {session_id}")
    
    async def close(self):
        #shuts down manager and all sessions
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        self.sessions.clear()


session_manager: Optional[OptimizedSessionManager] = None


class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None
    user_metadata: Optional[Dict[str, Any]] = None

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    #handles startup and shutdown for the api
    global session_manager
    
    session_manager = OptimizedSessionManager(session_timeout_minutes=60)
    
    logger.info("Session Manager initialized")
    
    yield
    
    if session_manager:
        await session_manager.close()
        logger.info("Session Manager closed")

app = FastAPI(
    title="Political Views Interview Agent API", 
    version="2.0.0",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware

#allows requests from qualtrics survey platform
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://duke.yul1.qualtrics.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_conversation_history(history: List[dict]) -> str:
    #converts message list to readable text
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)

async def get_or_create_session(
    session_id: Optional[str] = None, 
    user_id: Optional[str] = None,
    user_metadata: Optional[Dict[str, Any]] = None
) -> UserSession:
    #retrieves existing session or makes new one
    if session_id:
        session = await session_manager.get_session(session_id)
        if session:
            return session
        else:
            raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
    
    new_session_id = await session_manager.create_session(
        user_id=user_id,
        user_metadata=user_metadata
    )
    
    await session_manager.setup_session_agents(new_session_id)
    
    session = await session_manager.get_session(new_session_id)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to create session")
    
    return session

@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest):
    #endpoint to start new interview session
    try:
        user_id = request.user_id or f"user_{uuid.uuid4().hex[:8]}"
        
        session_id = await session_manager.create_session(
            user_id=user_id,
            user_metadata=request.user_metadata
        )
        
        await session_manager.setup_session_agents(session_id)
        
        return SessionCreateResponse(
            session_id=session_id,
            user_id=user_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.post("/chat", response_model=InterviewResponse)
async def chat_endpoint(request: InterviewRequest):
    #main endpoint for sending messages to interview agent
    try:
        #get or create session
        session = await get_or_create_session(request.session_id)
        
        #use stored history if none provided
        if not request.conversation_history and session.conversation_history:
            conversation_history = session.conversation_history
        else:
            conversation_history = request.conversation_history or []
        
        #handle interview kickoff
        if not conversation_history and request.message.lower() in ["hello", "hi", "start", "begin"]:
            opening_prompt = (
                "Start the political interview by introducing yourself and asking what their political party and ideology is - "
                "whether more liberal, moderate, or conservative - and what issues they are most passionate about that informs their ideology. "
                "Be warm and curious in your approach."
            )
            #runner handles agent execution and handoffs
            result = await Runner.run(session.agents['interview_agent'], opening_prompt)
            
            response_content = result.final_output
            
            #save to history
            session.conversation_history.append({"role": "user", "content": request.message})
            session.conversation_history.append({"role": "assistant", "content": response_content})
            
            #check if interview ended
            end_signal = "conclude" if "CONCLUDE_INTERVIEW" in response_content else None
            
            return InterviewResponse(
                reply=response_content,
                session_id=session.session_id,
                end_signal=end_signal
            )
        
        #build conversation context
        convo_history = ""
        if conversation_history:
            convo_history = format_conversation_history(conversation_history)
        
        #check message safety
        guardrail_input = f"Political conversation:\n{convo_history}\n\nLatest input: {request.message}"
        guardrail_result = await Runner.run(session.agents['guardrail_agent'], guardrail_input)
        
        if "flag" in guardrail_result.final_output.lower():
            #flagged content - redirect conversation
            response_content = guardrail_result.final_output.replace("FLAG:", "").strip()
            if not response_content:
                response_content = "I appreciate your input, but let's keep our political discussion respectful and focused on understanding your views."
            
            session.conversation_history.append({"role": "user", "content": request.message})
            session.conversation_history.append({"role": "assistant", "content": response_content})
            
            return InterviewResponse(
                reply=response_content,
                session_id=session.session_id,
                end_signal=None
            )
        
        #add context to user message
        final_input = request.message
        if conversation_history:
            context_prompt = (
                f"Political conversation history:\n{convo_history}\n\n"
                f"User's new message: {request.message}\n\n"
                f"Rewrite this message with helpful context from the political discussion:"
            )
            context_result = await Runner.run(session.agents['context_agent'], context_prompt)
            final_input = context_result.final_output
        
        #run main interview agent with full context
        agent_input = (
            f"Political conversation so far:\n{convo_history}\n\n"
            f"User's latest response: {final_input}\n\n"
            f"As the political interviewer, respond to their answer and ask a thoughtful follow-up question that explores their political views deeper. "
            f"Focus on understanding the correlations between their different political positions and the underlying values that drive their beliefs. "
            f"Remember you are leading this political interview - always end with a question unless handing off to another agent."
        )
        
        #runner automatically handles agent handoffs
        result = await Runner.run(session.agents['interview_agent'], agent_input)
        response_content = result.final_output
        
        #check for conclusion signal
        end_signal = None
        if "CONCLUDE_INTERVIEW" in response_content:
            end_signal = "conclude"
        
        #save exchange to history
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
    #fetches full conversation history for a session
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "conversation_history": session.conversation_history,
        "created_at": session.created_at.isoformat(),
        "last_accessed": session.last_accessed.isoformat(),
        "user_metadata": session.user_metadata
    }

@app.delete("/sessions/{session_id}")
async def close_session_endpoint(session_id: str):
    #endpoint to manually close a session
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await session_manager.close_session(session_id)
    return {"message": f"Session {session_id} closed successfully"}

@app.get("/sessions")
async def list_sessions():
    #lists all active sessions for debugging
    sessions = []
    for session_id, session in session_manager.sessions.items():
        sessions.append({
            "session_id": session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "conversation_count": len(session.conversation_history),
            "user_metadata": session.user_metadata
        })
    
    return {"active_sessions": sessions}

@app.get("/health")
async def health_check():
    #simple endpoint to check if api is running
    return {
        "status": "healthy", 
        "message": "Political Views Interview Agent API is running",
        "active_sessions": len(session_manager.sessions) if session_manager else 0,
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    uvicorn.run(app, host="localhost", port=8000)
