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
    interview_phase: int = 1
    phase_exchange_count: int = 0 

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
            user_metadata=user_metadata or {},
            interview_phase=1,
            phase_exchange_count=0
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
            metadata_context = "\n".join(
                f"- {key}: {value}" for key, value in user_metadata.items()
            )

        #agent that wraps up the interview with summary
        summary_agent = Agent(
            name="Summary Agent",
            instructions=(
                "Summarize the respondent's political attitudes and beliefs as expressed in the interview. "
                "Provide a clear, comprehensive summary of their political identity, how they understand it, "
                "the connections they draw between their identity and specific policy issues, "
                "and any tensions or nuances they identified. "
                "After providing the summary, hand off to the end_interview_agent to conclude."
            ),
            model="gpt-4o"
        )

        #closes out the conversation
        end_interview_agent = Agent(
            name="End Interview Agent",
            instructions=(
                "Provide a thoughtful conclusion to the interview and thank the respondent for sharing their political views. "
                "Acknowledge the insights they shared about their political identity and beliefs. "
                "If the respondent wants to end the interview early, acknowledge that gracefully. "
                "Always end your response with 'CONCLUDE_INTERVIEW' to signal the interview is complete."
            ),
            model="gpt-4o"
        )

        #monitors for inappropriate content
        guardrail_agent = Agent(
            name="Guardrail Agent",
            instructions=(
                "Monitor political conversation for safety and appropriateness. "
                "Ensure the discussion remains respectful and focused on understanding the respondent's political views. "
                "If the conversation becomes hostile or unsafe, respond with 'FLAG: [reason]' and ask a clarifying question to redirect the conversation constructively. "
                "Otherwise, respond with 'CLEAR' to indicate the conversation is appropriate. "
                "Do not flag content that is related to political opinions, no matter how controversial, as long as the conversation remains on-topic and respectful."
            ),
            model="gpt-4o"
        )

        #retrieves relevant user data based on conversation
        context_agent = Agent(
            name="Context Agent",
            instructions=(
                "You provide relevant background information about the respondent to help guide the interview. "
                f"The respondent completed a pre-survey with the following information:\n{metadata_context}\n\n"
                "When asked, identify which pieces of this pre-survey data are most relevant to the current topic of discussion. "
                "Output ONLY the relevant data points that would help the interviewer ask more informed questions. "
            ),
            model="gpt-4o"
        )

        #decides when to switch interview phases
        phase_transition_agent = Agent(
            name="Phase Transition Agent",
            instructions=(
                "You manage transitions between the four phases of this political interview. "
                "Analyze the conversation to determine which phase we are in and whether the goals for that phase have been met.\n\n"
                "PHASE 1 - INTRODUCTION / ICE-BREAKING:\n"
                "Goals: Establish rapport, make respondent comfortable, learn basic background.\n"
                "Transition when: Respondent seems at ease and has shared some background about themselves.\n\n"
                "PHASE 2 - POLITICAL IDENTITY MEANING:\n"
                "Goals: Understand what the respondent's political identity (liberal/moderate/conservative) means to them personally.\n"
                "Transition when: Respondent has articulated their understanding of their political identity and what it represents to them.\n\n"
                "PHASE 3 - CONNECTIONS BETWEEN IDENTITY AND ISSUES:\n"
                "Goals: Explore how the respondent sees their policy stances connecting to their broader political identity.\n"
                "Transition when: Respondent has explained how at least some of their issue positions relate to their identity/worldview.\n\n"
                "PHASE 4 - TENSIONS BETWEEN IDENTITY AND ISSUES:\n"
                "Goals: Identify and explore any tensions or inconsistencies between stated identity and issue stances.\n"
                "Transition when: Respondent has reflected on potential tensions, or confirmed their views are consistent and explained why.\n\n"
                "DECISION RULES:\n"
                "- If the current phase goals have been met, hand off to interview_agent with clear guidance to move to the next phase.\n"
                "- If in Phase 4 and goals are met, hand off to summary_agent to conclude.\n"
                "- If the respondent wants to end early, hand off to end_interview_agent.\n"
                "- If the current phase goals are NOT yet met, hand off back to interview_agent with guidance on what still needs to be explored.\n"
                "- Always provide clear guidance about the current phase number and what to focus on next."
            ),
            model="gpt-4o",
            handoffs=[
                handoff(summary_agent),
                handoff(end_interview_agent)
            ]
        )

        #decides when to shift topics within a phase
        topic_transition_agent = Agent(
            name="Topic Transition Agent",
            instructions=(
                "You help the interviewer transition smoothly between different topics within the current interview phase. "
                "Based on the conversation history, determine whether:\n"
                "1. The current topic has been sufficiently explored\n"
                "2. There are other relevant topics to explore within this phase\n"
                "3. Whether additional context from the pre-survey would be helpful\n\n"
                "When transitioning topics:\n"
                "- Hand off to context_agent to retrieve relevant pre-survey data about the new topic\n"
                "- Then hand off to interview_agent with guidance on what topic to explore next\n\n"
                "If the current topic is not yet exhausted, hand off to interview_agent with guidance to continue the current line of inquiry.\n\n"
                "If it seems like the overall phase goals are complete, hand off to phase_transition_agent to assess whether to move to the next phase."
            ),
            model="gpt-4o",
            handoffs=[
                handoff(context_agent),
                handoff(phase_transition_agent)
            ]
        )

        #main interviewer agent that asks questions
        interview_agent = Agent(
            name="Political Interview Agent",
            instructions=(
                "You are conducting a thoughtful interview about the respondent's political belief systems. "
                "Your goal is to understand how they see the relationships between their political identity and their policy stances.\n\n"
                "INTERVIEW GUIDELINES:\n"
                "- Ask ONE question at a time\n"
                "- Ask open-ended questions, NOT yes/no questions\n"
                "- Remain non-judgmental - focus on understanding, not debating\n"
                "- Always lead with questions - do not wait for the respondent to start\n"
                "- Build naturally on what the respondent shares\n"
                "- When you need additional background information about the respondent to ask a more informed question, hand off to context_agent\n"
                "- When you feel you've sufficiently explored the current topic, hand off to topic_transition_agent to assess next steps\n"
                "- Always end your responses with a question unless concluding the interview"
            ),
            model="gpt-4o",
            handoffs=[
                handoff(context_agent),
                handoff(topic_transition_agent)
            ]
        )

        #wire up the handoff chains
        phase_transition_agent.handoffs.append(handoff(interview_agent))
        topic_transition_agent.handoffs.append(handoff(interview_agent))
        context_agent.handoffs = [handoff(interview_agent)]
        summary_agent.handoffs = [handoff(end_interview_agent)]

        return {
            'interview_agent': interview_agent,
            'phase_transition_agent': phase_transition_agent,
            'topic_transition_agent': topic_transition_agent,
            'summary_agent': summary_agent,
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
    title="Political Belief Systems Interview Agent API",
    version="3.0.0",
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
                "Begin the interview in Phase 1 (Introduction). "
                "Introduce yourself warmly as an interviewer interested in understanding their political beliefs. "
                "Establish rapport before moving into the substantive interview. "
                "Remember: ask ONE question at a time and keep it open-ended."
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
                response_content = "I appreciate your input, but let's keep our discussion respectful and focused on understanding your views."

            session.conversation_history.append({"role": "user", "content": request.message})
            session.conversation_history.append({"role": "assistant", "content": response_content})

            return InterviewResponse(
                reply=response_content,
                session_id=session.session_id,
                end_signal=None
            )

        #run main interview agent with full context
        agent_input = (
            f"Conversation so far:\n{convo_history}\n\n"
            f"Respondent's latest response: {request.message}\n\n"
            f"Current phase: {session.interview_phase}\n\n"
            f"As the interviewer, respond thoughtfully to their answer and ask a follow-up question. "
            f"If you need more context about the respondent's background to ask an informed question, hand off to context_agent. "
            f"If you feel you've sufficiently explored the current topic, hand off to topic_transition_agent. "
            f"Remember: ask ONE open-ended question at a time, remain non-judgmental, and focus on understanding their views."
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
        "user_metadata": session.user_metadata,
        "interview_phase": session.interview_phase
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
            "user_metadata": session.user_metadata,
            "interview_phase": session.interview_phase
        })

    return {"active_sessions": sessions}

@app.get("/health")
async def health_check():
    #simple endpoint to check if api is running
    return {
        "status": "healthy",
        "message": "Political Belief Systems Interview Agent API is running",
        "active_sessions": len(session_manager.sessions) if session_manager else 0,
        "version": "3.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")

    uvicorn.run(app, host="localhost", port=8000)
