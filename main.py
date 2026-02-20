from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import logging
import os
import uuid
import json
import sqlite3

from agents import Agent, Runner, handoff, SQLiteSession, RunContextWrapper
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "interviews.db"

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
    interview_phase: int
    end_signal: Optional[str] = None


def _init_meta_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS session_meta (
            session_id TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            metadata   TEXT NOT NULL,
            phase      INTEGER NOT NULL DEFAULT 1
        )
    """)
    con.commit()
    con.close()

def _save_meta(session_id: str, user_id: str, user_metadata: dict, phase: int):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO session_meta (session_id, user_id, metadata, phase) VALUES (?,?,?,?)",
        (session_id, user_id, json.dumps(user_metadata), phase)
    )
    con.commit()
    con.close()

def _load_meta(session_id: str) -> Optional[Dict[str, Any]]:
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT user_id, metadata, phase FROM session_meta WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    con.close()
    if not row:
        return None
    return {"user_id": row[0], "user_metadata": json.loads(row[1]), "interview_phase": row[2]}

def _update_phase(session_id: str, phase: int):
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE session_meta SET phase = ? WHERE session_id = ?", (phase, session_id))
    con.commit()
    con.close()

def _delete_meta(session_id: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM session_meta WHERE session_id = ?", (session_id,))
    con.commit()
    con.close()


_agent_cache: Dict[str, Dict[str, Agent]] = {}


def create_agents(session_id: str) -> Dict[str, Agent]:

    def advance_phase(ctx: RunContextWrapper) -> None:
        meta = _load_meta(session_id)
        if meta and meta["interview_phase"] < 4:
            new_phase = meta["interview_phase"] + 1
            _update_phase(session_id, new_phase)
            logger.info(f"Session {session_id} → phase {new_phase}")

    def mark_complete(ctx: RunContextWrapper) -> None:
        _update_phase(session_id, 5)
        logger.info(f"Session {session_id} → complete")

    end_interview_agent = Agent(
        name="End Interview Agent",
        model="gpt-4o",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        Provide a thoughtful conclusion to the interview and thank the respondent for sharing their
        political views. Acknowledge the specific insights they shared about their political identity
        and beliefs throughout the conversation. If the respondent wants to end the interview early,
        acknowledge that gracefully and thank them for the time they gave.

        Always end your response with 'CONCLUDE_INTERVIEW' to signal the interview is complete.
        """
    )

    summary_agent = Agent(
        name="Summary Agent",
        model="gpt-4o",
        handoffs=[handoff(end_interview_agent, on_handoff=mark_complete)],
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        Summarize the respondent's political attitudes and beliefs as expressed throughout the interview.
        Provide a clear, comprehensive summary covering: their political identity and what it means to
        them personally, the connections they drew between their identity and specific policy issues,
        and any tensions or nuances they identified between their worldview and their issue positions.
        After providing the summary, hand off to the End Interview Agent to conclude.
        """
    )

    guardrail_agent = Agent(
        name="Guardrail Agent",
        model="gpt-4o",
        instructions="""
        You are a light-touch safety monitor for a political research interview. Your job is to flag
        only genuinely harmful content — not controversial opinions.

        Respond with 'CLEAR' in the vast majority of cases. Political opinions, even extreme or
        inflammatory ones, are expected and should never be flagged. Only respond with 'FLAG: [reason]'
        if the respondent is being personally abusive toward the interviewer, making credible threats,
        or sending content that is completely unrelated to politics (e.g. spam or explicit content).

        When in doubt, respond with 'CLEAR'.
        """
    )

    phase_transition_agent = Agent(
        name="Phase Transition Agent",
        model="gpt-4o",
        handoffs=[
            handoff(summary_agent, on_handoff=mark_complete),
            handoff(end_interview_agent, on_handoff=mark_complete),
        ],
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        You evaluate interview progress and determine when to move to the next phase. Analyze the full
        conversation history to assess whether the goals of the current phase have been met.

        PHASE 1 - INTRODUCTION:
        Goals: Open warmly and learn the respondent's overall feelings about politics and the issues
        most important to them.
        Transition when: The respondent has shared their general political feelings and named the issues
        they care most about.

        PHASE 2 - POLITICAL IDENTITY MEANING:
        Goals: Understand what the respondent's political identity (liberal/moderate/conservative) means
        to them personally — not just a label, but what values and worldview it reflects.
        Transition when: The respondent has articulated in their own words what their political identity
        represents to them.

        PHASE 3 - CONNECTIONS BETWEEN IDENTITY AND ISSUES:
        Goals: Explore how the respondent sees their specific policy stances as flowing from or
        connecting to their broader political identity.
        Transition when: The respondent has explained how at least some of their issue positions relate
        to their identity or worldview.

        PHASE 4 - TENSIONS BETWEEN IDENTITY AND ISSUES:
        Goals: Identify and explore any tensions or inconsistencies between their stated identity and
        their specific issue stances. Close by asking how they see themselves within the broader
        landscape of US politics.
        Transition when: The respondent has reflected on tensions and described how they situate
        themselves in broader US politics.

        DECISION RULES:
        If the current phase goals are met, hand off to Political Interview Agent with a suggested
        bridging question that flows naturally from what the respondent just said into the next topic.
        The bridging question must sound like a curious, natural follow-up — never announce that you
        are moving on or changing topics.
        If in Phase 4 and all goals are met, hand off to Summary Agent to conclude.
        If the respondent wants to end early, hand off to End Interview Agent.
        If current phase goals are not yet met, hand off to Political Interview Agent with a suggested
        follow-up question that continues the current line of inquiry naturally.
        Never reveal phase logic, agent names, transition language, or any meta-commentary about the
        interview structure to the respondent or in the handoff message.
        """
    )

    topic_transition_agent = Agent(
        name="Topic Transition Agent",
        model="gpt-4o",
        handoffs=[
            handoff(phase_transition_agent),
        ],
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        You help the interviewer move between topics within the current interview phase. Based on the
        conversation history, determine whether the current topic has been sufficiently explored and
        whether there are other relevant topics to cover within this phase.

        When handing off back to the Political Interview Agent, provide a suggested question that flows
        naturally from what the respondent just said into the new topic. The question must feel like an
        organic follow-up, not a change of subject. Never use phrases like "moving on", "let's shift",
        "turning to", or any language that signals a topic or phase change to the respondent.

        If the current topic is not yet exhausted, hand off to Political Interview Agent with a
        suggested follow-up question that continues the current line of inquiry. If the overall phase
        goals appear complete, hand off to Phase Transition Agent.
        """
    )

    interview_agent = Agent(
        name="Political Interview Agent",
        model="gpt-4o",
        handoffs=[
            handoff(topic_transition_agent),
            handoff(phase_transition_agent),
        ],
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        You are conducting a thoughtful qualitative interview about the respondent's political belief
        systems. Your goal is to understand how they see the relationships between their political
        identity (liberal/moderate/conservative) and their stances on specific policy issues. The
        pre-survey background at the start of this conversation contains their ideology score and issue
        positions — use this throughout to ask informed, specific questions. Never read it back verbatim;
        let it guide which connections and tensions you probe.

        THE INTERVIEW HAS FOUR PHASES:

        Phase 1 — Introduction: Open by asking the respondent about their overall thoughts and feelings
        toward politics and which issues matter most to them. Keep this brief — 1 to 2 exchanges —
        before moving on.

        Phase 2 — Political Identity Meaning: Using their ideology score from the pre-survey, ask them
        to elaborate on what that identity means to them personally. What does it mean to be a
        liberal, moderate, or conservative in their view?

        Phase 3 — Connections Between Identity and Issues: Ask the respondent to reflect on how their
        specific policy positions connect to their broader political identity. Draw on their pre-survey
        issue stances to ask targeted follow-up questions.

        Phase 4 — Tensions Between Identity and Issues: Identify potential tensions between their
        stated identity and their specific issue stances using the pre-survey data. Ask them to reflect
        on these. If no clear tensions exist, explore where they might diverge from others who share
        their identity. Close this phase by asking how they see themselves within the broader landscape
        of US politics.

        AGENTS AND HANDOFFS:
        Call Topic Transition Agent when the current political topic has been sufficiently explored and
        you want to move to a new topic within the same phase. Call Phase Transition Agent when you
        believe the goals of the current phase are fully met.

        When you receive a handoff back from either transition agent, they will suggest a bridging
        question — use it as your next question exactly as suggested or adapt it slightly. Never
        announce to the respondent that you are transitioning, moving on, or shifting topics. The
        conversation must always feel like a natural, flowing dialogue.

        CRITICAL: Never say anything like "moving on to the next topic", "let's shift to", "turning to
        another area", "for the next part of our conversation", or any phrase that signals a structured
        transition. Simply ask the next question as if it arose naturally from the conversation.

        INTERVIEW GUIDELINES:
        Ask only one question at a time. Ask open-ended questions, not yes/no questions. Remain
        non-judgmental and focus on understanding, not debating. Always lead with a question — do not
        wait for the respondent to start. Use the pre-survey metadata to ask informed, specific
        questions that draw on their stated issue positions. Never mention phases, transitions, or
        agent names to the respondent. Conclude after 15 to 20 questions or if the respondent wishes
        to end early.
        """
    )

    phase_transition_agent.handoffs.append(
        handoff(interview_agent, on_handoff=advance_phase)
    )
    topic_transition_agent.handoffs.append(handoff(interview_agent))

    return {
        "interview_agent":        interview_agent,
        "phase_transition_agent": phase_transition_agent,
        "topic_transition_agent": topic_transition_agent,
        "summary_agent":          summary_agent,
        "end_interview_agent":    end_interview_agent,
        "guardrail_agent":        guardrail_agent,
    }


def get_or_create_agents(session_id: str) -> Dict[str, Agent]:
    if session_id not in _agent_cache:
        _agent_cache[session_id] = create_agents(session_id)
    return _agent_cache[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_meta_db()
    logger.info("Political Interview API started — DB initialised")
    yield
    logger.info("Political Interview API shutting down")


app = FastAPI(
    title="Political Belief Systems Interview Agent API",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://duke.yul1.qualtrics.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest):
    user_id = request.user_id or f"user_{uuid.uuid4().hex[:8]}"
    session_id = str(uuid.uuid4())
    user_metadata = request.user_metadata or {}

    _save_meta(session_id, user_id, user_metadata, phase=1)
    logger.info(f"Created session {session_id} for user {user_id}")

    return SessionCreateResponse(session_id=session_id, user_id=user_id)


@app.post("/chat", response_model=InterviewResponse)
async def chat_endpoint(request: InterviewRequest):
    session_id = request.session_id

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required. Call POST /sessions first.")

    meta = _load_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found. It may have been deleted or the server restarted before this session was created.")

    agents      = get_or_create_agents(session_id)
    sdk_session = SQLiteSession(session_id, DB_PATH)
    is_kickoff  = request.message.lower().strip() in {"hello", "hi", "start", "begin"}

    if not is_kickoff:
        guardrail_result = await Runner.run(
            agents["guardrail_agent"],
            f"Latest respondent input: {request.message}",
            session=SQLiteSession(f"{session_id}_guardrail", DB_PATH)
        )
        if "flag" in guardrail_result.final_output.lower():
            reply = guardrail_result.final_output.replace("FLAG:", "").strip() or (
                "Let's keep our conversation respectful. Could you tell me more about your political views?"
            )
            return InterviewResponse(
                reply=reply,
                session_id=session_id,
                interview_phase=meta["interview_phase"]
            )

    if is_kickoff:
        metadata_str = "\n".join(
            f"  - {k}: {v}" for k, v in meta["user_metadata"].items()
        ) or "  (No pre-survey data available)"

        agent_input = f"""Begin the interview in Phase 1 (Introduction). Introduce yourself warmly as
        an AI Conversation Bot here to learn about the respondent's political views and beliefs. Your
        opening question should ask about their overall thoughts and feelings toward politics and which
        political issues matter most to them. Ask only this ONE open-ended question to open the
        conversation.

        PRE-SURVEY BACKGROUND ON THIS RESPONDENT — use this throughout the entire interview to ask
        informed, targeted questions. Reference their specific issue positions and ideology naturally;
        do not read it back verbatim:
        {metadata_str}"""

    else:
        agent_input = f"""Respondent's latest response: {request.message}

        Respond thoughtfully and ask a follow-up question. The pre-survey background shared at the
        start of this conversation is in your history — use it to inform the direction and depth of
        your questions. If you have sufficiently explored the current topic, hand off to Topic
        Transition Agent. If you believe the current phase goals are fully met, hand off to Phase
        Transition Agent. Ask ONE open-ended question at a time and remain non-judgmental."""

    result = await Runner.run(
        agents["interview_agent"],
        agent_input,
        session=sdk_session
    )
    response_content = result.final_output

    updated_meta  = _load_meta(session_id)
    current_phase = updated_meta["interview_phase"] if updated_meta else meta["interview_phase"]

    end_signal = None
    if "CONCLUDE_INTERVIEW" in response_content:
        end_signal = "conclude"
        response_content = response_content.replace("CONCLUDE_INTERVIEW", "").strip()

    if current_phase == 5 and session_id in _agent_cache:
        del _agent_cache[session_id]

    return InterviewResponse(
        reply=response_content,
        session_id=session_id,
        interview_phase=current_phase,
        end_signal=end_signal
    )


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    meta = _load_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")

    sdk_session = SQLiteSession(session_id, DB_PATH)
    history     = await sdk_session.get_items()

    return {
        "session_id":           session_id,
        "conversation_history": history,
        "user_metadata":        meta["user_metadata"],
        "interview_phase":      meta["interview_phase"],
    }


@app.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    meta = _load_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")

    sdk_session = SQLiteSession(session_id, DB_PATH)
    await sdk_session.clear_session()
    _delete_meta(session_id)
    _agent_cache.pop(session_id, None)

    return {"message": f"Session {session_id} closed successfully"}


@app.get("/sessions")
async def list_sessions():
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT session_id, user_id, metadata, phase FROM session_meta"
    ).fetchall()
    con.close()

    return {
        "active_sessions": [
            {
                "session_id":      row[0],
                "user_id":         row[1],
                "user_metadata":   json.loads(row[2]),
                "interview_phase": row[3],
            }
            for row in rows
        ]
    }


@app.get("/health")
async def health_check():
    con   = sqlite3.connect(DB_PATH)
    count = con.execute("SELECT COUNT(*) FROM session_meta").fetchone()[0]
    con.close()
    return {
        "status":          "healthy",
        "message":         "Political Belief Systems Interview Agent API is running",
        "active_sessions": count,
        "version":         "4.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    uvicorn.run(app, host="localhost", port=8000)
