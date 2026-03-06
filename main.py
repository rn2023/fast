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
        if meta and meta["interview_phase"] < 3:
            new_phase = meta["interview_phase"] + 1
            _update_phase(session_id, new_phase)
            logger.info(f"Session {session_id} → phase {new_phase}")

    def enter_summary_reaction(ctx: RunContextWrapper) -> None:
        # Phase 4 = summary presented, waiting for user reaction
        _update_phase(session_id, 4)
        logger.info(f"Session {session_id} → awaiting summary reaction (phase 4)")

    def mark_complete(ctx: RunContextWrapper) -> None:
        # Phase 5 = interview fully complete
        _update_phase(session_id, 5)
        logger.info(f"Session {session_id} → complete (phase 5)")

    end_interview_agent = Agent(
        name="End Interview Agent",
        model="gpt-4o",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        You are closing a political research interview. The respondent has already received a summary
        of their views and has shared their reaction to it. Your job is to write a warm, genuine
        closing message that:
        - Thanks them sincerely for their time and thoughtful participation
        - Briefly acknowledges something specific from their reaction to the summary
        - Wishes them well

        Keep it concise — 3 to 5 sentences. Do not re-summarize their views again.

        Always end your response with the token CONCLUDE_INTERVIEW on its own at the very end,
        with no punctuation after it. This token signals the system to close the interview.
        """
    )

    summary_agent = Agent(
        name="Summary Agent",
        model="gpt-4o",
        handoffs=[handoff(end_interview_agent, on_handoff=mark_complete)],
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        You are the Summary Agent. You have two distinct jobs depending on the context:

        JOB 1 — PRESENT SUMMARY AND ASK FOR REACTION (when first activated):
        You will be called when the interview's substantive phases are complete. Do the following:
        1. Present a clear, comprehensive summary of the respondent's political views covering:
           - Their political identity and what it means to them personally
           - The connections they drew between their identity and specific policy issues
           - Any tensions or nuances between their worldview and their specific issue positions
        2. Immediately after the summary, ask ONE open-ended question inviting their reaction:
           something like "How does that summary land with you — does it feel like an accurate
           picture of how you see yourself politically?" or similar. The question should feel
           warm and curious, not clinical.
        3. Do NOT hand off yet. Wait for the respondent to reply.

        JOB 2 — RECEIVE REACTION AND HAND OFF TO END INTERVIEW AGENT (when respondent has replied):
        You will be called again after the respondent has reacted to the summary. At this point:
        1. Acknowledge their reaction briefly and warmly (1 to 2 sentences maximum).
        2. Immediately hand off to the End Interview Agent to close the conversation.
        Do not ask any more questions. Do not re-summarize.

        HOW TO KNOW WHICH JOB TO DO:
        - If the most recent message in your context is a system/handoff message asking you to
          summarize, you are in Job 1.
        - If the most recent message is a respondent reply to your summary question, you are in Job 2.
          Hand off to End Interview Agent immediately after a brief acknowledgment.
        """
    )

    guardrail_agent = Agent(
        name="Guardrail Agent",
        model="gpt-4o",
        instructions="""
        You are a monitor for a qualitative political research interview with two jobs: maintaining
        interview integrity and catching harmful content. Evaluate every respondent message and
        respond with exactly one of the three tokens below — nothing else.

        CATEGORY 1 - CLEAR:
        The message is on-topic for a political interview. Use this in the vast majority of cases.
        On-topic includes: any political opinion, policy stance, ideological view, personal
        experience with politics, commentary on politicians or parties, frustration with politics,
        contradictory or nuanced views, or any elaboration on their beliefs. Be very generous —
        even venting about politics or politicians is CLEAR.

        CATEGORY 2 - REDIRECT:
        The message has drifted away from political topics but is not harmful. Use REDIRECT for:
        - Questions or comments about the survey structure, question design, research purpose,
          who built this tool, how the AI works, or how this interview is conducted
        - Any attempt to discuss the phases, transitions, or logic of the interview itself
        - Completely off-topic personal conversation unrelated to politics (weather, unrelated
          stories, casual chatter)
        - Requests to skip questions or change the subject without genuine political content
        - General confusion about what they are supposed to be doing
        Important: frustration WITH politics or politicians is NOT a redirect — it is on-topic.
        Only use REDIRECT if the respondent has genuinely left the political domain or is
        probing the interview structure.

        CATEGORY 3 - FLAG:
        The message contains genuinely harmful content. Use FLAG: [brief reason] only for:
        - Personal abuse or harassment directed at the interviewer
        - Credible threats of violence
        - Explicit sexual content
        - Pure spam or completely incoherent content with zero political relevance
        Political anger, profanity about politicians, and extreme opinions are never flagged.

        YOUR ENTIRE RESPONSE must be exactly one of these three forms with no other text:
          CLEAR
          REDIRECT
          FLAG: [brief reason]
        """
    )

    phase_transition_agent = Agent(
        name="Phase Transition Agent",
        model="gpt-4o",
        handoffs=[
            handoff(summary_agent, on_handoff=enter_summary_reaction),
            handoff(end_interview_agent, on_handoff=mark_complete),
        ],
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        You evaluate interview progress and determine when to move to the next phase. Analyze the full
        conversation history to assess whether the goals of the current phase have been met.

        THE INTERVIEW HAS THREE PHASES:

        PHASE 1 — INTRODUCTION:
        Goals: Open warmly and learn the respondent's overall feelings about politics and the issues
        most important to them. This phase should be brief — just 1 to 2 exchanges — enough to
        establish rapport and identify the issues they care about. Do not linger here.
        Transition when: The respondent has shared any general feelings about politics and named at
        least one issue they care about. Even a brief or partial answer is sufficient to move on.
        The transition into Phase 2 should feel completely seamless — the bridging question should
        flow naturally from whatever they just said and move gently toward what their political
        identity means to them personally.

        PHASE 2 — POLITICAL IDENTITY AND ISSUE CONNECTIONS:
        Goals: First, understand what the respondent's political identity (liberal/moderate/conservative)
        means to them personally — their values and worldview, not just a label. Then, explore how
        their specific policy stances flow from or connect to that identity. Draw on their pre-survey
        issue positions to ask targeted follow-up questions. This is the heart of the interview.
        Transition when: The respondent has articulated what their identity means to them AND has
        explained how at least some of their issue positions relate to that identity or worldview.

        PHASE 3 — TENSIONS AND BROADER POLITICAL LANDSCAPE:
        Goals: Identify and explore any tensions or inconsistencies between their stated identity and
        their specific issue stances using the pre-survey data. Ask them to reflect on these. If no
        clear tensions exist, explore where they might diverge from others who share their identity.
        Close by asking how they see themselves within the broader landscape of US politics.
        Transition when: The respondent has reflected on tensions and described how they situate
        themselves in broader US politics. Then hand off to Summary Agent.

        DECISION RULES:
        — If current phase goals are met, hand off to Political Interview Agent with a suggested
          bridging question that flows naturally from what the respondent just said into the next topic.
          The bridging question must sound like a curious, natural follow-up — never announce you are
          moving on or changing topics.
        — If in Phase 3 and all goals are met, hand off to Summary Agent to conclude.
        — If the respondent wants to end early, hand off to End Interview Agent.
        — If current phase goals are not yet met, hand off to Political Interview Agent with a
          suggested follow-up question that continues the current line of inquiry naturally.
        — Never reveal phase logic, agent names, transition language, or any meta-commentary about
          the interview structure to the respondent or in the handoff message.
        — Never use phrases like "moving on", "let's shift", "turning to", "for the next part", or
          any language that signals a structured transition to the respondent.
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
        identity (liberal/moderate/conservative) and their stances on specific policy issues.

        PRE-SURVEY METADATA USAGE — CRITICAL:
        The pre-survey background is provided at the start of this conversation. Before crafting
        EVERY question, briefly scan their metadata and their most recent response to sharpen or frame your next question. Never read
        metadata back verbatim — let it guide which connections and tensions you probe.

        THE INTERVIEW HAS THREE PHASES:

        Phase 1 — Introduction (BRIEF — 1 to 2 exchanges only):
        Open by asking the respondent about their overall thoughts and feelings toward politics.
        Keep this very short before moving on to Phase 2. Do not probe deeply here.

        Phase 2 — Political Identity and Issue Connections:
        Using their ideology score from the pre-survey, ask them to elaborate on what that identity
        means to them personally. What does it mean to be a liberal, moderate, or conservative in their
        view? Then explore how their specific policy positions connect to their broader political
        identity. Draw on their pre-survey issue stances to ask targeted, specific follow-up questions.
        This is the substantive core of the interview.

        Phase 3 — Tensions and Broader Political Landscape:
        Identify potential tensions between their stated identity and their specific issue stances using
        the pre-survey data. Ask them to reflect on these tensions. If no clear tensions exist, explore
        where they might diverge from others who share their identity. Close this phase by asking how
        they see themselves within the broader landscape of US politics.

        AGENTS AND HANDOFFS:
        — Call Topic Transition Agent when the current political topic has been sufficiently explored
          and you want to move to a new topic within the same phase.
        — Call Phase Transition Agent when you believe the goals of the current phase are fully met.
        — When you receive a handoff back from either transition agent, they will suggest a bridging
          question — use it exactly as suggested or adapt it only slightly.

        INTERVIEW GUIDELINES:
        — Ask only ONE question at a time.
        — Ask open-ended questions, not yes/no questions.
        — Remain non-judgmental and focus on understanding, not debating.
        — Always lead with a question — do not wait for the respondent to start.
        — Never mention phases, transitions, or agent names to the respondent.
        — Never say anything like "moving on to the next topic", "let's shift to", "turning to another
          area", "for the next part of our conversation", or any phrase that signals a structured
          transition. Simply ask the next question as if it arose naturally from the conversation.
        — Conclude after 15 to 20 questions or if the respondent wishes to end early.
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
    version="4.2.0",
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
        guardrail_output = guardrail_result.final_output.strip()
        guardrail_lower  = guardrail_output.lower()

        if guardrail_lower.startswith("redirect"):
            # Off-topic or meta/structural probe — re-engage with a bridging question
            redirect_result = await Runner.run(
                agents["interview_agent"],
                f"""The respondent just sent a message that is off-topic or asks about the survey
                structure rather than engaging with the interview questions. Do NOT answer their
                off-topic question or acknowledge the survey structure. Instead, write a single,
                warm sentence that gently acknowledges their comment without engaging with it,
                immediately followed by the next natural interview question that brings them back
                to the political topic at hand. The transition must feel natural — do not say
                anything like "let's get back to" or "returning to our interview". Just pivot
                smoothly with curiosity.

                Their off-topic message was: {request.message}""",
                session=sdk_session
            )
            return InterviewResponse(
                reply=redirect_result.final_output.strip(),
                session_id=session_id,
                interview_phase=meta["interview_phase"]
            )

        elif guardrail_lower.startswith("flag"):
            reply = guardrail_output.replace("FLAG:", "").replace("flag:", "").strip() or (
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
        opening question should ask about their overall thoughts and feelings toward politics. Ask only
        this ONE open-ended question to open the conversation.

        PRE-SURVEY BACKGROUND ON THIS RESPONDENT — reference this throughout the ENTIRE interview.
        Before each question you ask, scan this data and the respondent's latest response to see if
        you can make the next question more specific, personal, or probing based on their stated
        positions. Do not read this back verbatim; use it to shape your questions naturally:
        {metadata_str}"""

        starting_agent = agents["interview_agent"]

    elif meta["interview_phase"] == 4:
        # Summary was already presented; this is the respondent's reaction.
        # Route directly to Summary Agent so it can acknowledge and hand off to End Interview Agent.
        agent_input = f"""The respondent has just reacted to your summary. Here is their response:

        "{request.message}"

        Acknowledge their reaction briefly and warmly (1 to 2 sentences), then immediately hand off
        to the End Interview Agent to close the interview. Do not ask any more questions."""

        starting_agent = agents["summary_agent"]

    else:
        agent_input = f"""Respondent's latest response: {request.message}

        Before formulating your next question, briefly check: (1) Does their response connect to or
        contradict any specific position in their pre-survey metadata? (2) Is there a metadata detail
        you haven't yet explored that could deepen the conversation? (3) Can you make the next question
        feel more tailored by referencing something specific from their background?

        Respond thoughtfully and ask a follow-up question informed by both their latest response and
        their pre-survey background (in your conversation history). If you have sufficiently explored
        the current topic, hand off to Topic Transition Agent. If you believe the current phase goals
        are fully met, hand off to Phase Transition Agent. Ask ONE open-ended question at a time and
        remain non-judgmental."""

        starting_agent = agents["interview_agent"]

    result = await Runner.run(
        starting_agent,
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
        "version":         "4.2.0",
    }


if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    uvicorn.run(app, host="localhost", port=8000)
