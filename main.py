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

PHASE_GUARDRAIL_CONTEXT = {
    1: "The interviewer just asked about the respondent's overall feelings toward politics.",
    2: "The interviewer is exploring what the respondent's political identity (liberal/moderate/conservative) means to them personally.",
    3: "The interviewer is asking how the respondent's specific policy positions connect to their broader political identity.",
    4: "The interviewer is probing tensions between the respondent's ideology label and specific issue stances.",
    5: "The interviewer has presented a summary and asked for the respondent's reaction.",
}


def _build_guardrail_agent(phase: int) -> Agent:
    """Built fresh per request — phase-aware, stateless, fast."""
    phase_ctx = PHASE_GUARDRAIL_CONTEXT.get(phase, PHASE_GUARDRAIL_CONTEXT[1])
    return Agent(
        name="Guardrail Agent",
        model="gpt-4o-mini",
        instructions=f"""
        You are a monitor for a qualitative political research interview. Evaluate the respondent
        message and respond with exactly one of the four tokens below — nothing else.

        CURRENT INTERVIEW CONTEXT: {phase_ctx}

        CATEGORY 1 - CLEAR:
        The message is on-topic and the respondent understands what is being asked. Use this in the
        vast majority of cases. On-topic includes: any political opinion, policy stance, ideological
        view, personal experience with politics, commentary on politicians or parties, frustration
        with politics, contradictory or nuanced views. Be very generous — even venting about politics
        or politicians is CLEAR.

        CATEGORY 2 - CLARIFY:
        The respondent seems confused about the question being asked, has misunderstood it, or is
        explicitly asking for clarification before they can answer. Use CLARIFY when:
        - They say things like "I'm not sure what you mean", "can you explain that?", "what do you
          mean by that?", "I don't understand the question", or similar
        - Their response does not engage with the question at all in a way that suggests confusion
          rather than a deliberate non-answer
        - They answer a clearly different question than the one asked, suggesting a misread
        Do NOT use CLARIFY just because the respondent gave a short or vague answer — that is CLEAR.
        Only use CLARIFY when there is genuine evidence of confusion or a direct request for
        clarification.

        CATEGORY 3 - REDIRECT:
        The respondent has left the political interview domain but is not harmful. Use REDIRECT for:
        - Questions or comments about the survey structure, question design, research purpose,
          who built this tool, how the AI works, or how this interview is conducted
        - Any attempt to discuss the phases, transitions, or logic of the interview itself
        - Completely off-topic personal conversation unrelated to politics (weather, unrelated
          stories, casual chatter with no political relevance)
        - Requests to skip questions or change the subject without genuine political content
        Note: frustration WITH politics or politicians is NOT a redirect — it is on-topic.

        CATEGORY 4 - FLAG:
        Genuinely harmful content only. Use FLAG: [brief reason] for:
        - Personal abuse or harassment directed at the interviewer
        - Credible threats of violence
        - Explicit sexual content
        - Pure spam or completely incoherent content with zero political relevance
        Political anger, profanity about politicians, and extreme opinions are never flagged.

        YOUR ENTIRE RESPONSE must be exactly one of these four forms with no other text:
          CLEAR
          CLARIFY
          REDIRECT
          FLAG: [brief reason]
        """
    )


def create_agents(session_id: str) -> Dict[str, Agent]:

    last_respondent_msg: dict = {"text": "Please continue the interview."}

    def advance_phase(ctx: RunContextWrapper) -> None:
        meta = _load_meta(session_id)
        if meta and meta["interview_phase"] < 4:
            new_phase = meta["interview_phase"] + 1
            _update_phase(session_id, new_phase)
            logger.info(f"Session {session_id} → phase {new_phase}")

    def enter_summary_reaction(ctx: RunContextWrapper) -> None:
        _update_phase(session_id, 5)
        logger.info(f"Session {session_id} → awaiting summary reaction (phase 5)")

    def mark_complete(ctx: RunContextWrapper) -> None:
        _update_phase(session_id, 6)
        logger.info(f"Session {session_id} → complete (phase 6)")

    end_interview_agent = Agent(
        name="End Interview Agent",
        model="gpt-4o",
        instructions="""
        Wrap up the interview in a warm, friendly way. Thank the person for sharing their thoughts
        on politics with you. Mention one or two specific things they said that stood out. Keep it
        brief and genuine. If they wanted to end early, thank them for the time they gave.

        Always end your response with 'CONCLUDE_INTERVIEW' to signal the interview is complete.
        """
    )

    summary_agent = Agent(
        name="Summary Agent",
        model="gpt-4o",
        handoffs=[handoff(end_interview_agent, on_handoff=mark_complete)],
        instructions="""
        You are the Summary Agent. You have two distinct jobs depending on context:

        JOB 1 — PRESENT SUMMARY AND ASK FOR REACTION (when first activated):
        The interview's main questions are done. Do the following in order:
        1. Give a plain, easy-to-follow summary of what the person said about their political views:
           - How they see their own political identity and what it means to them
           - How they connected that identity to specific issues they care about
           - Any places where their views didn't fit neatly into their label
        2. Right after the summary, ask ONE simple, friendly question — something like:
           "Does that feel like a fair picture of how you see yourself politically, or is there
           anything you'd want to add or change?"
        3. Do NOT hand off yet. Wait for them to reply.

        JOB 2 — RECEIVE REACTION AND CLOSE (when respondent has replied to the summary):
        1. Acknowledge what they said in 1 to 2 short sentences.
        2. Hand off to the End Interview Agent right away. No more questions.

        HOW TO KNOW WHICH JOB TO DO:
        - If the most recent message is asking you to summarize → Job 1.
        - If the most recent message is the person's reaction to your summary → Job 2, hand off immediately.
        """
    )

    phase_transition_agent = Agent(
        name="Phase Transition Agent",
        model="gpt-4o",
        handoffs=[
            handoff(summary_agent, on_handoff=enter_summary_reaction),
            handoff(end_interview_agent, on_handoff=mark_complete),
        ],
        instructions="""
        You decide whether the current phase of the interview is truly done and whether it is time
        to move on. Read the full conversation history carefully before making any decision.

        PHASE 1 — OPENING:
        Goal: The person has shared how they generally feel about politics and mentioned at least one
        or two issues they care about.
        Minimum before advancing: At least 3 back-and-forth exchanges in this phase.
        Move on when: They have answered the opening question and given at least one follow-up answer
        about what matters to them politically.
        Bridging to Phase 2: Connect what they just said to how they see themselves politically.
        Use their ideology score from the pre-survey. Keep the bridge question simple and personal —
        for example: "You mentioned [what they said] — would you say you're more on the
        conservative side, liberal side, or somewhere in the middle when it comes to politics?"
        Do not use those exact words; make it feel natural.

        PHASE 2 — WHAT THEIR POLITICAL IDENTITY MEANS TO THEM:
        Goal: The person has explained IN THEIR OWN WORDS what their political label means to them —
        not just picked a label, but said something about the values or beliefs behind it.
        Minimum before advancing: At least 3 back-and-forth exchanges specifically about what their
        identity means. A passing comment during Phase 1 does NOT count.
        HARD RULE — DO NOT ADVANCE PAST PHASE 2 UNLESS:
          (a) The person was directly asked what their political identity means to them personally, AND
          (b) They gave a real answer about their values or worldview, not just a one-word label, AND
          (c) There have been at least 3 exchanges in this phase.
        If any of those three conditions are not met, hand back to the Interview Agent to keep going.

        PHASE 3 — CONNECTING IDENTITY TO SPECIFIC ISSUES:
        Goal: The person has talked about how at least 3 specific issues from the pre-survey connect
        to their broader political identity.
        Minimum before advancing: At least 3 different issues discussed, each with at least 2
        exchanges. Do not advance if fewer than 3 issues have been genuinely explored.
        Move on when: 3 or more issues have been connected to their identity in a meaningful way.

        PHASE 4 — TENSIONS AND CONTRADICTIONS:
        Goal: The person has reflected on at least one place where their specific views don't
        perfectly match their political label. They have also said something about where they fit
        in the bigger picture of US politics.
        Minimum before advancing: At least 3 exchanges probing tensions, plus at least one answer
        about how they see themselves in the broader political landscape.
        Move on when: Both of those goals are met.

        DECISION RULES:
        - You may only move forward ONE phase at a time. Never skip.
        - Count the exchanges in each phase carefully before deciding.
        - If goals and minimums are met → hand off to Interview Agent to ask the bridging question.
        - If in Phase 4 and all goals are met → hand off to Summary Agent.
        - If the person wants to stop early → hand off to End Interview Agent.
        - If goals are NOT fully met → hand off back to Interview Agent to keep going.

        OUTPUT RULE — STRICTLY ENFORCED: When handing off to the Political Interview Agent,
        your message must be ONLY the last respondent's message verbatim, prefixed with
        "Respondent's latest response: ". Nothing else — no reasoning, no phase summaries,
        no suggested questions, no "Based on...", no "I suggest asking:", no markdown.
        Example: "Respondent's latest response: i try to be in the middle and make my own decisions"
        """
    )

    topic_transition_agent = Agent(
        name="Topic Transition Agent",
        model="gpt-4o",
        handoffs=[
            handoff(phase_transition_agent),
        ],
        instructions="""
        You help move the conversation to a new topic within the current phase. Look at the full
        conversation history and figure out which pre-survey issues have already been talked about
        and which ones haven't been touched yet.

        Always move to an issue that hasn't been covered yet. Pick one that connects naturally to
        what the person just said. Frame the next question so it feels like a natural follow-up,
        not a sudden change of subject. Never say things like "moving on" or "next question" or
        "let's talk about something else."

        If the current topic still has more to explore, hand off back to the Interview Agent.
        If all the phase goals look complete, hand off to the Phase Transition Agent instead.

        OUTPUT RULE — STRICTLY ENFORCED: When handing off to the Political Interview Agent,
        your message must be ONLY the last respondent's message verbatim, prefixed with
        "Respondent's latest response: ". Nothing else — no reasoning, no topic summaries,
        no suggested questions, no markdown.
        Example: "Respondent's latest response: i try to be in the middle and make my own decisions"
        """
    )

    interview_agent = Agent(
        name="Political Interview Agent",
        model="gpt-4o",
        handoffs=[
            handoff(topic_transition_agent),
            handoff(phase_transition_agent),
        ],
        instructions="""
        You are having a friendly conversation with someone about their political views. Your job is
        to understand how they see the connection between their political identity
        (conservative, liberal, moderate) and the specific issues they care about. You have their
        pre-survey answers at the start of this conversation — use those to ask informed, specific
        questions. Never read the pre-survey back to them word for word.

        KEEP YOUR QUESTIONS SIMPLE:
        Use plain, everyday language. Avoid academic or formal phrasing. Ask questions the way a
        curious, friendly person would in a normal conversation. Short questions are better than
        long ones. One question at a time — always.

        THE FOUR PHASES OF THE INTERVIEW:

        Phase 1 — Opening (2 to 3 exchanges):
        Ask how they generally feel about politics these days. What issues matter most to them?
        Keep it casual and open. After 2 to 3 exchanges, hand off to Phase Transition Agent.

        Phase 2 — What does their political identity mean to them? (3 to 4 exchanges):
        This phase MUST happen. Do not skip it or rush through it.
        Ask them directly: what does it mean to them to be [their ideology from pre-survey]?
        What values or beliefs come with that for them personally?
        Stay in this phase for at least 3 exchanges. Dig into what the label actually means to them
        — not just what party they vote for, but what they believe in.
        Only hand off to Phase Transition Agent after at least 3 real exchanges on this topic.

        Phase 3 — Connecting identity to specific issues (4 to 6 exchanges, at least 3 issues):
        Ask how their specific views on issues connect to how they see themselves politically.
        You MUST cover at least 3 different issues from the pre-survey — one per topic cycle.
        Start with the issue that most clearly lines up with their ideology, then work through
        others. Keep track of which issues you've already asked about and don't repeat them.
        Only hand off to Phase Transition Agent after at least 3 issues have been genuinely explored.

        Phase 4 — Tensions and contradictions (3 to 4 exchanges):
        Look at the pre-survey for places where their specific positions don't quite match their
        identity label. Ask about the clearest one you can find. If everything lines up, ask where
        they might see themselves as different from others with the same label.
        End this phase by asking how they see themselves in the bigger picture of US politics today.
        Only hand off after at least 3 exchanges AND after they've answered that broader question.

        WHEN TO HAND OFF:
        Hand off to Topic Transition Agent when the current issue/topic has been well covered (at
        least 2 exchanges on it) and you want to move to a new one within the same phase.
        Hand off to Phase Transition Agent when you genuinely believe ALL the goals for the current
        phase have been met AND the minimum number of exchanges has happened.
        Do NOT hand off after just one or two answers. Stay in the phase until it's truly done.

        NATURALNESS:
        - After they answer, briefly reflect back one specific thing they said before asking your
          next question. One short sentence — just enough to show you heard them.
          Example: "That makes sense given what you said about [topic] — [next question]?"
        - Vary how you start questions. Don't begin every question the same way. Mix it up:
          "What do you think about...", "How does that fit with...", "Where does that come from
          for you?", "Does that feel like it connects to...", "I'm curious — do you think..."
        - If they say something personal or emotional, acknowledge it briefly before moving on.

        IMPORTANT RULES:
        - Never tell them you are transitioning, moving to a new phase, or switching topics.
        - Never mention phases, agents, or the interview structure.
        - Always ask open-ended questions, not yes/no questions.
        - Stay warm, curious, and non-judgmental throughout.
        - Never output internal reasoning or phase notes — only speak directly to the respondent.
        - Wrap up after about 15 to 20 questions total, or sooner if they want to stop.
        """
    )

    def advance_phase_and_relay(ctx: RunContextWrapper) -> None:
        advance_phase(ctx)

    def _make_relay_handoff(target_agent, on_handoff_fn=None):
        h = handoff(target_agent, on_handoff=on_handoff_fn) if on_handoff_fn else handoff(target_agent)
        import types as _types
        h.get_transfer_message = _types.MethodType(
            lambda self, agent: f"Respondent's latest response: {last_respondent_msg['text']}",
            h
        )
        return h

    phase_transition_agent.handoffs.append(
        _make_relay_handoff(interview_agent, advance_phase_and_relay)
    )
    topic_transition_agent.handoffs.append(_make_relay_handoff(interview_agent))

    return {
        "interview_agent":        interview_agent,
        "phase_transition_agent": phase_transition_agent,
        "topic_transition_agent": topic_transition_agent,
        "summary_agent":          summary_agent,
        "end_interview_agent":    end_interview_agent,
        "_last_respondent_msg":   last_respondent_msg,
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
    version="4.14.0",
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

    agents        = get_or_create_agents(session_id)
    sdk_session   = SQLiteSession(session_id, DB_PATH)
    current_phase = meta["interview_phase"]
    is_kickoff    = request.message.lower().strip() in {"hello", "hi", "start", "begin"}

    if not is_kickoff:
        guardrail_agent  = _build_guardrail_agent(current_phase)
        guardrail_result = await Runner.run(
            guardrail_agent,
            f"Latest respondent input: {request.message}",
            session=SQLiteSession(f"{session_id}_guardrail", DB_PATH)
        )
        guardrail_output = guardrail_result.final_output.strip()
        guardrail_lower  = guardrail_output.lower()
        logger.info(f"Guardrail [phase {current_phase}] → {guardrail_output!r}")

        if guardrail_lower.startswith("clarify"):
            clarify_result = await Runner.run(
                agents["interview_agent"],
                f"""The respondent seems confused or has asked for clarification about your last
                question. Do NOT move to a new topic. Instead:
                1. Briefly clarify what you were asking in simple, plain language (1 sentence).
                2. Re-ask the same question in a simpler, clearer way.
                Keep the tone warm and easy — make them feel comfortable, not tested.
                Current phase context: {PHASE_GUARDRAIL_CONTEXT.get(current_phase, '')}

                Their response was: {request.message}""",
                session=sdk_session
            )
            return InterviewResponse(
                reply=clarify_result.final_output.strip(),
                session_id=session_id,
                interview_phase=current_phase
            )

        elif guardrail_lower.startswith("redirect"):
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
                Current phase context: {PHASE_GUARDRAIL_CONTEXT.get(current_phase, '')}

                Their off-topic message was: {request.message}""",
                session=sdk_session
            )
            return InterviewResponse(
                reply=redirect_result.final_output.strip(),
                session_id=session_id,
                interview_phase=current_phase
            )

        elif guardrail_lower.startswith("flag"):
            reply = guardrail_output.replace("FLAG:", "").replace("flag:", "").strip() or (
                "Let's keep our conversation respectful. Could you tell me more about your political views?"
            )
            return InterviewResponse(
                reply=reply,
                session_id=session_id,
                interview_phase=current_phase
            )

    if is_kickoff:
        metadata_str = "\n".join(
            f"  - {k}: {v}" for k, v in meta["user_metadata"].items()
        ) or "  (No pre-survey data available)"

        agent_input = f"""Begin the interview in Phase 1 (Opening). Introduce yourself briefly and
        warmly as an AI here to learn about the respondent's political views. Keep the intro to one
        or two sentences. Then ask ONE simple, open-ended question about how they generally feel
        about politics these days. Do not ask more than one question.

        PRE-SURVEY BACKGROUND ON THIS RESPONDENT — use this throughout the entire interview to ask
        informed, targeted questions. Reference their specific issue positions and ideology naturally;
        do not read it back verbatim:
        {metadata_str}"""

        starting_agent = agents["interview_agent"]

    elif current_phase == 5:
        agent_input = f"""The respondent has just reacted to your summary. Here is their response:

        "{request.message}"

        Acknowledge their reaction briefly and warmly (1 to 2 sentences), then immediately hand off
        to the End Interview Agent to close the interview. Do not ask any more questions."""

        starting_agent = agents["summary_agent"]

    else:
        agents["_last_respondent_msg"]["text"] = request.message

        agent_input = f"""Respondent's latest response: {request.message}

        Respond in a warm, friendly way and ask one simple follow-up question. Use the pre-survey
        background from earlier in this conversation to guide what you ask about. Keep your language
        plain and conversational — no formal or academic phrasing.
        If you have covered the current topic well (at least 2 exchanges on it), hand off to Topic
        Transition Agent to move to a new topic within this phase.
        If you believe ALL the goals for the current phase are fully met AND the minimum number of
        exchanges has happened, hand off to Phase Transition Agent.
        Do NOT hand off too early — stay in the phase until it is genuinely complete."""

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

    if current_phase == 6 and session_id in _agent_cache:
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
        "version":         "4.14.0",
    }


if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    uvicorn.run(app, host="localhost", port=8000)
