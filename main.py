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

    # --- Phase callbacks ---
    # Called automatically when one agent hands off to another.
    # They keep the phase number in the database in sync with where the interview actually is.

    def advance_phase(ctx: RunContextWrapper) -> None:
        # Move forward by one phase (caps at phase 4 — summary/completion handled separately)
        meta = _load_meta(session_id)
        if meta and meta["interview_phase"] < 4:
            new_phase = meta["interview_phase"] + 1
            _update_phase(session_id, new_phase)
            logger.info(f"Session {session_id} → phase {new_phase}")

    def enter_summary_reaction(ctx: RunContextWrapper) -> None:
        # Phase 5 = summary has been presented, waiting for the respondent's reaction
        _update_phase(session_id, 5)
        logger.info(f"Session {session_id} → awaiting summary reaction (phase 5)")

    def mark_complete(ctx: RunContextWrapper) -> None:
        # Phase 6 = interview fully complete
        _update_phase(session_id, 6)
        logger.info(f"Session {session_id} → complete (phase 6)")


    # =========================================================================
    # END INTERVIEW AGENT
    # Writes the final farewell message. Must append CONCLUDE_INTERVIEW at the
    # end — the server strips it from visible text and sends end_signal="conclude"
    # to Qualtrics, which then shows the Next button.
    # =========================================================================
    end_interview_agent = Agent(
        name="End Interview Agent",
        model="gpt-4o",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        You are closing a political research interview. The respondent has already received a summary
        of their views and has shared their reaction to it. Write a warm, genuine closing message that:
        - Thanks them sincerely for their time and thoughtful participation
        - Briefly acknowledges something specific from their reaction to the summary
        - Wishes them well

        Keep it concise — 2-3 sentences. Do not re-summarize their views.

        Always end your response with the token CONCLUDE_INTERVIEW on its own at the very end,
        with no punctuation after it. This signals the system to close the interview.
        """
    )

    # =========================================================================
    # SUMMARY AGENT
    # Two-turn agent: first presents the summary and asks for reaction (Job 1),
    # then on the next turn acknowledges the reaction and closes out (Job 2).
    # =========================================================================
    summary_agent = Agent(
        name="Summary Agent",
        model="gpt-4o",
        handoffs=[handoff(end_interview_agent, on_handoff=mark_complete)],
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        You are the Summary Agent. You have two distinct jobs depending on context:

        JOB 1 — PRESENT SUMMARY AND ASK FOR REACTION (when first activated):
        The interview's substantive phases are complete. Do the following in order:
        1. Present a clear, comprehensive summary of the respondent's political views covering:
           - Their political identity and what it means to them personally
           - The connections they drew between their identity and specific policy issues
           - Any tensions or nuances between their worldview and their specific issue positions
        2. Immediately after the summary, ask ONE warm, open-ended question inviting their reaction —
           for example: "How does that summary land with you — does it feel like an accurate picture
           of how you see yourself politically, or is there something you'd push back on?"
        3. Do NOT hand off yet. Stop and wait for the respondent to reply.

        JOB 2 — RECEIVE REACTION AND CLOSE (when respondent has replied to the summary):
        You will be called again after the respondent reacts. At this point:
        1. Acknowledge their reaction briefly and warmly (1 to 2 sentences only).
        2. Immediately hand off to the End Interview Agent. Do not ask more questions.

        HOW TO KNOW WHICH JOB TO DO:
        - Most recent message is a handoff/system message asking you to summarize → Job 1.
        - Most recent message is a respondent reply to your summary question → Job 2, hand off immediately.
        """
    )

    # =========================================================================
    # GUARDRAIL AGENT
    # Silently checks every respondent message before it reaches the interview agent.
    # Returns one of four tokens: CLEAR, REDIRECT, CLARIFY, or FLAG.
    # =========================================================================
    guardrail_agent = Agent(
        name="Guardrail Agent",
        model="gpt-4o",
        instructions="""
        You are a monitor for a qualitative political research interview. Evaluate every respondent
        message and respond with exactly one of the four tokens below — nothing else.

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

    # =========================================================================
    # PHASE TRANSITION AGENT
    # Evaluates whether the current phase is complete and decides what happens next.
    # Never speaks to the respondent directly — produces instructions for other agents.
    # Now handles FOUR substantive phases before handing off to Summary Agent.
    # =========================================================================
    phase_transition_agent = Agent(
        name="Phase Transition Agent",
        model="gpt-4o",
        handoffs=[
            handoff(summary_agent, on_handoff=enter_summary_reaction),
            handoff(end_interview_agent, on_handoff=mark_complete),
        ],
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        You evaluate interview progress and decide when to advance to the next phase. Read the full
        conversation history carefully before making any decision.

        THE INTERVIEW HAS FOUR SUBSTANTIVE PHASES:

        PHASE 1 — INTRODUCTION (2 questions):
        Goals: Establish warm rapport. Learn the respondent's overall feelings about politics. 
        This phase should feel like the natural opening of a
        conversation, not an interrogation. Allow 2 genuine exchanges before moving on. 
        The respondent should feel heard before the interview deepens.
        Transition when: The respondent has shared their general political feelings 
        AND the conversation feels naturally warmed up.
        Bridging into Phase 2: The first question in Phase 2 should arise organically from something
        the respondent just said. It should feel like genuine curiosity, not a gear-shift.

        PHASE 2 — POLITICAL IDENTITY AND MEANING:
        Goals: Understand what the respondent's political identity (liberal/moderate/conservative)
        means to them personally — their values, worldview, and what being that label actually
        represents in their life. This is about depth, not breadth. Allow multiple exchanges.
        Transition when: The respondent has articulated in their own words what their identity means
        to them, not just stated a label. They should have reflected on it, not just named it.

        PHASE 3 — CONNECTIONS BETWEEN IDENTITY AND ISSUES:
        Goals: Explore how the respondent's specific policy positions flow from or connect to their
        broader political identity. Draw on their pre-survey issue positions to ask targeted questions.
        Ask about multiple issues if time permits — look for patterns and connections they draw.
        Transition when: The respondent has explained how at least two or three of their issue
        positions relate to their identity or values, and the connections feel substantively explored.

        PHASE 4 — TENSIONS AND BROADER POLITICAL LANDSCAPE:
        Goals: Surface any tensions or inconsistencies between their stated identity and their
        specific issue stances using the pre-survey data. If no clear tensions exist, explore where
        they diverge from others who share their label. Close by asking how they situate themselves
        within the broader landscape of US politics today.
        Transition when: The respondent has reflected on tensions (real or hypothetical) AND has
        described how they see themselves in the broader political landscape.

        DECISION RULES:
        — If current phase goals are not yet met: hand off to Political Interview Agent with a
          suggested follow-up question that continues the current line of inquiry naturally.
        — If current phase goals are met (phases 1–3): hand off to Political Interview Agent with
          a bridging question that flows naturally into the next phase. The bridging question must
          feel like a genuine, curious follow-up — never announce a transition.
        — If Phase 4 goals are met: hand off to Summary Agent.
        — If the respondent wants to end early: hand off to End Interview Agent.
        — Never reveal phase names, agent names, transition language, or interview structure.
        — Never use "moving on", "let's shift", "turning to", "next topic", or any transition signal.
        — When handing off to Political Interview Agent, always include a concrete suggested question
          in your handoff message so the interview agent has something specific to work with.
        """
    )

    # =========================================================================
    # TOPIC TRANSITION AGENT
    # A more conservative gatekeeper for switching topics within a phase.
    # Requires genuine exhaustion of the current topic before moving on.
    # =========================================================================
    topic_transition_agent = Agent(
        name="Topic Transition Agent",
        model="gpt-4o",
        handoffs=[
            handoff(phase_transition_agent),
        ],
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

        You manage topic transitions within the current interview phase. Your default stance is
        conservative — you should only recommend moving to a new topic when the current one is
        genuinely exhausted, not merely because one or two exchanges have passed.

        WHEN TO STAY ON THE CURRENT TOPIC (recommend a follow-up question):
        - The respondent gave a brief or surface-level answer that could be probed deeper
        - There are natural follow-up angles that haven't been explored yet
        - The respondent touched on something interesting but the interviewer hasn't followed up
        - The current topic has only had 1 to 2 exchanges
        - There are tensions, specifics, or examples within this topic that haven't been drawn out

        WHEN TO MOVE TO A NEW TOPIC (suggest a bridging question):
        - The respondent has given a thorough, substantive answer that covered the main angles
        - The natural follow-up questions have been asked and answered
        - The conversation has genuinely run its course on this topic (typically 3 or more exchanges)
        - Pressing further on this topic would feel repetitive or pushy

        WHEN TO ESCALATE TO PHASE TRANSITION AGENT:
        - You believe the overall goals of the current phase are fully met
        - All major topics within the phase have been meaningfully explored

        HANDOFF INSTRUCTIONS:
        When handing back to the Political Interview Agent, always include a concrete suggested
        question in your handoff message. The question must:
        - Feel like a natural, organic continuation of the conversation
        - Never signal a topic change explicitly ("moving on", "let's talk about", "turning to", etc.)
        - Build on something the respondent actually said in their most recent response
        - Sound like genuine curiosity, not a scripted pivot
        """
    )

    # =========================================================================
    # POLITICAL INTERVIEW AGENT (main agent)
    # The only agent the respondent talks to. Uses pre-survey metadata to ask
    # informed, personalized questions across four phases.
    # =========================================================================
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
        The pre-survey background is at the start of this conversation. Before crafting EVERY question,
        scan their metadata and their latest response to enhance the goals and fluidity of the conversation. Never read metadata
        back verbatim — let it guide which connections and tensions you probe.

        THE INTERVIEW HAS FOUR PHASES:

        Phase 1 — Introduction (2 questions MAX):
        Open with genuine warmth. Ask about the respondent's overall thoughts and feelings toward
        politics in today's environment. Let the conversation breathe — this phase should feel like
        the start of a real conversation, not a checklist. Ask 2 questions here before moving on.
        Follow up naturally on what they say. Show genuine interest in their perspective.

        Phase 2 — Political Identity and Meaning:
        Using their ideology score from the pre-survey, explore what that identity means to them
        personally. Not just "I'm a liberal" but what that label actually represents — their values,
        what they feel political identity is for, what it means in their daily life and worldview.
        Probe with genuine curiosity and follow the thread of what they say. USE THE PRE-SURVEY

        Phase 3 — Connections Between Identity and Issues:
        Ask the respondent to reflect on how their specific policy positions connect to their broader
        political identity. Draw on their pre-survey issue stances to ask targeted, specific questions.
        Explore multiple issues where possible. Look for patterns in how they connect identity to policy. USE THE PRE-SURVEY

        Phase 4 — Tensions and Broader Political Landscape:
        Surface potential tensions between their stated identity and their specific issue stances using
        the pre-survey data. Ask them to reflect on these tensions. If no clear tensions exist, explore
        where they might diverge from others who share their identity. Close by asking how they see
        themselves within the broader landscape of US politics today. USE THE PRE-SURVEY

        AGENTS AND HANDOFFS:
        — Call Topic Transition Agent when the current topic feels explored but the phase is not done.
          Do this after at least 2 to 3 exchanges on a topic, not after just one answer.
        — Call Phase Transition Agent when you believe the goals of the current phase are fully met.
        — When you receive a handoff from a transition agent, they will suggest a bridging question —
          use it as written or adapt it slightly. The conversation must feel seamless and natural.

        INTERVIEW GUIDELINES:
        — Ask only ONE question at a time — never stack two questions in one message.
        — Ask open-ended questions, not yes/no questions.
        — Remain genuinely curious and non-judgmental throughout.
        — Never mention phases, transitions, agent names, or interview structure to the respondent.
        — Never use "moving on", "let's shift", "turning to", "for the next part", or any language
          that signals a structured transition. Every question should arise naturally from the conversation.
        — Aim for 15 to 20 questions total across all phases.
        — If the respondent seems to want to end early, respect that and call Phase Transition Agent.
        """
    )

    # Wire up return handoffs
    # (these are added after interview_agent is defined because it didn't exist yet above)
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
    """Returns agents for a session, creating them if they don't exist yet."""
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
    version="4.3.0",
    lifespan=lifespan
)

# Only the Duke Qualtrics domain is permitted to call this server.
# Update allow_origins if the survey URL changes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://duke.yul1.qualtrics.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest):
    """Called once at the start of each interview to create a session record."""
    user_id    = request.user_id or f"user_{uuid.uuid4().hex[:8]}"
    session_id = str(uuid.uuid4())
    user_metadata = request.user_metadata or {}

    _save_meta(session_id, user_id, user_metadata, phase=1)
    logger.info(f"Created session {session_id} for user {user_id}")

    return SessionCreateResponse(session_id=session_id, user_id=user_id)


@app.post("/chat", response_model=InterviewResponse)
async def chat_endpoint(request: InterviewRequest):
    """
    Main interview endpoint. Called on every respondent message.

    Routing logic:
      Kickoff ("hello" etc.)  → interview agent  (start the interview)
      Phase 5                 → summary agent    (receive reaction to summary, then close)
      CLARIFY from guardrail  → interview agent  (re-ask the question with clarification)
      REDIRECT from guardrail → interview agent  (gentle pivot back to interview topic)
      FLAG from guardrail     → return de-escalation message immediately
      All other phases        → interview agent  (continue the conversation)
    """
    session_id = request.session_id

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required. Call POST /sessions first.")

    meta = _load_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found.")

    agents      = get_or_create_agents(session_id)
    sdk_session = SQLiteSession(session_id, DB_PATH)
    is_kickoff  = request.message.lower().strip() in {"hello", "hi", "start", "begin"}

    # -------------------------------------------------------------------------
    # GUARDRAIL CHECK (every message except the initial kickoff)
    # Uses a separate session history so evaluations don't pollute the transcript.
    # -------------------------------------------------------------------------
    if not is_kickoff:
        guardrail_result = await Runner.run(
            agents["guardrail_agent"],
            f"Latest respondent input: {request.message}",
            session=SQLiteSession(f"{session_id}_guardrail", DB_PATH)
        )
        guardrail_output = guardrail_result.final_output.strip()
        guardrail_lower  = guardrail_output.lower()

        if guardrail_lower.startswith("clarify"):
            # Respondent is confused or asked for clarification.
            # Ask the interview agent to re-explain and re-ask the question warmly.
            clarify_result = await Runner.run(
                agents["interview_agent"],
                f"""The respondent seems confused or has asked for clarification about your last
                question. Do NOT move to a new topic. Instead:
                1. Briefly clarify what you were asking in simple, accessible language (1 sentence).
                2. Re-ask the same question in a slightly different, clearer way.
                Keep the tone warm and reassuring — make them feel comfortable, not tested.

                Their response was: {request.message}""",
                session=sdk_session
            )
            return InterviewResponse(
                reply=clarify_result.final_output.strip(),
                session_id=session_id,
                interview_phase=meta["interview_phase"]
            )

        elif guardrail_lower.startswith("redirect"):
            # Respondent went off-topic or asked about the survey structure.
            # Acknowledge briefly and pivot back naturally without flagging the detour.
            redirect_result = await Runner.run(
                agents["interview_agent"],
                f"""The respondent's message has drifted off-topic or is asking about the survey
                structure. Do NOT answer their off-topic question or acknowledge the survey structure.
                Instead, write one warm sentence that briefly acknowledges their comment without
                engaging with it, then immediately ask the next natural interview question to bring
                them back to the political conversation. The pivot must feel smooth and curious —
                never say "let's get back to" or "returning to our interview".

                Their message was: {request.message}""",
                session=sdk_session
            )
            return InterviewResponse(
                reply=redirect_result.final_output.strip(),
                session_id=session_id,
                interview_phase=meta["interview_phase"]
            )

        elif guardrail_lower.startswith("flag"):
            # Genuinely harmful content — return a calm de-escalation response.
            reply = guardrail_output.replace("FLAG:", "").replace("flag:", "").strip() or (
                "Let's keep our conversation respectful. Could you tell me more about your political views?"
            )
            return InterviewResponse(
                reply=reply,
                session_id=session_id,
                interview_phase=meta["interview_phase"]
            )

        # CLEAR → fall through to normal routing below.

    # -------------------------------------------------------------------------
    # ROUTING: pick the right agent and prompt for this turn
    # -------------------------------------------------------------------------

    if is_kickoff:
        # Format pre-survey answers as a readable list for the agent's context
        metadata_str = "\n".join(
            f"  - {k}: {v}" for k, v in meta["user_metadata"].items()
        ) or "  (No pre-survey data available)"

        agent_input = f"""Begin the interview in Phase 1 (Introduction). Introduce yourself warmly as
        an AI Conversation Bot here to learn about the respondent's political views and beliefs.
        Ask your first open-ended question about their overall thoughts and feelings toward politics
        in today's environment. Ask only this ONE question to open — keep it warm and inviting.
        Remember Phase 1 should have 2 to 3 exchanges before moving on, so pace yourself.

        PRE-SURVEY BACKGROUND ON THIS RESPONDENT — use this throughout the entire interview to ask
        informed, personalized questions. Scan it before each question. Do not read it back verbatim:
        {metadata_str}"""

        starting_agent = agents["interview_agent"]

    elif meta["interview_phase"] == 5:
        # Phase 5: summary was already presented last turn. This is the respondent's reaction.
        # Route directly to Summary Agent (Job 2) to acknowledge and close out.
        agent_input = f"""The respondent has just reacted to your summary. Their response:

        "{request.message}"

        Acknowledge their reaction briefly and warmly (1 to 2 sentences), then immediately
        hand off to the End Interview Agent. Do not ask more questions."""

        starting_agent = agents["summary_agent"]

    else:
        # Normal turn — continue the interview
        agent_input = f"""Respondent's latest response: {request.message}

        Before writing your next question, quickly check the pre-survey metadata in your history:
        (1) Does their response connect to or contradict a specific pre-survey position worth probing?
        (2) Is there a metadata detail not yet explored that could deepen the conversation?
        (3) Can you make the next question feel more personal by drawing on something specific?

        Respond thoughtfully. If the current topic still has more to explore, stay on it with a
        genuine follow-up — don't rush to a new topic after just one exchange. If you've genuinely
        exhausted the current topic (2 to 3 exchanges), call Topic Transition Agent. If the full
        phase goals are met, call Phase Transition Agent. Ask ONE open-ended question at a time."""

        starting_agent = agents["interview_agent"]

    # -------------------------------------------------------------------------
    # RUN THE AGENT
    # -------------------------------------------------------------------------
    result = await Runner.run(
        starting_agent,
        agent_input,
        session=sdk_session
    )
    response_content = result.final_output

    # Re-read the phase — a callback may have updated it during the agent run
    updated_meta  = _load_meta(session_id)
    current_phase = updated_meta["interview_phase"] if updated_meta else meta["interview_phase"]

    # Detect and strip the CONCLUDE_INTERVIEW token, convert to end_signal
    end_signal = None
    if "CONCLUDE_INTERVIEW" in response_content:
        end_signal = "conclude"
        response_content = response_content.replace("CONCLUDE_INTERVIEW", "").strip()

    # Clean up agent cache once interview is fully complete (phase 6)
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
    """Returns the full conversation transcript and metadata for a session."""
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
    """Permanently deletes all data for a session. Cannot be undone."""
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
    """Lists all sessions in the database (active and completed)."""
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
                "interview_phase": row[3],  # 6 = complete, 1-5 = in progress
            }
            for row in rows
        ]
    }


@app.get("/health")
async def health_check():
    """Status check — confirms the server is running."""
    con   = sqlite3.connect(DB_PATH)
    count = con.execute("SELECT COUNT(*) FROM session_meta").fetchone()[0]
    con.close()
    return {
        "status":          "healthy",
        "message":         "Political Belief Systems Interview Agent API is running",
        "active_sessions": count,
        "version":         "4.3.0",
    }


if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    uvicorn.run(app, host="localhost", port=8000)
