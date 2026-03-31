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
import asyncio
import re

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
_redirect_streaks: Dict[str, int] = {}   # session_id → consecutive REDIRECT count

# Detects when a transition agent leaked its own name as the response text instead of
# making a silent tool call.  Catches both "Phase Transition Agent" and "[Phase Transition Agent]".
_LEAKED_AGENT_RE = re.compile(
    r'^\[?(Phase Transition Agent|Topic Transition Agent)\]?\s*$',
    re.IGNORECASE,
)

# Phase-aware context for guardrail — injected at call time so CLARIFY/REDIRECT
# decisions are grounded in what the interview is actually asking about right now.
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
        - Exremely off-topic personal conversation unrelated to politics (weather, unrelated
          stories, casual chatter with no political relevance)
        - Requests to skip questions or change the subject without genuine political content
        Note: frustration WITH politics or politicians is NOT a redirect — it is on-topic.


        YOUR ENTIRE RESPONSE must be exactly one of these four forms with no other text:
          CLEAR
          CLARIFY
          REDIRECT
        """
    )


def create_agents(session_id: str) -> Dict[str, Agent]:

    # Mutable container so on_handoff callbacks can pass the last respondent
    # message to the interview agent, replacing the SDK's default
    # {"assistant": "..."} transfer message which causes empty output.
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
        instructions="""
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
        - Your input contains "FIRST ACTIVATION" or asks you to "present your summary" → Job 1.
          CRITICAL for Job 1: output your summary text and reaction question, then STOP.
          Do NOT call any handoff tool. Do NOT hand off to End Interview Agent. Just output text.
        - Your input contains "respondent has just reacted" or "reacted to your summary" → Job 2.
          Acknowledge their reaction briefly, then immediately hand off to End Interview Agent.
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
        You evaluate interview progress and determine when to move to the next phase. Analyze the full
        conversation history to assess whether the goals of the current phase have been met.

        PHASE 1 - INTRODUCTION:
        Goals: Open warmly and learn the respondent's overall feelings about politics.
        Transition when: The respondent has shared their general political feelings
        
        PHASE 2 - POLITICAL IDENTITY MEANING:
        Goals: Cover three questions — (1) what their political identity label means to them personally,
        (2) how their most-important pre-survey issue(s) connect to their ideology and self-description
        as liberal/moderate/conservative, and (3) whether they fit the group norms of their political
        identity group.
        Transition when: All three questions have received substantive answers.

        PHASE 3 - CONNECTIONS BETWEEN IDENTITY AND ISSUES:
        Goals: Start with the respondent's most-important pre-survey issue, citing their specific
        policy position on it. Then work through AT LEAST 3 more specific policy positions from
        the pre-survey, one per exchange. Every question must name a specific pre-survey policy
        the respondent expressed a view on — no generic questions. This is about alignment and
        meaning, not contradictions.
        Transition when: The most-important issue has been covered AND AT LEAST 3 additional
        specific pre-survey policy positions have been connected to their identity. Do not
        transition early.

        PHASE 4 - TENSIONS BETWEEN IDENTITY AND ISSUES:
        Goals: Identify and explore AT LEAST 3 contradictions or tensions between their stated
        identity and their specific issue stances. Pull from both the pre-survey positions AND
        things the respondent said during the interview. Each tension is its own exchange.
        Close by asking how they see themselves within the broader landscape of US politics.
        Transition when: AT LEAST 3 distinct tensions have been substantively explored AND the
        respondent has described where they sit in the broader US political landscape. Do not
        transition after only one or two tensions.

        DECISION RULES:
        First, identify the CURRENT phase by counting how many phases have been fully completed in
        the conversation history. You may only advance ONE phase at a time — never skip a phase.
        If the current phase goals are met, hand off to Political Interview Agent.
        If in Phase 4 and all goals are met, hand off to Summary Agent to conclude.
        If the respondent wants to end early, hand off to End Interview Agent.
        If current phase goals are not yet met, hand off to Political Interview Agent.

        YOUR ONLY JOB IS TO CALL A HANDOFF TOOL. Do not write any text whatsoever. Do not write
        "Transitioning to the next phase", "Please proceed", or anything else. Do not reason out
        loud. Do not summarize. Do not explain. Your entire response must be a tool call and
        nothing else. Any text you write will be shown directly to the respondent, which is a
        critical error that breaks the interview.

        IMPORTANT — PHASE 2 CANNOT BE SKIPPED AND REQUIRES THREE QUESTIONS: Phase 2 goals require
        all three dedicated exchanges — (1) identity meaning, (2) key issues + ideology connection,
        (3) group norm fit — not just a passing remark in Phase 1. If any of the three have not been
        asked and answered substantively, do not transition past Phase 2.
        """
    )

    topic_transition_agent = Agent(
        name="Topic Transition Agent",
        model="gpt-4o",
        handoffs=[
            handoff(phase_transition_agent),
        ],
        instructions="""
        You help the interviewer move between topics within the current interview phase. Based on the
        conversation history, determine whether the current topic has been sufficiently explored and
        whether there are other relevant topics to cover within this phase.

        When deciding which topic to move to next, scan the pre-survey background from the start of
        the conversation. Identify which issue positions have already been discussed and which have
        not yet been touched. Prefer moving to an unexplored pre-survey issue rather than returning
        to one already covered.

        When handing off back to the Political Interview Agent, provide a suggested question that flows
        naturally from what the respondent just said into the new topic. The question must feel like an
        organic follow-up, not a change of subject. Ground the suggested question in a specific
        pre-survey issue position that has not yet been explored. Never use phrases like "moving on",
        "let's shift", "turning to", or any language that signals a topic or phase change to the
        respondent.

        If the current topic is not yet exhausted, hand off to Political Interview Agent.
        If the overall phase goals appear complete, hand off to Phase Transition Agent.

        YOUR ONLY JOB IS TO CALL A HANDOFF TOOL. Do not write any text. Do not reason out loud.
        Do not summarize the topic. Do not suggest questions. Your entire response must be a tool
        call — nothing else. Any text you write will be shown directly to the respondent as the
        interviewer's reply, which would be a critical error.
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
        You are conducting a thoughtful qualitative interview about the respondent's political belief
        systems. Your goal is to understand how they see the relationships between their political
        identity (liberal/moderate/conservative) and their stances on specific policy issues. The
        pre-survey background at the start of this conversation contains their ideology score and issue
        positions — use this throughout to ask informed, specific questions. Never read it back verbatim;
        let it guide which connections and tensions you probe.

        THE INTERVIEW HAS FOUR PHASES:

        Phase 1 — Introduction: Introduce yourself as an AI Conversation Bot here to learn about
        the respondent's political views. Ask ONE simple question: how engaged are they with
        politics? Keep this phase brief — 1 to 2 exchanges — before moving on.

        Phase 2 — Political Identity Meaning: This phase requires exactly THREE exchanges, asked
        strictly in order. Do NOT skip Q1. Do NOT mention any policy issue until Q2.

          Q1 — Identity meaning (ALWAYS FIRST): Ask exactly one question: what does it mean
          to them to be [their label] — e.g. "What does being a moderate mean to you?" Do not
          add a second sentence or a follow-up clause. Do NOT reference any policy issue.

          Q2 — Issues and identity (ONLY after Q1 is answered): Pull the issue the respondent
          marked as most important from the pre-survey. Ask how that issue connects to their
          political ideology and how it factors into seeing themselves as liberal/moderate/conservative.

          Q3 — Group norms (ONLY after Q2 is answered): Ask whether they fit the typical profile
          of others who share their political identity — do they see themselves as a "typical"
          member of their group, or do they differ in meaningful ways?

          All three must receive substantive answers before this phase is complete.
          Do NOT ask a fourth question. Move to Phase 3 as soon as all three are done.

        Phase 3 — Connections Between Identity and Issues: Ask the respondent to explain how their
        specific policy positions CONNECT TO and FLOW FROM their political identity. This phase is
        EXCLUSIVELY about alignment and meaning — not contradictions, not tensions, not anything
        that requires two positions to be weighed against each other. Before asking the first
        question, look up the issue the respondent marked as most important in the pre-survey
        metadata, then find their specific policy stance on that issue. Your FIRST question MUST
        cite that exact policy by name — do not ask about the issue category in general. Then
        maintain a mental checklist and work through AT LEAST 3 more distinct pre-survey policy
        positions — one per exchange, each named specifically. NEVER bundle two positions together
        in one question (do NOT say "you support both X and Y"). Ask about ONE policy at a time.

        STRICT PHASE 3 LANGUAGE RULES — VIOLATION BREAKS THE STUDY:
        NEVER use any of these words or phrases in Phase 3: "tension", "reconcile", "contradict",
        "conflict", "inconsistency", "on the other hand", "both X and Y", "how do you balance",
        "how do you square". If you catch yourself writing any of these, you are asking a Phase 4
        question by mistake — stop, rephrase as a simple connection question, or call Phase
        Transition Agent to advance the phase first. Phase 3 questions have positive, connecting
        frames: "How does X connect to...?", "Where does your support for X come from?", "How does
        X fit with how you see yourself as [label]?". Do NOT move to Phase 4 until at least 3
        specific policy connections have been explained beyond the first.

        Phase 4 — Tensions Between Identity and Issues: This phase is ONLY about contradictions,
        inconsistencies, and tensions — not connections. Use the pre-survey AND things the
        respondent said earlier in the interview to identify where their views CONTRADICT or
        complicate their identity label. Each tension is its own exchange — surface it, hear their
        explanation, then move to the next. You must explore AT LEAST 3 distinct tensions. Lead
        with the sharpest contradiction first. If their positions are very consistent, probe where
        they diverge from typical members of their group or where they feel pulled in conflicting
        directions. Only after AT LEAST 3 tensions have been genuinely explored, close by asking
        how they see themselves within the broader landscape of US politics. Once the respondent
        answers that question, your ONLY action is to call Phase Transition Agent via tool call.
        Do NOT write any closing message, thank you, or summary — the Summary Agent and End
        Interview Agent handle that. Any text you write instead of the tool call is a critical error.

        PHASE 4 ENTRY — NATURAL BRIDGE: Your very first Phase 4 question must open with a warm,
        natural pivot that briefly references a theme from what they shared (e.g., "A lot of what
        you've described sounds like it flows from a core belief in X — I want to explore a couple
        spots where things feel a bit more complicated..."). Do NOT announce a new phase, do NOT
        say "now I want to look at tensions", and do NOT use the word "tension" in the opening
        sentence. The bridge should feel like a natural deepening of the conversation, not a hard
        reset. After the bridge sentence, ask your first tension question as normal.

        AGENTS AND HANDOFFS:
        Call Topic Transition Agent when the current political topic has been sufficiently explored and
        you want to move to a new topic within the same phase — but only after at least 2 substantive
        exchanges on that topic. Call Phase Transition Agent when you believe the goals of the current
        phase are fully met — but only after the phase has been genuinely explored, not after a single
        on-topic answer.
        If at ANY point the respondent indicates they want to stop or end the interview (e.g. "I'm
        done", "stop", "end", "I don't want to continue", "that's enough"), immediately call Phase
        Transition Agent so the interview can be concluded gracefully via the Summary and End agents.

        When you receive a handoff back from either transition agent, they will suggest a bridging
        question — use it as your next question exactly as suggested or adapt it slightly. Never
        announce to the respondent that you are transitioning, moving on, or shifting topics. The
        conversation must always feel like a natural, flowing dialogue.

        CRITICAL — OUTPUT ONLY YOUR REPLY TO THE RESPONDENT: Never output internal reasoning,
        phase summaries, agent commentary, or anything prefixed with labels like "Phase X goals",
        "Suggested Bridging Question:", "I will now advance", or similar. If any such text appears
        in a handoff message you receive, ignore it entirely. Your response is always and only
        what you say directly to the respondent.

        CRITICAL: Never say anything like "moving on to the next topic", "let's shift to", "turning to
        another area", "for the next part of our conversation", or any phrase that signals a structured
        transition. Simply ask the next question as if it arose naturally from the conversation.

        NATURALNESS — apply throughout:
        - After the respondent gives a substantive answer, briefly echo ONE specific thing they said
          before asking your next question. One sentence maximum — make it feel heard, not repeated.
          Example: "That idea of personal responsibility really comes through in how you're describing
          it — [next question]."
        - Vary your question openers across the conversation. Do not start consecutive questions with
          the same phrase. Mix in: "What draws you to...", "How do you think about...", "Where does
          that come from for you?", "Does that feel consistent with...?", "I'm curious whether...",
          "How does that sit with you when...".
        - If the respondent shares something emotionally resonant or personal, acknowledge it in one
          warm sentence before continuing. Keep the interview human, not clinical.

        METADATA USAGE — CRITICAL FOR PHASES 3 AND 4:
        Before every question in phases 3 and 4, identify a SPECIFIC issue position from the pre-survey
        relevant to what the respondent just said. Build your question around that position — make it
        feel personal to this respondent, not generic. Rotate through their issues; do not return to
        one already explored. In phase 4, actively look for the sharpest tension between their ideology
        score and a specific issue position and lead with that.

        CRITICAL — ONE QUESTION ONLY: Every single response must end with exactly ONE question.
        Never write two sentences that are both questions. Never add a clarifying question after
        your main question. Never follow a question with "or" and another question. If you catch
        yourself writing a second question mark, delete everything after the first one.

        INTERVIEW GUIDELINES:
        Ask only one question at a time. Ask open-ended questions, not yes/no questions. Remain
        non-judgmental and focus on understanding, not debating. Always lead with a question — do not
        wait for the respondent to start. Use the pre-survey metadata to ask informed, specific
        questions that draw on their stated issue positions. Never mention phases, transitions, or
        agent names to the respondent. Conclude after 15 to 20 questions or if the respondent wishes
        to end early.
        """
    )

    def advance_phase_and_relay(ctx: RunContextWrapper) -> None:
        advance_phase(ctx)

    # Custom Handoff subclass that injects the actual respondent message instead
    # of the SDK default {"assistant": "Agent Name"} which causes empty output.
    def _make_relay_handoff(target_agent, on_handoff_fn=None):
        # Build a normal handoff then patch get_transfer_message using object.__setattr__
        # so it works even on frozen dataclasses (as seen in some deployed SDK versions).
        h = handoff(target_agent, on_handoff=on_handoff_fn) if on_handoff_fn else handoff(target_agent)
        object.__setattr__(
            h,
            'get_transfer_message',
            lambda agent: f"Respondent's latest response: {last_respondent_msg['text']}"
        )
        return h

    phase_transition_agent.handoffs.append(
        _make_relay_handoff(interview_agent, advance_phase_and_relay)
    )
    topic_transition_agent.handoffs.append(_make_relay_handoff(interview_agent))

    # guardrail_agent removed — built fresh per request via _build_guardrail_agent()
    return {
        "interview_agent":        interview_agent,
        "phase_transition_agent": phase_transition_agent,
        "topic_transition_agent": topic_transition_agent,
        "summary_agent":          summary_agent,
        "end_interview_agent":    end_interview_agent,
        "_last_respondent_msg":   last_respondent_msg,  # mutable dict, updated each turn
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
    version="4.13.0",
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
    relay_msg: dict = agents["_last_respondent_msg"]  # type: ignore[assignment]
    sdk_session   = SQLiteSession(session_id, DB_PATH)
    current_phase = meta["interview_phase"]
    is_kickoff    = request.message.lower().strip() in {"hello", "hi", "start", "begin"}

    agent_input    = None
    starting_agent = None

    if not is_kickoff:
        # Built fresh each request: phase-aware, gpt-4o-mini, stateless
        guardrail_agent  = _build_guardrail_agent(current_phase)
        guardrail_result = await Runner.run(
            guardrail_agent,
            f"Latest respondent input: {request.message}",
            session=SQLiteSession(f"{session_id}_guardrail", DB_PATH)
        )
        guardrail_output = guardrail_result.final_output.strip()
        guardrail_lower  = guardrail_output.lower()
        logger.info(f"[{session_id}] Guardrail [phase {current_phase}] → {guardrail_output!r}")

        if guardrail_lower.startswith("clarify"):
            _redirect_streaks[session_id] = 0
            relay_msg["text"] = request.message
            agent_input = f"""The respondent seems confused or has asked for clarification about your last
                question. Do NOT move to a new topic. Instead:
                1. Briefly clarify what you were asking in simple, accessible language (1 sentence).
                2. Re-ask the same question in a slightly different, clearer way.
                Keep the tone warm and reassuring — make them feel comfortable, not tested.
                Current phase context: {PHASE_GUARDRAIL_CONTEXT.get(current_phase, '')}

                Their response was: {request.message}"""
            starting_agent = agents["interview_agent"]

        elif guardrail_lower.startswith("redirect"):
            streak = _redirect_streaks.get(session_id, 0) + 1
            _redirect_streaks[session_id] = streak
            if streak >= 3:
                # 3 consecutive redirects — end the interview gracefully
                _redirect_streaks.pop(session_id, None)
                logger.info(f"[{session_id}] 3 consecutive redirects — ending interview")
                end_result = await Runner.run(
                    agents["end_interview_agent"],
                    "The respondent has repeatedly gone off-topic. Close the interview warmly.",
                    session=sdk_session,
                )
                end_content = end_result.final_output or ""
                end_signal_val = None
                if "CONCLUDE_INTERVIEW" in end_content:
                    end_signal_val = "conclude"
                    end_content = end_content.replace("CONCLUDE_INTERVIEW", "").strip()
                _update_phase(session_id, 6)
                if session_id in _agent_cache:
                    del _agent_cache[session_id]
                return InterviewResponse(
                    reply=end_content,
                    session_id=session_id,
                    interview_phase=6,
                    end_signal=end_signal_val or "conclude",
                )
            relay_msg["text"] = request.message
            agent_input = f"""The respondent just sent a message that is off-topic or asks about the survey
                structure rather than engaging with the interview questions. Do NOT answer their
                off-topic question or acknowledge the survey structure. Instead, write a single,
                warm sentence that gently acknowledges their comment without engaging with it,
                immediately followed by the next natural interview question that brings them back
                to the political topic at hand. The transition must feel natural — do not say
                anything like "let's get back to" or "returning to our interview". Just pivot
                smoothly with curiosity.
                Current phase context: {PHASE_GUARDRAIL_CONTEXT.get(current_phase, '')}

                Their off-topic message was: {request.message}"""
            starting_agent = agents["interview_agent"]

        elif guardrail_lower.startswith("flag"):
            _redirect_streaks[session_id] = 0
            relay_msg["text"] = request.message
            agent_input = f"""The respondent's message contained content that needs a gentle redirect.
                Respond in a single warm sentence that acknowledges you'd like to keep the conversation
                respectful and constructive, then ask the next natural interview question.
                Current phase context: {PHASE_GUARDRAIL_CONTEXT.get(current_phase, '')}
                Their message was: {request.message}"""
            starting_agent = agents["interview_agent"]

        else:  # CLEAR — fall through to normal processing below
            _redirect_streaks[session_id] = 0

    if agent_input is not None:
        pass  # guardrail already set agent_input and starting_agent
    elif is_kickoff:
        metadata_str = "\n".join(
            f"  - {k}: {v}" for k, v in meta["user_metadata"].items()
        ) or "  (No pre-survey data available)"

        agent_input = f"""Begin the interview. Introduce yourself in one sentence as an AI
        Conversation Bot here to learn about their political views. Then ask exactly this, word
        for word: "How engaged are you with politics?" Nothing more.

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
        # Update the relay message so transition agent handoffs carry the real respondent text
        relay_msg["text"] = request.message
        logger.info(f"[{session_id}] Normal turn — phase={current_phase} msg_preview={request.message[:80]!r}")

        agent_input = f"""Respondent's latest response: {request.message}

        Respond thoughtfully and ask a follow-up question. The pre-survey background shared at the
        start of this conversation is in your history — use it to inform the direction and depth of
        your questions. If you have sufficiently explored the current topic, hand off to Topic
        Transition Agent. If you believe the current phase goals are fully met, hand off to Phase
        Transition Agent. Ask ONE open-ended question at a time and remain non-judgmental."""

        starting_agent = agents["interview_agent"]

    assert starting_agent is not None and agent_input is not None
    logger.info(f"[{session_id}] Starting Runner.run with agent={starting_agent.name} phase={current_phase}")
    result = await Runner.run(
        starting_agent,
        agent_input,
        session=sdk_session,
    )
    logger.info(f"[{session_id}] Runner.run complete — last_agent={getattr(result.last_agent, 'name', '?')} output_len={len(str(result.final_output or ''))}")
    response_content = result.final_output

    updated_meta  = _load_meta(session_id)
    current_phase = updated_meta["interview_phase"] if updated_meta else meta["interview_phase"]
    logger.info(f"[{session_id}] Phase after run={current_phase} final_output_preview={str(response_content or '')[:120]!r}")

    # If a transition agent ended up as the last agent, its reasoning text leaked
    # into final_output. Route correctly based on phase.
    # Also catches the case where Phase Transition Agent made a handoff tool call to
    # Summary Agent (enter_summary_reaction fired) but Summary Agent produced empty
    # output — the SDK preserves the Phase Transition Agent's leaked text as final_output
    # and last_agent becomes Summary Agent, bypassing the name check.
    last_agent_name = getattr(result.last_agent, "name", "")
    _resp_stripped   = str(response_content or "").strip()
    _is_transition_leak = (
        last_agent_name in ("Phase Transition Agent", "Topic Transition Agent")
        or bool(_LEAKED_AGENT_RE.match(_resp_stripped))
    )
    if _is_transition_leak:
        logger.warning(f"[{session_id}] Transition agent '{last_agent_name}' leaked as last_agent — phase={current_phase}")
        if current_phase >= 4:
            # Phase 4 complete — route to Summary Agent to present summary and close
            logger.info(f"[{session_id}] Phase 4 done — running Summary Agent")
            _update_phase(session_id, 5)
            current_phase = 5
            summary_result = await Runner.run(
                agents["summary_agent"],
                (
                    "FIRST ACTIVATION — JOB 1: The interview's substantive phases are now complete. "
                    "Present a comprehensive summary of the respondent's political views, then ask "
                    "ONE reaction question. You MUST output your summary as text and STOP — do NOT "
                    "hand off to the End Interview Agent yet. Wait for the respondent's reply first."
                ),
                session=sdk_session,
            )
            logger.info(f"[{session_id}] Summary Agent done — last_agent={getattr(summary_result.last_agent, 'name', '?')} output_len={len(str(summary_result.final_output or ''))}")
            response_content = summary_result.final_output
            # Safety: if summary_agent still produced nothing, return a graceful fallback
            if not str(response_content or "").strip():
                logger.error(f"[{session_id}] Summary Agent returned empty output — using fallback")
                response_content = (
                    "I really appreciate you sharing your views with me today. "
                    "How does this conversation feel to you — does it capture where you stand politically, "
                    "or is there anything you'd push back on or add?"
                )
        else:
            logger.info(f"[{session_id}] Re-running interview agent (phase {current_phase})")
            rerun_result = await Runner.run(
                agents["interview_agent"],
                f"Respondent's latest response: {request.message}",
                session=sdk_session,
            )
            logger.info(f"[{session_id}] Rerun complete — last_agent={getattr(rerun_result.last_agent, 'name', '?')}")
            response_content = rerun_result.final_output

    end_signal = None
    response_content = str(response_content or "").strip()
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
        "version":         "4.13.0",
    }


if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    uvicorn.run(app, host="localhost", port=8000)



    
