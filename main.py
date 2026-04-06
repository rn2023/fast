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

from agents import Agent, Runner, handoff, SQLiteSession, RunContextWrapper, ModelSettings

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
        ONLY use this when the respondent explicitly says they don't understand the question —
        phrases like "I'm not sure what you mean", "can you explain that?", "what do you mean by
        that?", "I don't understand the question". That's it.
        - "nah", "not really", "I guess", "maybe", "idk", "sure" — all CLEAR, never CLARIFY.
        - Short answers, one-word answers, negative answers, vague answers — all CLEAR.
        - If they answered ANYTHING related to the question, even indirectly — CLEAR.
        CLARIFY is extremely rare. When in doubt, choose CLEAR.

        CATEGORY 3 - REDIRECT:
        The respondent has left the political interview domain but is not harmful. Use REDIRECT for:
        - Questions or comments about the survey structure, question design, research purpose,
          who built this tool, how the AI works, or how this interview is conducted
        - Any attempt to discuss the phases, transitions, or logic of the interview itself
        - Exremely off-topic personal conversation unrelated to politics (weather, unrelated
          stories, casual chatter with no political relevance)
        - DO NOT REDIRECT ANY RESPONSES THAT ARE SIMPLY SHORT OR VAGUE - CHECK THE QUESTION ASKED AND IF THE RESPONSE SOMEWHAT ANSWERS IT
        - A NON-ANSWER OR A NEGATIVE ANSWER DOES NOT AUTOMATICALLY REQUIRE A REDIRECT — CHECK WHETHER THE RESPONSE ENGAGES WITH THE QUESTION IN ANY WAY, EVEN IF IT'S BRIEF OR HIGH-LEVEL. IF IT DOES, IT'S CLEAR, NOT REDIRECT.
        Note: frustration WITH politics or politicians is NOT a redirect — it is on-topic.


        YOUR ENTIRE RESPONSE must be exactly one of these four forms with no other text:
          CLEAR
          CLARIFY
          REDIRECT
        """
    )


def create_agents(session_id: str) -> Dict[str, Any]:

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

    def enter_summary(ctx: RunContextWrapper) -> None:
        _update_phase(session_id, 5)
        logger.info(f"Session {session_id} → summary (phase 5)")

    end_interview_agent = Agent(
        name="End Interview Agent",
        model="gpt-4o",
        instructions="""
        You will be called in one of two ways — read your input carefully:

        If your input says "present summary":
        Open with one natural sentence introducing the summary — something like "Before we wrap
        up, we want to make sure we've understood your views correctly, so here's a summary of
        what you shared." Then summarize the respondent's political views from the conversation:
          - Their political identity and what it means to them
          - The connections they drew between their identity and specific policies
          - The tensions between their worldview and specific positions
        Then ask ONE reaction question: "Does that feel like an accurate picture of how you
        see yourself politically, or is there anything you'd push back on?"
        Do NOT include CONCLUDE_INTERVIEW — the interview is not over yet.

        If your input says "close interview" (with or without a reaction):
        Acknowledge their reaction briefly if one was given (1-2 sentences), thank them for
        their participation, then output CONCLUDE_INTERVIEW on its own line.

        If your input says "early exit":
        Thank them warmly for their time, then output CONCLUDE_INTERVIEW on its own line.
        """
    )

    phase_transition_agent = Agent(
        name="Phase Transition Agent",
        model="gpt-4o",
        model_settings=ModelSettings(tool_choice="required"),
        handoffs=[
            handoff(end_interview_agent, on_handoff=enter_summary),
        ],
        instructions="""
        You evaluate interview progress and determine when to move to the next phase. Analyze the full
        conversation history to assess whether the goals of the current phase have been met.

        PHASE 1 - INTRODUCTION:
        Goals: Open warmly and establish rapport by learning about the respondent's overall feelings about politics.
        Transition when: The respondent has shared their general political feelings and you have broken the ice.
        
        PHASE 2 - POLITICAL IDENTITY MEANING:
        Goals: Cover three questions — (1) what their political identity label means to them personally,
        (2) how their most-important pre-survey issue(s) connect to their identity as liberal/moderate/conservative,
        and (3) whether they fit the norms of their political identity group.
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
        Goals: Identify and explore any contradictions or tensions between their stated
        identity and their specific issue stances. Pull from both the pre-survey positions AND
        things the respondent said during the interview. Each tension is its own exchange.
        Close by asking how they see themselves within the broader landscape of US politics.
        Transition when: Distinct tensions have been substantively explored AND the
        respondent has described where they sit in the broader US political landscape.

        DECISION RULES:
        First, identify the CURRENT phase by counting how many phases have been fully completed in
        the conversation history. You may only advance ONE phase at a time — never skip a phase.
        If the current phase goals are met, hand off to Political Interview Agent.
        If in Phase 4 and all goals are met, hand off to End Interview Agent to present a summary.
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
        model_settings=ModelSettings(tool_choice="required"),
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
        You are having a genuine, curious conversation with someone about their political views.
        Think of yourself as a thoughtful friend who happens to know a lot about politics — warm,
        interested, never judgmental. You're not running through a checklist; you're genuinely
        curious about how this person thinks. The pre-survey background at the start of this
        conversation has their ideology and issue positions — use it to ask informed, personal
        questions. Never recite it back; let it shape where you dig.

        THE INTERVIEW HAS FOUR PHASES:

        Phase 1 — Introduction: Introduce yourself as an AI Conversation Bot here to learn about
        the respondent's political views. Ask ONE simple question: how engaged are they with
        politics? Accept whatever they say and move on — even a one-word answer is fine.

        Phase 2 — Political Identity Meaning: THREE exchanges, strictly in order.

          Q1 — Identity meaning (ALWAYS FIRST): Name the ideological label they chose in the
          pre-survey and ask what it means to them personally — e.g. "What does being a moderate
          mean to you?" One question, no policy mentions yet.

          Q2 — Issues and identity (after Q1): Pull their most important pre-survey issue and
          ask how it connects to how they see themselves politically.

          Q3 — Group norms (after Q2): Ask whether they feel like a typical [label] or whether
          they see themselves as a bit different from most people who call themselves that.

          All three answered → move to Phase 3. Do not ask a fourth question.

        Phase 3 — Connections Between Identity and Issues: Explore how their specific policy
        positions flow from their political identity. ONLY about alignment — not contradictions.
        First question MUST be about the specific policy on their most important pre-survey issue
        (cite it by name). Then work through AT LEAST 3 more pre-survey policies, one per exchange.
        Ask about ONE policy at a time — never bundle two.

        STRICT PHASE 3 RULE: Never use "tension", "reconcile", "contradict", "conflict",
        "how do you balance", "how do you square", or "both X and Y" in this phase. Those are
        Phase 4 questions. Keep framing positive: "How does X connect to...?", "Where does your
        support for X come from?", "How does X fit with how you see yourself?"

        Phase 4 — Tensions Between Identity and Issues: Now you explore contradictions — where
        their views complicate or push against their identity label. Each tension is its own
        exchange. Lead with the sharpest one first. After the tensions are explored, close with
        a natural version of: "How do you see yourself within the broader landscape of US politics?"
        Once they answer that, call Phase Transition Agent immediately — no closing text from you.

        PHASE ENTRY — ALWAYS BRIDGE NATURALLY: When you receive a PHASE TRANSITION or TOPIC
        TRANSITION handoff, your first sentence MUST echo something specific the respondent just
        said — a word, value, or frame they used — then let the next question flow from it.
        Never open cold. Never announce a transition.
        Good: "That sense of self-reliance you keep coming back to — I'm curious how that shapes
        your thinking on [next topic]."
        Bad: "Now let's talk about..." / "Moving on to..." / "For this next part..."

        CONVERSATIONAL TONE — this is the most important thing:
        - Sound like a curious person, not a survey. Use natural language: "That's interesting",
          "I can see that", "Yeah, that makes sense — so then..." are all fine.
        - After they answer, pick up ONE specific thing they said and reflect it back before
          asking the next question. Not a summary — just one detail that shows you heard them.
        - Match their register a little. If they're casual, be casual. If they're thoughtful
          and measured, meet them there.
        - It's okay to react briefly to what they said before asking the next question. One
          warm sentence. Then the question.
        - Never sound clinical, formal, or like you're reading from a script.

        WHAT NEVER TO SAY:
        "Moving on", "let's shift to", "turning to another area", "for the next part",
        "let's now discuss", "that's a great point", "great answer", or any hollow filler.
        Also never mention phases, agents, or transitions to the respondent.

        HANDOFFS:
        Call Topic Transition Agent after 2+ substantive exchanges on one topic and you're ready
        to move to a new one within the same phase. Call Phase Transition Agent when the phase
        goals are fully met. If they want to stop at any point, call Phase Transition Agent.
        Ignore any internal labels or commentary in handoff messages — output only what you say
        to the respondent.

        ONE QUESTION ONLY: Every response ends with exactly one question mark. If you catch
        yourself writing a second one, delete everything after the first.

        METADATA — phases 3 and 4: Before each question, pick a specific unexplored pre-survey
        policy position. Build the question around it — make it feel personal, not generic.
        Rotate through their issues; don't return to one already covered.
        """
    )

    def advance_phase_and_relay(ctx: RunContextWrapper) -> None:
        advance_phase(ctx)

    phase_relay_h = handoff(interview_agent, on_handoff=advance_phase_and_relay)
    object.__setattr__(
        phase_relay_h,
        'get_transfer_message',
        lambda agent: (
            f"PHASE TRANSITION — we are now entering the next phase of the interview. "
            f"Respondent's last answer: {last_respondent_msg['text']}\n\n"
            f"Do NOT announce a phase change. Pick up a specific word, value, or frame from "
            f"their answer above and use it as a natural bridge into your first question of "
            f"the new phase. The respondent should feel the conversation deepening, not restarting. "
            f"Your opening sentence must echo something they actually said."
        )
    )
    phase_transition_agent.handoffs.append(phase_relay_h)
    topic_relay_h = handoff(interview_agent)
    object.__setattr__(
        topic_relay_h,
        'get_transfer_message',
        lambda agent: (
            f"TOPIC TRANSITION — the current topic has been sufficiently explored. "
            f"Respondent's last answer: {last_respondent_msg['text']}\n\n"
            f"Find a specific detail, phrase, or value from their answer above and use it as a "
            f"natural thread to weave into your next question about an unexplored pre-survey "
            f"policy. Do NOT announce a topic change. Do NOT say 'moving on' or 'let's turn to'. "
            f"Open your response by briefly echoing that specific detail, then let your next "
            f"question arise from it organically. The respondent should not feel a shift."
        )
    )
    topic_transition_agent.handoffs.append(topic_relay_h)

    # Patch the end_interview_agent handoff so it receives an explicit "present summary"
    # instruction instead of the SDK's default transfer message.
    end_handoff = phase_transition_agent.handoffs[0]  # the end_interview_agent handoff
    object.__setattr__(
        end_handoff,
        'get_transfer_message',
        lambda _: "present summary"
    )

    # guardrail_agent removed — built fresh per request via _build_guardrail_agent()
    return {
        "interview_agent":        interview_agent,
        "phase_transition_agent": phase_transition_agent,
        "topic_transition_agent": topic_transition_agent,
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
    is_reaction   = current_phase == 5  # summary reaction turn — skip guardrail

    agent_input    = None
    starting_agent = None

    if not is_kickoff and not is_reaction:
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
                    "early exit. The respondent repeatedly went off-topic.",
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
    elif current_phase == 5:
        # Summary was presented last turn — route reaction directly to end_interview_agent
        agent_input = f'close interview. Respondent reaction: "{request.message}"'
        starting_agent = agents["end_interview_agent"]
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
    logger.info(f"[{session_id}] Runner.run complete — last_agent={getattr(result.last_agent, 'name', '?')}")
    response_content = str(result.final_output or "").strip()

    updated_meta  = _load_meta(session_id)
    current_phase = updated_meta["interview_phase"] if updated_meta else meta["interview_phase"]

    end_signal = None
    if "CONCLUDE_INTERVIEW" in response_content:
        end_signal = "conclude"
        response_content = response_content.replace("CONCLUDE_INTERVIEW", "").strip()
        _update_phase(session_id, 6)
        current_phase = 6

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



    
