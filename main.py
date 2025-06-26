from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import os
from typing import List, Optional
from agents import Agent, Runner, handoff

app = FastAPI(title="Interview Agent API", version="1.0.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://duke.yul1.qualtrics.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class InterviewRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []

class InterviewResponse(BaseModel):
    reply: str
    end_signal: Optional[str] = None

# --- Agents ---
transition_agent = Agent(
    name="Transition Agent",
    instructions=(
        "Analyze the conversation and decide when to transition topics. "
        "If the conversation is ready for summarization, hand off to summary_agent. "
        "If it's time for a final question, hand off to final_question_agent. "
        "If the interview should end, hand off to end_interview_agent. "
        "Otherwise, provide guidance on how to continue the interview."
    ),
    model="gpt-4o",
    handoffs=[]
)

summary_agent = Agent(
    name="Summary Agent", 
    instructions="Summarize the user's view and key takeaways from the interview. Provide a clear, concise summary of their position and main points discussed.",
    model="gpt-4o"
)

final_question_agent = Agent(
    name="Final Question Agent",
    instructions="Ask one final thoughtful question to help the user reflect on the topic discussed. Make it open-ended and insightful.",
    model="gpt-4o"
)

end_interview_agent = Agent(
    name="End Interview Agent", 
    instructions=(
        "Provide a thoughtful conclusion to the interview. Thank the user and offer any final thoughts or resources if appropriate. "
        "Always end your response with 'CONCLUDE_INTERVIEW' to signal the interview is complete."
    ),
    model="gpt-4o"
)

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="Monitor conversation for safety and relevance. If the conversation is off-track, inappropriate, or unsafe, respond with 'FLAG: [reason]'. Otherwise, respond with 'CLEAR'.",
    model="gpt-4o"
)

context_agent = Agent(
    name="Context Agent",
    instructions="Given a conversation history and a new user message, rewrite the message with useful context from the conversation. Output only the rewritten message, making it clearer and more specific based on what has been discussed.",
    model="gpt-4o"
)

interview_agent = Agent(
    name="Interview Agent",
    instructions=(
        "You are a skilled interviewer conducting a thoughtful interview. Your role is to guide the conversation and ask insightful questions. "
        "ALWAYS lead with questions - never wait for the user to start. "
        "Start interviews with an opening question about their views, experiences, or opinions on a topic. "
        "Ask open-ended, clarifying questions to deeply understand the user's position. "
        "Listen carefully to their responses and ask thoughtful follow-up questions. "
        "Guide the conversation by asking probing questions that help explore different angles. "
        "Keep the conversation focused and engaging by steering it with your questions. "
        "After 4-6 exchanges, you can hand off to other agents when appropriate. "
        "Always end your responses with a question unless concluding the interview."
    ),
    model="gpt-4o",
    handoffs=[
        handoff(transition_agent),
        handoff(summary_agent), 
        handoff(final_question_agent),
        handoff(end_interview_agent)
    ]
)

# Set handoffs for transition agent after all agents are defined
transition_agent = Agent(
    name="Transition Agent",
    instructions=transition_agent.instructions,
    model="gpt-4o",
    handoffs=[
        handoff(summary_agent),
        handoff(final_question_agent), 
        handoff(end_interview_agent)
    ]
)

def format_conversation_history(history: List[dict]) -> str:
    """Format conversation history as a string"""
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)

@app.post("/chat", response_model=InterviewResponse)
async def chat_endpoint(request: InterviewRequest):
    try:
        # If this is the start of the interview (no history and greeting message)
        if not request.conversation_history and request.message.lower() in ["hello", "hi", "start", "begin"]:
            opening_prompt = "Start the interview by asking an engaging opening question. Choose a topic that would be interesting to explore and ask the user about their thoughts, experiences, or perspective on it."
            result = await Runner.run(interview_agent, opening_prompt)
            
            response_content = result.final_output
            
            return InterviewResponse(
                reply=response_content,
                end_signal=None
            )
        
        # Format conversation history if it exists
        convo_history = ""
        if request.conversation_history:
            convo_history = format_conversation_history(request.conversation_history)
        
        # Guardrail check
        guardrail_input = f"Conversation:\n{convo_history}\n\nLatest input: {request.message}"
        guardrail_result = await Runner.run(guardrail_agent, guardrail_input)
        
        if "flag" in guardrail_result.final_output.lower():
            return InterviewResponse(
                reply=f"I appreciate your input, but let's keep our conversation focused on the interview topic. {guardrail_result.final_output}",
                end_signal=None
            )
        
        # Context agent rewrites input if there's conversation history
        final_input = request.message
        if request.conversation_history:
            context_prompt = (
                f"Conversation history:\n{convo_history}\n\n"
                f"User's new message: {request.message}\n\n"
                f"Rewrite this message with helpful context:"
            )
            context_result = await Runner.run(context_agent, context_prompt)
            final_input = context_result.final_output
        
        # Get response from interview agent
        agent_input = (
            f"Conversation so far:\n{convo_history}\n\n"
            f"User's latest response: {final_input}\n\n"
            f"As the interviewer, respond to their answer and ask a thoughtful follow-up question to guide the conversation deeper. "
            f"Remember you are leading this interview - always end with a question unless handing off to another agent."
        )
        
        result = await Runner.run(interview_agent, agent_input)
        response_content = result.final_output
        
        # Check if interview should end
        end_signal = None
        if "CONCLUDE_INTERVIEW" in response_content:
            end_signal = "conclude"
        
        return InterviewResponse(
            reply=response_content,
            end_signal=end_signal
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Interview Agent API is running"}

if __name__ == "__main__":
    import uvicorn
    # Make sure OPENAI_API_KEY is set in environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
