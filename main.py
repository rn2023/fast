from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import os
from typing import List, Optional
from agents import Agent, Runner, handoff

app = FastAPI(title="Political Views Interview Agent API", version="1.0.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://duke.yul1.qualtrics.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InterviewRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []

class InterviewResponse(BaseModel):
    reply: str
    end_signal: Optional[str] = None

transition_agent = Agent(
    name="Transition Agent",
    instructions=(
        "Analyze the political conversation and decide when to transition topics. "
        "If the conversation is ready for summarization of their political views, hand off to summary_agent. "
        "If it's time for a final reflection question about their political beliefs, hand off to final_question_agent. "
        "If the political interview should end, hand off to end_interview_agent. "
        "Otherwise, provide guidance on how to continue exploring their political views and correlations."
    ),
    model="gpt-4o",
    handoffs=[]
)

summary_agent = Agent(
    name="Summary Agent", 
    instructions="Summarize the user's political views, key positions, and the correlations between different political issues they discussed. Provide a clear and comprehensive summary of their political beliefs and how they connect their various positions.",
    model="gpt-4o"
)

final_question_agent = Agent(
    name="Final Question Agent",
    instructions="Ask one final thoughtful question about their political views to help them reflect on their overall political philosophy or how their beliefs have evolved. Make it open-ended and deeply informative about their political journey.",
    model="gpt-4o"
)

end_interview_agent = Agent(
    name="End Interview Agent", 
    instructions=(
        "Provide a thoughtful conclusion to the political interview. Thank them for sharing their political views and offer any final reflections on the complexity of political beliefs. "
        "Always end your response with 'CONCLUDE_INTERVIEW' to signal the interview is complete."
    ),
    model="gpt-4o"
)

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="Monitor political conversation for safety and appropriateness. Ensure the discussion remains respectful and focused on understanding views rather than debate. If the conversation becomes inappropriate, hostile, or unsafe, respond with 'FLAG: [reason]'. Otherwise, respond with 'CLEAR'.",
    model="gpt-4o"
)

context_agent = Agent(
    name="Context Agent",
    instructions="Given a political conversation history and a new user message, rewrite the message with useful context from the previous political discussion. Output only the rewritten message, making it clearer and more specific based on the political topics that have been discussed.",
    model="gpt-4o"
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
        "After 3-5 substantial exchanges about their political views, consider handing off to the transition agents. "
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
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)

@app.post("/chat", response_model=InterviewResponse)
async def chat_endpoint(request: InterviewRequest):
    try:
        if not request.conversation_history and request.message.lower() in ["hello", "hi", "start", "begin"]:
            opening_prompt = "Start the political interview by asking what political issues they are most passionate about and why those issues matter to them personally. Be warm and curious in your approach."
            result = await Runner.run(interview_agent, opening_prompt)
            
            response_content = result.final_output
            
            return InterviewResponse(
                reply=response_content,
                end_signal=None
            )
        
        convo_history = ""
        if request.conversation_history:
            convo_history = format_conversation_history(request.conversation_history)
        
        guardrail_input = f"Political conversation:\n{convo_history}\n\nLatest input: {request.message}"
        guardrail_result = await Runner.run(guardrail_agent, guardrail_input)
        
        if "flag" in guardrail_result.final_output.lower():
            return InterviewResponse(
                reply=f"I appreciate your input, but let's keep our political discussion respectful and focused on understanding your views. {guardrail_result.final_output}",
                end_signal=None
            )
        
        final_input = request.message
        if request.conversation_history:
            context_prompt = (
                f"Political conversation history:\n{convo_history}\n\n"
                f"User's new message: {request.message}\n\n"
                f"Rewrite this message with helpful context from the political discussion:"
            )
            context_result = await Runner.run(context_agent, context_prompt)
            final_input = context_result.final_output
        
        agent_input = (
            f"Political conversation so far:\n{convo_history}\n\n"
            f"User's latest response: {final_input}\n\n"
            f"As the political interviewer, respond to their answer and ask a thoughtful follow-up question that explores their political views deeper. "
            f"Focus on understanding the correlations between their different political positions and the underlying values that drive their beliefs. "
            f"Remember you are leading this political interview and always end with a question unless handing off to another agent."
        )
        
        result = await Runner.run(interview_agent, agent_input)
        response_content = result.final_output
        
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
    return {"status": "healthy", "message": "Political Views Interview Agent API is running"}

if __name__ == "__main__":
    import uvicorn
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
