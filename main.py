import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

app = FastAPI()

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config={
        "max_output_tokens": 500
    }
)

chat = model.start_chat(history=[])

class ChatRequest(BaseModel):
    message: str
    context: str

@app.post("/ai-assistance")
async def chat_endpoint(req: ChatRequest):
    try:
        system_instruction = (
            "Respond only in Amharic. No matter what language the user uses, always reply in fluent Amharic.\n"
            "Here is some background information to consider when replying:\n"
            f"{req.context}\n\n"
            "Now respond to the user's message below."
        )
        full_message = f"{system_instruction}\n\nUser: {req.message}"

        result = chat.send_message(full_message)
        
        response = result.text
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
