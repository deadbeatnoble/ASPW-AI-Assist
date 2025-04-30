import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import requests

load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")

app = FastAPI()

FAL_FLUX_URL = "https://queue.fal.run/fal-ai/flux/dev"

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config={
        "max_output_tokens": 500
    }
)

chat = model.start_chat(history=[])

class ImagePrompt(BaseModel):
    scene: str
    extra: str | None = None

class ChatRequest(BaseModel):
    message: str
    context: str | None = None

@app.post("/ai-assistance")
async def chat_endpoint(req: ChatRequest):
    try:
        system_instruction = (
            "Respond only in Amharic. No matter what language the user uses, always reply in fluent Amharic.\n"
        )

        if(req.context):
            system_instruction += "Here is some background information to consider when replying:\n"f"{req.context}\n\n"

        system_instruction += "Now respond to the user's message below."

        full_message = f"{system_instruction}\n\nUser: {req.message}"

        result = chat.send_message(full_message)
        
        response = result.text
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/generate-image")
def generate_image(prompt: ImagePrompt):
    full_prompt = prompt.scene
    if prompt.extra:
        full_prompt += f"\nAdditional context: {prompt.extra}"

    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        FAL_FLUX_URL,
        headers=headers,
        json={"prompt": full_prompt}
    )

    if response.status_code != 200:
        return {"error": "Failed to queue image generation", "details": response.text}

    res_json = response.json()
    return {
        "status": res_json["status"],
        "request_id": res_json["request_id"],
        "status_check_url": res_json["status_url"],
        "image_url": res_json["response_url"],
        "cancel_url": res_json["cancel_url"]
    }