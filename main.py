from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # React dev server
        "https://renderhostingtestfrontend.onrender.com",  # Render frontend
        "http://renderhostingtestfrontend.onrender.com"  # HTTP version
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Initialize the Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str

messages_history= []
# Routes
@app.get("/")
async def read_root():
    return {"status": "ok", "message": "Chat API is running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get the last user message
        last_message = request.messages[-1]
        if last_message.role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")
        
        # Generate response
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["You are a friendly assistant, given the message history and user message, your task is to reply the user message",f"message_history: {messages_history}", f"user_message: {last_message.content}"],
        )
        
        messages_history.append(last_message)
        messages_history.append(Message(role="assistant", content=response.text))
        if not response.text:
            raise HTTPException(status_code=500, detail="No response generated from Gemini API")
        
        return ChatResponse(response=response.text)
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in chat endpoint: {error_message}")
        
        # Handle region restriction error
        if "User location is not supported" in error_message:
            raise HTTPException(
                status_code=403,
                detail="The Gemini API is not available in your region. Please try using a VPN or contact support for assistance."
            )
        
        # Handle other API errors
        if "FAILED_PRECONDITION" in error_message:
            raise HTTPException(
                status_code=400,
                detail="Unable to process the request. Please check your API key and try again."
            )
        
        # Generic error for other cases
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again later."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 