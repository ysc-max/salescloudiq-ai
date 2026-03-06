from fastapi import FastAPI, UploadFile, File
import tempfile
from faster_whisper import WhisperModel
import openai
import os
import json

app = FastAPI()

# Load Whisper model (lightweight)
model = WhisperModel("tiny")

# OpenAI key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.get("/")
def root():
    return {"status": "SalesCloud IQ AI running"}


@app.post("/analyze-call")
async def analyze_call(file: UploadFile = File(...)):

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        contents = await file.read()
        temp.write(contents)
        audio_path = temp.name

    # Transcribe audio
    segments, _ = model.transcribe(audio_path)
    transcript = " ".join([segment.text for segment in segments])

    # Build AI prompt
    prompt = (
        "You are an AI system that analyzes sales calls.\n\n"
        "Analyze the following transcript and return JSON.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Return ONLY valid JSON in this format:\n"
        "{"
        "\"deal_score\": number between 0 and 100,"
        "\"summary\": \"short summary of the call\","
        "\"objections\": [\"list of objections mentioned\"],"
        "\"coaching_tips\": [\"actionable tips for the sales rep\"]"
        "}"
    )

    # Call OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content

    # Try to parse JSON safely
    try:
        analysis_json = json.loads(content)
    except:
        analysis_json = {
            "deal_score": 0,
            "summary": content,
            "objections": [],
            "coaching_tips": []
        }

    return {
        "transcript": transcript,
        "analysis": analysis_json
    }
