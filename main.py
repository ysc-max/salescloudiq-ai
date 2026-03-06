from fastapi import FastAPI, UploadFile, File
import tempfile
from faster_whisper import WhisperModel
import openai
import os

app = FastAPI()

model = WhisperModel("base")

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/")
def root():
    return {"status": "SalesCloud IQ AI running"}

@app.post("/analyze-call")
async def analyze_call(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        contents = await file.read()
        temp.write(contents)
        audio_path = temp.name

    segments, _ = model.transcribe(audio_path)

    transcript = " ".join([segment.text for segment in segments])

    prompt = f"""
You are an AI sales call analysis system.

Analyze the following transcript.

Transcript:
{transcript}

Return ONLY valid JSON in this format:

{{
  "deal_score": number between 0 and 100,
  "summary": "short summary of the call",
  "objections": ["list of objections mentioned"],
  "coaching_tips": ["actionable tips for the sales rep"]
}}
"""
    
Analyze the following sales call transcript.

Transcript:
{transcript}

Return:

1. Deal Score (0-100)
2. Objections detected
3. Coaching tips for the sales rep
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    analysis = response.choices[0].message.content

    return {
        "transcript": transcript,
        "analysis": analysis
    }
