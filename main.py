from fastapi import FastAPI, UploadFile, File
import tempfile
from faster_whisper import WhisperModel
import openai
import os
import json

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
You are an AI system that analyzes sales calls.

Analyze the following transcript and return JSON.

Transcript:
{transcript}

Return ONLY valid JSON like this:

{{
  "deal_score": 0,
  "summary": "short summary",
  "objections": ["list objections"],
  "coaching_tips": ["tip1", "tip2"]
}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content

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
