from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
import tempfile
import os
import json

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"status": "SalesCloud IQ AI running"}

@app.post("/analyze-call")
async def analyze_call(file: UploadFile = File(...)):

    try:

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            contents = await file.read()
            temp.write(contents)
            audio_path = temp.name

        # Transcribe with OpenAI Whisper API (no local model)
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=(file.filename, audio_file, "audio/wav")
            )

        transcript = transcription.text

        prompt = f"""
You are an AI sales call analyzer.

Analyze this transcript and return JSON only.

Transcript:
{transcript}

Return format:

{{
"deal_score": number 0-100,
"summary": "short summary",
"objections": ["list objections"],
"coaching_tips": ["list coaching tips"]
}}
"""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        content = completion.choices[0].message.content

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

    except Exception as e:
        return {"error": str(e)}
