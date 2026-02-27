import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai

# Gemini API 설정
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
genai.configure(api_key=api_key)

app = FastAPI()

# CORS 설정 (클라우드 환경에서 발생할 수 있는 통신 차단 방지)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

system_instruction = "번역: [source]->[target]. 입력된 텍스트를 지정된 대상 언어로만 번역할 것. 부가 설명 금지."
generation_config = {
    "temperature": 0.1,
    "max_output_tokens": 512,
}

model = genai.GenerativeModel(
    model_name="gemini-flash-lite-latest",
    generation_config=generation_config,
    system_instruction=system_instruction
)

@app.post("/translate")
async def translate_text(req: TranslationRequest):
    try:
        prompt = f"[{req.source_lang}]->[{req.target_lang}]\n{req.text}"
        response = model.generate_content(prompt)
        return {"translated_text": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())