"""
app.py — Kisha-Tech Electronics AI Backend
Render: kisha-tech-backend.onrender.com

Frontend sends:  { message: str, context: str, history?: [] }
Backend returns: { reply: str }

Sarah is Zovrix property. Name never changes per client.
"""

import os
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kishatech.backend")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    logger.error("FATAL: GROQ_API_KEY not set")
    raise SystemExit("Missing GROQ_API_KEY")

logger.info(f"=== Kisha-Tech Backend Starting === Groq: {GROQ_API_KEY[:8]}...")

app = Flask(__name__)
CORS(app, origins=["https://kishatechadmin.vercel.app", "https://kishatech.vercel.app", "*"])

client = Groq(api_key=GROQ_API_KEY)
MODEL  = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are Sarah, AI financial advisor for Kisha-Tech Electronics & Hardware Store, Machakos Kenya.

The shop owner is asking you questions about their business.
You will receive full shop data as context — transactions, inventory, financials.
Use ONLY the provided data. Never invent numbers.
Be direct, specific, short. Always quote KSh figures.
Under 200 words unless a breakdown is explicitly needed."""


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "kisha-tech-backend", "timestamp": int(time.time())}), 200


@app.route("/ask", methods=["POST"])
def ask():
    start = time.time()
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        body    = request.get_json(silent=True) or {}
        message = (body.get("message") or "").strip()
        context = (body.get("context") or "").strip()
        history = body.get("history") or []

        if not message:
            return jsonify({"error": "message is required"}), 400

        logger.info(f"/ask | msg={message[:80]} | ctx={len(context)} chars")

        system_content = SYSTEM_PROMPT
        if context:
            system_content += f"\n\n=== SHOP DATA ===\n{context[:6000]}"

        messages = [{"role": "system", "content": system_content}]

        for turn in history[-10:]:
            role    = turn.get("role", "")
            content = str(turn.get("content", ""))[:800]
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
            timeout=20,
        )

        reply = completion.choices[0].message.content.strip()
        ms    = round((time.time() - start) * 1000)

        logger.info(f"/ask | {ms}ms | reply={reply[:80]}")
        return jsonify({"reply": reply, "latency_ms": ms})

    except Exception as e:
        logger.exception("ASK ERROR")
        return jsonify({"error": "AI service error", "detail": str(e)}), 500


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "Kisha-Tech AI — Sarah is live"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Listening on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)