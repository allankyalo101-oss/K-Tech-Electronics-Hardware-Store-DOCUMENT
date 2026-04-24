"""
app.py — Kisha-Tech Electronics AI Backend
Hosted on Render at: kisha-tech-backend.onrender.com
"""

import os
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kishatech.backend")

# ── ENV ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY missing")
    raise SystemExit("Missing GROQ_API_KEY")

logger.info("=== STARTUP ===")
logger.info(f"Groq: {'OK' if GROQ_API_KEY else 'MISSING'}")
logger.info(f"Supabase: {'OK' if SUPABASE_URL else 'NOT SET'}")

# ── APP ───────────────────────────────────────────────
app = Flask(__name__)

CORS(app, origins=[
    "https://kishatechadmin.vercel.app",
    "https://kishatech.vercel.app",
    "*"
])

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

SYSTEM_BASE = """
You are Sarah, AI assistant for Kisha-Tech Electronics.
Always respond in clear business intelligence format.
Use KSh for money.
Never hallucinate inventory.
"""


# ───────────────────────────────────────────────────────
# HEALTH (CRITICAL — frontend depends on this)
# ───────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "kisha-tech-backend",
        "timestamp": int(time.time())
    }), 200


# ───────────────────────────────────────────────────────
# FULL HEALTH DEBUG
# ───────────────────────────────────────────────────────
@app.route("/health/full", methods=["GET"])
def health_full():
    return jsonify({
        "status": "ok",
        "service": "kisha-tech-backend",
        "groq": bool(GROQ_API_KEY),
        "supabase": bool(SUPABASE_URL),
        "model": MODEL
    }), 200


# ───────────────────────────────────────────────────────
# AI ENDPOINT
# ───────────────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    start = time.time()

    try:
        if not request.is_json:
            return jsonify({"error": "JSON required"}), 400

        data = request.get_json() or {}
        message = (data.get("message") or "").strip()

        if not message:
            return jsonify({"error": "message required"}), 400

        messages = [
            {"role": "system", "content": SYSTEM_BASE},
            {"role": "user", "content": message}
        ]

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )

        reply = completion.choices[0].message.content.strip()

        return jsonify({
            "status": "ok",
            "reply": reply,
            "latency_ms": round((time.time() - start) * 1000)
        }), 200

    except Exception as e:
        logger.exception("ASK ERROR")
        return jsonify({
            "status": "error",
            "message": "AI failure",
            "detail": str(e)
        }), 500


# ───────────────────────────────────────────────────────
# ROOT
# ───────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "service": "kisha-tech-backend",
        "message": "Sarah AI live"
    })


# ───────────────────────────────────────────────────────
# START
# ───────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Running on {port}")
    app.run(host="0.0.0.0", port=port, debug=False)