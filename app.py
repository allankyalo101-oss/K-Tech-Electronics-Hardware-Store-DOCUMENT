"""
app.py — Kisha-Tech Electronics AI Backend
Hosted on Render at: kisha-tech-backend.onrender.com

Endpoints:
    POST /ask      — AI query from Volta OS admin frontend
    GET  /health   — uptime check (Volta OS pings this)

Architecture:
    - Groq key lives server-side — never exposed to browser
    - Receives: { message, context, history }
    - Returns:  { reply }
    - Full shop context (inventory + financials) sent from Volta OS frontend
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kishatech.backend")

# ── Env validation — fail loudly on startup if key is missing ─────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    logger.error("FATAL: GROQ_API_KEY not set. Set it in Render environment vars.")
    raise SystemExit("GROQ_API_KEY required")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

logger.info(f"Kisha-Tech backend starting | Groq key: {GROQ_API_KEY[:8]}...")
logger.info(f"Supabase: {'✓ configured' if SUPABASE_URL else '✗ not set'}")

# ── Flask + Groq ──────────────────────────────────────────────────────────────
app   = Flask(__name__)
CORS(app, origins=["https://kishatechadmin.vercel.app", "https://kishatech.vercel.app"])

client = Groq(api_key=GROQ_API_KEY)
MODEL  = "llama-3.3-70b-versatile"

SYSTEM_BASE = """You are Sarah, the AI assistant for Kisha-Tech Electronics & Hardware Store.
Location: Machakos Kenya Israel, opposite Manza College.
Hours: Mon–Sat 7:00 AM – 7:00 PM, Sun 9:00 AM – 5:00 PM.

You are the shop's financial and inventory advisor. The shop owner asks you questions about
their business data. Be direct, specific, and always use KSh figures.
Answer in plain English. Under 300 words unless a detailed breakdown is requested.
Never make up inventory items or prices — only use what's in the context provided."""


# ── /ask ──────────────────────────────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data    = request.get_json(force=True)
        message = (data.get("message") or "").strip()
        context = (data.get("context") or "").strip()
        history = data.get("history") or []

        if not message:
            return jsonify({"error": "message is required"}), 400

        logger.info(f"/ask | msg={message[:60]} | context_len={len(context)}")

        # Build messages array
        system_content = SYSTEM_BASE
        if context:
            system_content += f"\n\n=== SHOP DATA ===\n{context[:4000]}"

        messages = [{"role": "system", "content": system_content}]

        # Inject conversation history (cap at last 10 turns)
        for turn in history[-10:]:
            role    = turn.get("role", "user")
            content = str(turn.get("content", ""))[:600]
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
            timeout=15,
        )

        reply = completion.choices[0].message.content.strip()
        logger.info(f"/ask | reply={reply[:60]}")

        return jsonify({"reply": reply})

    except Exception as e:
        logger.error(f"/ask error: {repr(e)}")
        return jsonify({"error": "AI service temporarily unavailable. Try again."}), 500


# ── /health ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "kisha-tech-backend",
        "model": MODEL,
        "groq": "configured" if GROQ_API_KEY else "missing",
        "supabase": "configured" if SUPABASE_URL else "not set",
    })


# ── Root ──────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Kisha-Tech AI Backend — Sarah is live"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)