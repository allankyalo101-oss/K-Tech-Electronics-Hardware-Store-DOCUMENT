# ── LOAD ENV (HARDENED) ─────────────────────────────────────
import os
import time
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# ── IMPORTS ──────────────────────────────────────────────────
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
from supabase import create_client
import { Analytics } from "@vercel/analytics/next"

# ── APP INIT ─────────────────────────────────────────────────
app = Flask(__name__)

# CORS: keep open for now (can lock to domains later)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── ENV CONFIG (STRICT) ──────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not all([GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise EnvironmentError("❌ Missing required environment variables")

# ── SUPABASE INIT (SAFE FAIL) ────────────────────────────────
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    raise RuntimeError(f"❌ Supabase init failed: {str(e)}")

# ── BASIC RATE LIMIT (IN-MEMORY) ─────────────────────────────
# NOTE: upgrade to Redis later for scale
REQUEST_LOG = {}
RATE_LIMIT_SECONDS = 2  # per IP cooldown

def is_rate_limited(ip: str) -> bool:
    now = time.time()
    last = REQUEST_LOG.get(ip, 0)
    if now - last < RATE_LIMIT_SECONDS:
        return True
    REQUEST_LOG[ip] = now
    return False

# ── INVENTORY FETCH (SAFE) ───────────────────────────────────
def get_inventory():
    try:
        res = supabase.table("inventory").select(
            "name, category, sell_price, qty, unit"
        ).execute()

        return res.data or []

    except Exception as e:
        print("Inventory fetch error:", str(e))
        return []

# ── METRICS ENGINE ───────────────────────────────────────────
def compute_metrics(items):
    try:
        total_items = len(items)

        total_stock_value = sum(
            (i.get("sell_price") or 0) * (i.get("qty") or 0)
            for i in items
        )

        low_stock_count = sum(
            1 for i in items if (i.get("qty") or 0) <= 5
        )

        return {
            "total_items": total_items,
            "total_stock_value": round(total_stock_value, 2),
            "low_stock_count": low_stock_count,
        }

    except Exception as e:
        print("Metrics error:", str(e))
        return {
            "total_items": 0,
            "total_stock_value": 0,
            "low_stock_count": 0,
        }

# ── INPUT VALIDATION ─────────────────────────────────────────
def validate_input(data):
    if not data:
        return None
    msg = data.get("message", "")
    msg = msg.strip()
    if len(msg) < 1:
        return None
    if len(msg) > 500:
        return None
    return msg

# ── GROQ CALL WRAPPER (RESILIENT) ────────────────────────────
def call_groq(system_prompt, user_input):
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.3
            },
            timeout=25
        )

        if response.status_code != 200:
            print("Groq error:", response.text)
            return None

        data = response.json()

        return (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", None)
        )

    except Exception as e:
        print("Groq request failed:", str(e))
        return None

# ── MAIN AI ENDPOINT ─────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask_ai():
    try:
        ip = request.remote_addr

        # Rate limit protection
        if is_rate_limited(ip):
            return jsonify({
                "error": "Too many requests. Slow down."
            }), 429

        body = request.get_json(silent=True)
        user_input = validate_input(body)

        if not user_input:
            return jsonify({"error": "Invalid or empty message"}), 400

        # Fetch data layer
        inventory = get_inventory()
        metrics = compute_metrics(inventory)

        # Trim dataset (protect tokens)
        sample = inventory[:40]

        system_prompt = f"""
You are a high-precision retail intelligence engine.

Return structured, actionable business insight.

BUSINESS METRICS:
{metrics}

INVENTORY SAMPLE:
{sample}

RULES:
- Be concise
- Use numbers where possible
- Prioritize profit actions, restock signals, and demand patterns
- Avoid fluff
"""

        reply = call_groq(system_prompt, user_input)

        if not reply:
            return jsonify({
                "error": "AI service unavailable"
            }), 503

        return jsonify({
            "reply": reply,
            "meta": {
                "items_analyzed": len(sample),
                "timestamp": time.time()
            }
        })

    except Exception as e:
        print("Server crash:", str(e))
        return jsonify({
            "error": "Internal server error"
        }), 500

# ── HEALTH CHECK ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "KISHA-TECH AI backend online",
        "version": "1.0.0"
    })

# ── START SERVER ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )