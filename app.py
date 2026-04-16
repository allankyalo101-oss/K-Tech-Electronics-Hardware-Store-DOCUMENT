# ── LOAD ENV ─────────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── IMPORTS ──────────────────────────────────────────────
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
from supabase import create_client

# ── INIT ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── ENV ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env")
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL not set in .env")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY not set in .env")

# ── SUPABASE ──────────────────────────────────────────────
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    raise RuntimeError(f"Supabase init failed: {str(e)}")

# ── HELPERS ───────────────────────────────────────────────
def get_inventory():
    try:
        res = supabase.table("inventory").select("*").execute()
        return res.data if res.data else []
    except Exception as e:
        print("[INVENTORY] Fetch error:", str(e))
        return []

def compute_metrics(items):
    try:
        total_items = len(items)
        total_stock_value = sum(
            (i.get("sell_price", 0) or 0) * (i.get("qty", 0) or 0)
            for i in items
        )
        low_stock = [i for i in items if (i.get("qty", 0) or 0) <= 5]
        return {
            "total_items":      total_items,
            "total_stock_value": total_stock_value,
            "low_stock_count":  len(low_stock),
        }
    except Exception as e:
        print("[METRICS] Error:", str(e))
        return {}

def call_groq(messages_payload: list, max_tokens: int = 800) -> str:
    """Single Groq call. Raises on failure."""
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":       "llama-3.3-70b-versatile",
            "messages":    messages_payload,
            "max_tokens":  max_tokens,
            "temperature": 0.3,
        },
        timeout=30,
    )
    if response.status_code != 200:
        print("[GROQ] Error:", response.text)
        raise RuntimeError(f"Groq API error {response.status_code}")
    data = response.json()
    return (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "No response from AI")
    )


# ── /ask — main AI endpoint ───────────────────────────────
#
# Accepts two modes:
#
# Mode A — lightweight (frontend sends only the message):
#   { "message": "What is my gross margin?" }
#   Backend builds context from Supabase inventory.
#
# Mode B — rich (frontend sends full context + conversation history):
#   {
#     "message":  "What should I restock urgently?",
#     "context":  "<full shop context string built by frontend>",
#     "history":  [{ "role": "user"|"assistant", "content": "..." }, ...]
#   }
#   Backend uses the provided context directly — no Supabase call needed.
#
# Both modes return: { "reply": "<string>" }

@app.route("/ask", methods=["POST"])
def ask_ai():
    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "No JSON body"}), 400

        user_input = (body.get("message") or "").strip()
        if not user_input:
            return jsonify({"error": "message cannot be empty"}), 400

        # ── Mode B: frontend-provided context (richer, preferred) ──────────
        rich_context = (body.get("context") or "").strip()
        history      = body.get("history") or []

        if rich_context:
            # Frontend sent full shop context — use it directly
            system_content = (
                f"You are a sharp, concise financial AI advisor for an electronics "
                f"and hardware shop in Machakos, Kenya (Kisha-Tech).\n"
                f"Answer using real numbers. Be direct, specific, practical. "
                f"Always quote KSh figures. Under 200 words unless a breakdown is needed.\n\n"
                f"=== SHOP DATA ===\n{rich_context}"
            )
            messages_payload = [{"role": "system", "content": system_content}]
            # Replay conversation history so context is maintained
            for turn in history[-10:]:   # cap at 10 turns to stay within token budget
                if turn.get("role") in ("user", "assistant") and turn.get("content"):
                    messages_payload.append({
                        "role":    turn["role"],
                        "content": str(turn["content"])[:800],  # truncate long turns
                    })
            messages_payload.append({"role": "user", "content": user_input})

        else:
            # ── Mode A: lightweight — build context from Supabase ───────────
            inventory = get_inventory()
            metrics   = compute_metrics(inventory)
            sample    = inventory[:50]

            system_content = (
                f"You are a professional electronics shop analyst for Kisha-Tech, Machakos, Kenya.\n\n"
                f"Business metrics:\n{metrics}\n\n"
                f"Inventory sample ({len(inventory)} total items):\n{sample}\n\n"
                f"Rules:\n- Be concise\n- Use KSh figures\n- Give actionable advice"
            )
            messages_payload = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_input},
            ]

        reply = call_groq(messages_payload)
        return jsonify({"reply": reply})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500
    except Exception as e:
        print("[/ask] Server error:", str(e))
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ── /health ───────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Lightweight check — no Supabase call, just confirms server is up."""
    return jsonify({"status": "ok", "service": "Kisha-Tech AI Backend"})


# ── / ─────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Kisha-Tech AI backend running"})


# ── RUN ───────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)