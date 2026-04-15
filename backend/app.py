from flask import Flask, request, jsonify
import requests
import os
from flask_cors import CORS
from supabase import create_client

app = Flask(__name__)
CORS(app)

# ── ENV CONFIG ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SUPABASE_URL = "https://irywiqzlirxkjrolbjvy.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── LOAD INVENTORY ─────────────────────────
def get_inventory():
    res = supabase.table("inventory").select("*").execute()
    return res.data if res.data else []

# ── AI ENDPOINT ────────────────────────────
@app.route("/ask", methods=["POST"])
def ask_ai():
    user_input = request.json.get("message", "")

    inventory = get_inventory()

    # Limit data for token efficiency
    sample = inventory[:50]

    context = f"""
    You are an electronics shop business analyst.

    Inventory data:
    {sample}

    Answer clearly using numbers where possible.
    """

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": user_input}
            ]
        }
    )

    data = response.json()
    reply = data["choices"][0]["message"]["content"]

    return jsonify({"reply": reply})

# ── RUN ────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)