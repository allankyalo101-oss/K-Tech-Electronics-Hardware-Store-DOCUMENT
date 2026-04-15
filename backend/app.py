from flask import Flask, request, jsonify
import requests
import os
from flask_cors import CORS
from supabase import create_client

app = Flask(__name__)
CORS(app)

# ── ENV CONFIG ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validate required environment variables
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY environment variable not set")

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Supabase client: {str(e)}")

# ── LOAD INVENTORY ─────────────────────────
def get_inventory():
    try:
        res = supabase.table("inventory").select("*").execute()
        return res.data if res.data else []
    except Exception as e:
        print(f"Error fetching inventory: {str(e)}")
        return []

# ── AI ENDPOINT ────────────────────────────
@app.route("/ask", methods=["POST"])
def ask_ai():
    try:
        user_input = request.json.get("message", "").strip() if request.json else ""
        
        if not user_input:
            return jsonify({"error": "Message cannot be empty"}), 400

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
            },
            timeout=30
        )

        if response.status_code != 200:
            return jsonify({"error": f"Groq API error: {response.status_code}"}), 500

        data = response.json()
        if "choices" not in data or not data["choices"]:
            return jsonify({"error": "Invalid response from Groq API"}), 500
        
        reply = data["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})
    
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ── RUN ────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)