import json
import uuid
from flask import Flask, render_template, request, Response, stream_with_context
from openai import OpenAI

app = Flask(__name__)

# LM Studio chạy local ở port 1234 theo mặc định
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # LM Studio không cần key thật, nhưng bắt buộc phải có
)

# Lưu conversations theo session_id (in-memory, reset khi restart server)
conversations = {}


def get_model():
    """Lấy model đầu tiên đang chạy trong LM Studio."""
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
        return None
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    result = []
    for sid, data in conversations.items():
        if data["messages"]:
            first_user_msg = next(
                (m["content"] for m in data["messages"] if m["role"] == "user"), "Cuộc trò chuyện mới"
            )
            result.append({
                "id": sid,
                "title": first_user_msg[:50] + ("..." if len(first_user_msg) > 50 else ""),
                "message_count": len(data["messages"])
            })
    return json.dumps(result[::-1])  # mới nhất lên đầu


@app.route("/api/conversations", methods=["POST"])
def new_conversation():
    sid = str(uuid.uuid4())
    conversations[sid] = {"messages": []}
    return json.dumps({"id": sid})


@app.route("/api/conversations/<sid>", methods=["GET"])
def get_conversation(sid):
    if sid not in conversations:
        return json.dumps({"error": "Không tìm thấy"}), 404
    return json.dumps(conversations[sid]["messages"])


@app.route("/api/conversations/<sid>", methods=["DELETE"])
def delete_conversation(sid):
    if sid in conversations:
        del conversations[sid]
    return json.dumps({"ok": True})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    sid = data.get("session_id")
    user_message = data.get("message", "").strip()

    if not user_message:
        return json.dumps({"error": "Tin nhắn trống"}), 400

    if sid not in conversations:
        conversations[sid] = {"messages": []}

    conversations[sid]["messages"].append({"role": "user", "content": user_message})

    model = get_model()
    if not model:
        return json.dumps({"error": "LM Studio chưa chạy hoặc chưa load model"}), 503

    def generate():
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=conversations[sid]["messages"],
                stream=True,
                temperature=0.7,
                max_tokens=2048,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                yield f"data: {json.dumps({'delta': delta})}\n\n"

            conversations[sid]["messages"].append({"role": "assistant", "content": full_response})
            yield f"data: {json.dumps({'done': True, 'session_id': sid})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/status", methods=["GET"])
def status():
    model = get_model()
    return json.dumps({"model": model, "online": model is not None})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
