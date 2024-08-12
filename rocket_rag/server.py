from flask import Flask, request, jsonify
import uuid
from datetime import datetime
from agent import RagAgent

app = Flask(__name__)
# Handle CORS

placeholder_response_message = "This is a placeholder response message. It is here to simulate a response from a chatbot. It is not a real response."
rocket_rag = RagAgent('../store/nodes_20kg.pkl')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, DELETE')
    return response

# In-memory data storage for conversations
conversations = {}

# Helper function to create a unique identifier
def generate_id():
    return str(uuid.uuid4())

# Helper function to get the current time in ISO format
def get_current_time():
    return datetime.utcnow().isoformat()

# Route to create a new conversation
@app.route('/conversation', methods=['POST'])
def create_conversation():
    conversation_id = generate_id()
    created_at = get_current_time()
    conversation_header = {
        "id": conversation_id,
        "title": f"Conversation " + str(conversation_id),
        "createdAt": created_at
    }
    conversation = {
        "header": conversation_header,
        "messages": [],
        "attachments": []
    }
    conversations[conversation_id] = conversation
    return jsonify(conversation_header), 201  # Return only the header

# Route to get all conversation headers
@app.route('/conversationHeaders', methods=['GET'])
def get_conversation_headers():
    headers = [conv["header"] for conv in conversations.values()]
    return jsonify(headers), 200

# Route to delete a conversation
@app.route('/conversation/<string:conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    if conversation_id in conversations:
        del conversations[conversation_id]
        return jsonify({"message": "Conversation deleted", "status": 200}), 200
    else:
        return jsonify({"message": "Conversation not found", "status": 404}), 404

# Route to get full conversation data
@app.route('/conversationFull/<string:conversation_id>', methods=['GET'])
def load_full_conversation(conversation_id):
    if conversation_id in conversations:
        return jsonify(conversations[conversation_id]), 200
    else:
        return jsonify({"message": "Conversation not found", "status":404}), 404

# Route to send a message to a conversation
@app.route('/conversationMessage', methods=['POST'])
def send_message():
    data = request.json
    conversation_id = data["header"]["id"]
    message = data["message"]
    message["id"] = generate_id()  # Generate a unique ID for the message
    message_text = [
        {"role": "user", "content": message["text"]}
    ]
    
    if conversation_id in conversations:
        conversations[conversation_id]["messages"].append(message)
        gpt_response = rocket_rag.generate_response(prompts=message_text)
        print(gpt_response)
        response_str = gpt_response.choices[0].message.content
        print(response_str)
        conversations[conversation_id]["messages"].append({ "id": generate_id(), "text": response_str, "sender": "bot" })
        return conversations[conversation_id]
    else:
        return jsonify({"message": "Conversation not found", "status":404}), 404

# Route to send an attachment to a conversation
@app.route('/conversationAttachment', methods=['POST'])
def send_attachment():
    data = request.json
    conversation_id = data["header"]["id"]
    attachment = data["attachment"]
    attachment["id"] = generate_id()  # Generate a unique ID for the attachment
    
    if conversation_id in conversations:
        conversations[conversation_id]["attachments"].append(attachment)
        return jsonify({"message": "Attachment sent" , "status":200}), 200
    else:
        return jsonify({"error": "Conversation not found", "status":404}), 404

if __name__ == '__main__':
    result = rocket_rag.get_fault_prediction()
    app.run(debug=True)
