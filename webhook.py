import json
import os
import requests 
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
API_KEY = "TAKE_YOUR_KEY_FROM_HUGGING_FACE"

@app.route('/', methods=['GET', 'POST'])
def home():
    return 'OK', 200


def query_huggingface_api(query):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "inputs": query,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.8,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Hugging Face API error: {str(e)}"} 


@app.route('/dialogflow', methods=['POST'])
def dialogflow():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON input'}), 400

    print("Incoming request data:", json.dumps(data, indent=2))

    query = data.get('text')
    if not query:
        return jsonify({'error': 'No query text provided'}), 400

    print(f"Query received: {query}")

    hf_response = query_huggingface_api(query)

    if 'error' in hf_response:
        return jsonify({'error': hf_response['error']}), 500

    if isinstance(hf_response, list) and len(hf_response) > 0:
        generated_text = hf_response[0].get('generated_text', 'Sorry, I could not generate a response.')
    else:
        generated_text = 'Sorry, I could not generate a response.'

    return jsonify({
        'fulfillmentResponse': {
            'messages': [{'text': {'text': [generated_text]}}]
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
