from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/<path:subpath>')
def proxy(subpath):
    # Extract query parameters and reconstruct the full API URL
    query_string = request.query_string.decode("utf-8")
    url = f"https://api.met.no/weatherapi/{subpath}"
    if query_string:
        url += f"?{query_string}"

    headers = {
        "User-Agent": "YourApp/1.0 (your@email.com)",  # MET API requires a User-Agent
    }

    try:
        # Forward request to MET API
        response = requests.get(url, headers=headers)
        
        # Add CORS headers to allow requests from the frontend
        flask_response = jsonify(response.json())
        flask_response.headers.add("Access-Control-Allow-Origin", "*")
        flask_response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        flask_response.headers.add("Access-Control-Allow-Headers", "Content-Type")

        return flask_response, response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to fetch data", "details": str(e)}), 500

# Required for Vercel Serverless Functions
def handler(request):
    return app(request)
