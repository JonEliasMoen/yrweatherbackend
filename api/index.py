from flask import Flask, request, jsonify, Response
import requests
app = Flask(__name__)

@app.route('/<path:subpath>', methods=['GET', 'OPTIONS'])
def proxy(subpath):
    """ Forwards requests to MET API with CORS handling """
    query_string = request.query_string.decode("utf-8")
    met_url = f"https://api.met.no/weatherapi/{subpath}"
    if query_string:
        met_url += f"?{query_string}"

    headers = {
        "User-Agent": "YourApp/1.0 (your@email.com)",  # MET API requires User-Agent
    }

    try:
        response = requests.get(met_url, headers=headers)
        flask_response = jsonify(response.json())

        # Add CORS headers
        flask_response.headers.add("Access-Control-Allow-Origin", "*")
        flask_response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        flask_response.headers.add("Access-Control-Allow-Headers", "Content-Type")

        return flask_response, response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to fetch data", "details": str(e)}), 500
@app.route('/tidalwater/<version>/', methods=['GET'])
def tidalwater(version):
    """ Returns raw text from MET API for tidalwater """
    query_string = request.query_string.decode("utf-8")
    met_url = f"https://api.met.no/weatherapi/tidalwater/{version}/"
    if query_string:
        met_url += f"?{query_string}"

    headers = {
        "User-Agent": "YourApp/1.0 (your@email.com)",  # MET API requires User-Agent
    }

    try:
        response = requests.get(met_url, headers=headers)
        return Response(response.text, content_type='text/plain'), response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to fetch data", "details": str(e)}), 500


@app.route('/about')
def about():
    return 'About'