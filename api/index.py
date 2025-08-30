from flask import Flask, request, jsonify, Response, redirect
import requests
import os
from other.optimize import optimize

app = Flask(__name__)
app.config['CLIENT_SECRET'] = os.environ.get('CLIENT_SECRET', 'default_secret_key')


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
        flask_response = Response(response.text, content_type='text/plain')

        # Add CORS headers
        flask_response.headers.add("Access-Control-Allow-Origin", "*")
        flask_response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        flask_response.headers.add("Access-Control-Allow-Headers", "Content-Type")

        return flask_response, response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to fetch data", "details": str(e)}), 500


@app.route('/map/<path:subpath>', methods=['GET'])
def skisporet(subpath):
    """Proxy requests to skisporet"""
    query_string = request.query_string.decode("utf-8")
    skisporet_url = f"https://www.skisporet.no/map/{subpath}"
    if query_string:
        skisporet_url += f"?{query_string}"

    headers = {
        "Authorization": "Basic c2tpc3BvcmV0LWFwaTokUk5qYnEjYjUkZFBwUzNDMnBRNjQmdGNI",  # Replace with actual credentials securely
    }

    try:
        response = requests.get(skisporet_url, headers=headers)

        # Handle different response types
        try:
            data = response.json()
            flask_response = jsonify(data)
        except ValueError:
            flask_response = Response(response.text, content_type=response.headers.get('Content-Type', 'text/plain'))

        flask_response.headers.add("Access-Control-Allow-Origin", "*")
        flask_response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        flask_response.headers.add("Access-Control-Allow-Headers", "Content-Type")

        return flask_response, response.status_code

    except requests.exceptions.RequestException as e:
        error_response = jsonify({"error": "Failed to fetch data", "details": str(e)})
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        error_response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        error_response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return error_response, 500

@app.route('/about')
def about():
    return 'About'

@app.route('/strava/refresh', methods=['GET', 'OPTIONS'])
def get_refresh_token():
    """Exchanges authorization code for an access token and refresh token."""
    if request.method == 'OPTIONS':
        return handle_cors()

    code = request.args.get('code')
    
    if not code:
        return jsonify({'error': 'Missing authorization code'}), 400
    
    response = requests.post('https://www.strava.com/oauth/token', params={
        'client_id': '108568',
        'client_secret': app.config['CLIENT_SECRET'],
        'code': code,
        'grant_type': 'authorization_code'
    })
    
    return handle_response(response)

@app.route('/strava/exchange', methods=['GET', 'OPTIONS'])
def exchange_refresh_token():
    """Exchanges refresh token for a new access token."""
    if request.method == 'OPTIONS':
        return handle_cors()

    refresh_token = request.args.get('refresh_token')
    
    if not refresh_token:
        return jsonify({'error': 'Missing refresh token'}), 400
    
    response = requests.post('https://www.strava.com/api/v3/oauth/token', params={
        'client_id': '108568',
        'client_secret': app.config['CLIENT_SECRET'],
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token'
    })

    return handle_response(response)

def handle_response(response):
    """Handles API response and applies CORS headers."""
    try:
        response_data = response.json()
    except ValueError:
        return jsonify({'error': 'Invalid JSON response from Strava'}), 500
    
    flask_response = jsonify(response_data)
    flask_response.status_code = response.status_code
    return add_cors_headers(flask_response)

def handle_cors():
    """Handles CORS preflight requests."""
    response = jsonify({'message': 'CORS preflight successful'})
    return add_cors_headers(response)

def add_cors_headers(response):
    """Adds CORS headers to response."""
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response

@app.route("/redirect", methods=['GET', 'OPTIONS'])
def strava_redirect():
    if request.method == 'OPTIONS':
        return handle_cors()
    user_agent = request.headers.get("User-Agent", "").lower()
    query_params = request.query_string.decode()

    if "android" in user_agent:
        return add_cors_headers(redirect(f"com.joneliasmewoen.yrweather://settings?{query_params}"))
    else:
        return add_cors_headers(redirect(f"https://yrweather.expo.app/settings?{query_params}"))
    
@app.route("/optimize", methods=['POST', 'OPTIONS'])
def hrv_acwr_opt():
    if request.method == 'OPTIONS':
        return handle_cors()
    data = request.get_json()
    hrv = data.get("hrv")
    ctl = data.get("ctl")
    acrs = data.get("acwr")
    print(data)
    result = optimize(acrs, ctl, hrv)
    return add_cors_headers(jsonify(result))

#if __name__ == "__main__":
#    app.run(debug=True, port=5000)
    