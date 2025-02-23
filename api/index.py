from flask import Flask, request, Response
import requests

app = Flask(__name__)

MET_API_BASE_URL = "https://api.met.no/weatherapi"

@app.route('/<path:subpath>', methods=["GET", "POST", "PUT", "DELETE"])
def proxy(subpath):
    """Forward the request to the MET API while keeping everything intact."""
    
    # Construct the full MET API URL
    target_url = f"{MET_API_BASE_URL}/{subpath}"
    
    # Forward all headers except `Host`
    headers = {key: value for key, value in request.headers if key.lower() != "host"}
    
    # Ensure we include the User-Agent (required by MET API)
    headers["User-Agent"] = "Yrweather/1.0 (jon39334@gmail.com)"
    
    # Forward the request to MET API
    response = requests.request(
        method=request.method,
        url=target_url,
        headers=headers,
        params=request.args,
        data=request.data
    )
    
    # Create a response with forwarded content and add CORS headers
    flask_response = Response(response.content, response.status_code, response.headers.items())
    flask_response.headers["Access-Control-Allow-Origin"] = "*"
    flask_response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    flask_response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    
    return flask_response