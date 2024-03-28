import helperRT
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    # Extract user data from request
    data = request.get_json()
    user_id = data['user_id']
    # Load and preprocess user data here
    # Implement TensorRT inference here
    # Return movie recommendations as JSON
    return jsonify({"recommendations": ["Movie 1", "Movie 2", "Movie 3"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
