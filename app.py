from flask import Flask, request, jsonify, render_template
from chat import get_response


app = Flask(__name__)

# Define a route for the index page using a GET request
@app.get('/')
def index_get():
    return render_template('base.html')
# Define a route for the prediction endpoint using a POST request
@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)