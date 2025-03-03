from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask Server!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    response = {"message": "Prediction result", "input_data": data}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
