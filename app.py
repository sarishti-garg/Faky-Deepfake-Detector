import os

from flask import Flask, render_template, request, jsonify
from model import make_predictions

app = Flask(__name__)

model_paths = [
    "parameters/FFPP.pt",
    "parameters/MobileViT_FFPP.pt",
    "parameters/Mesonet_FFPP.pt"
]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        predictions = make_predictions(file_path, model_paths)
        return "Predictions: " + str(predictions)

if __name__ == '__main__':
    app.run(debug=True)
