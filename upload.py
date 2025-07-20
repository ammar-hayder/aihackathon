from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = "orders.csv"

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    try:
        file_path = os.path.join(UPLOAD_FOLDER, CSV_FILENAME)
        file.save(file_path)
        return jsonify({"message": f"File uploaded successfully as {CSV_FILENAME}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)  # runs on different port from app.py
