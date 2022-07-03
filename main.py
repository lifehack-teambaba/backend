from flask import Flask, request, jsonify
from flask_cors import CORS
import flair_script

app = Flask(__name__)
CORS(app)

print('Initialising classifier')
classifier = flair_script.init_classifier()
print('Classifier initialised')

@app.route('/')
def test():
    return "Server is running!"

@app.route('/generate-analysis', methods=['POST'])
def generate_analysis():
    data = request.get_json()
    transcript = data['transcript']
    # print(transcript)
    value, score = flair_script.predict(transcript, classifier)
    return jsonify(
        value = value,
        score = score
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)

