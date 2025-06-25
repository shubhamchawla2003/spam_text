from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']
    transformed = vectorizer.transform([message])
    result = model.predict(transformed)
    prediction = 'Spam' if result[0] == 1 else 'Not Spam'
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(port=3000,debug=True)
