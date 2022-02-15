from flask import Flask,jsonify,request
from classifier import getPrediction

app = Flask(__name__)
@app.route('/predict',methods=['POST'])
def predict():
    image = request.files.get("image")
    prediction = getPrediction(image)
    return jsonify(({
        "prediction":prediction
    })),200

if __name__ == '__main__':
    app.run(debug=True)