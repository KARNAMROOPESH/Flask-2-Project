from flask import Flask , request , jsonify
from predict import getprediction

web = Flask(__name__)

@web.route('/get-prediction' , methods = ["POST"])
def get_prediction():
    image = request.files.get("Digit")
    prediction = getprediction(image)
    return jsonify({
        "prediction": prediction
    },200)

if __name__ == "__main__" :
    web.run(debug=True)
