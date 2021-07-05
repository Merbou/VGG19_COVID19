from flask import Flask, request, render_template, jsonify, json,make_response
from flask_cors import CORS,cross_origin
from model.detector import Classfier
from dotenv import dotenv_values
import cv2
import numpy as np

config = dotenv_values(".env")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
cors = CORS(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if len(request.files)>int(config['MAX_PREDICTION']):
        Exception("Do not upload more than 100 images")
        return make_response(jsonify({"errors": 'Do not upload more than 100 images'}),400)
    imgs=[]
    for  bnr_img_name,bnr_img in request.files.items():
        if not allowed_file(bnr_img.filename):
            Exception("Type of file not allowed")
            return make_response(jsonify({"errors": 'Type of file not allowed'}),400)

        imgs.append(cv2.imdecode(np.fromfile(bnr_img, np.uint8), cv2.IMREAD_COLOR))
    size=224
    clsf=Classfier(size=size)
    predicted,labels=clsf.predictList(imgs)
    _response=jsonify({"predicted_list": predicted,"labels": labels})
    return make_response(_response, 200, {'ContentType':'application/json'})

if __name__ == "__main__":
    print('Starting Server...')
    app.run(debug=True)
