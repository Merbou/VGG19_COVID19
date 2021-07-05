from flask import Flask, request, render_template, jsonify, json,make_response
from flask_cors import CORS,cross_origin
# from vendor.preprocessing.detector import DataGenerator
# train_dataset,test_dataset=DataGenerator((size,size))
from model.detector import Classfier
import cv2
import numpy as np
app = Flask(__name__)
cors = CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    imgs=[cv2.imdecode(np.fromfile(bnr_img, np.uint8), cv2.IMREAD_COLOR) for bnr_img_name,bnr_img in request.files.items()]
    size=224
    clsf=Classfier(size=size)
    predicted,labels=clsf.predictList(imgs)
    _response=jsonify({"predicted_list": predicted,"labels": labels})
    return make_response(_response, 200, {'ContentType':'application/json'})

if __name__ == "__main__":
    print('Starting Server...')
    app.run(debug=True)