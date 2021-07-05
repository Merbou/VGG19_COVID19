from tensorflow import keras
from tensorflow.keras.layers import Softmax,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
import tensorflow as tf
import numpy as np
import json
import cv2
import os
from vendor.preprocessing.image import AjustStuckImage
from dotenv import dotenv_values
from model.segmentor import Lung
ln=Lung()

config = dotenv_values(".env")


class Classfier():
    def __init__(self,size=512,device='/device:CPU:0'):
        self.size=size

        if device == '/device:CPU:0':
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.device=device
        self.load()


    def train(self,train_dataset,test_dataset,_epochs=12,_patience=5):
        self.defineModel()

        with tf.device(self.device):
            earlystopping = EarlyStopping(monitor ="val_loss", 
                                                    mode ="min", patience = _patience, 
                                                    restore_best_weights = True)
            self.model.fit(train_dataset,epochs=_epochs,validation_data=test_dataset,callbacks=[earlystopping])
        self.save(train_dataset.class_indices)

    def defineModel(self):
        self.createArchitecture()
        self.model.compile(loss='categorical_crossentropy',optimizer=Adamax(learning_rate=1e-5),metrics=['accuracy'])
        return self.model

    def createArchitecture(self):
        base_model=tf.keras.applications.VGG19(
            include_top=True,
            weights="imagenet",
            input_shape=(224,224,3)
        )
        base_model.trainable = True
        base_input=base_model.layers[0].input
        
        base_outputs=base_model.layers[-3].input
        base_outputs=Dense(4096,kernel_regularizer=regularizers.l2(l2=1e-5))(base_outputs)
        base_outputs=Dropout(.2)(base_outputs)
        base_outputs=Dense(4096,kernel_regularizer=regularizers.l2(l2=1e-5))(base_outputs)
        base_outputs=Dense(2)(base_outputs)

        final_outputs=Softmax()(base_outputs)
        
        self.model = keras.Model(inputs=base_input, outputs=final_outputs)
        #self.model.summary()
        return self.model


    def save(self,_class):
        self.model.save(config['STORE']+"model")
        a_file = open(config['STORE']+"model/class.json", "w")
        json.dump(_class, a_file)
        a_file.close()
    
    def load(self):
        with tf.device(self.device):
            self.model = tf.keras.models.load_model(config['STORE']+"model")
        with open(config['STORE']+"model/class.json") as json_file:
            self.labels = json.load(json_file)
        
    def predict(self,img_path,show=False):
        labels = list(self.labels.keys())
        # code_label = list(self.labels.values())
        lungImg=self.__LoadImage(img_path,show)
        predicted_list=self.model.predict(lungImg)
        # predicted_list = [ '%.2f' % elem for elem in predicted_list ]
        # index = code_label.index(np.argmax(predicted_list))
        return predicted_list,labels

    def predictList(self,imgs,show=False):
        imgs=ln.predictList(imgs)
        labels = list(self.labels.keys())
        imgs=np.array([AjustStuckImage(img)/255 for img in imgs])
        print(imgs.shape)
        predicted_lists=self.model.predict(imgs)
        predicted_lists = [[ '%.2f' % elem for elem in predicted_list ] for predicted_list in predicted_lists]

        return predicted_lists,labels


    def __LoadImage(self,img_path, _show):
        img=pathOrDataImage(img_path)
        img=ln.predictedLung(img,size_output=self.size,show=_show)
        img=AjustStuckImage(img)/255
        return np.expand_dims(img, axis=0)

def pathOrDataImage(inp):
    if type(inp) == str:
        return cv2.imread(inp,0)
    return cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)