from vendor.preprocessing.segmentor import DataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from vendor.preprocessing.image import AjustImage,removeSmallestContour,MorphologicalTransf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
config = dotenv_values(".env")
class Lung():
    model=None
    def train(self,epochs=11,_patience=5):
        #create unet CNN model 
        self.model=self.modelGenerator()
        #create dict options of data augmentation
        data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')    

        #generate data train & test
        data_train = DataGenerator(2,config['GLOBAL_UNET_TRAIN_DATA_PATH'],'images','masks',data_gen_args)
        data_test = DataGenerator(2,config['GLOBAL_UNET_TEST_DATA_PATH'],'images','masks',data_gen_args)
        #train until over overfitting then save the best weights
        earlystopping = EarlyStopping(monitor ="loss",mode ="min", patience = _patience,restore_best_weights = True)
        #run training on gpu using CUDA
        with tf.device('/device:GPU:0'):
            self.model.fit(data_train,steps_per_epoch=300, epochs=epochs,callbacks=[earlystopping])
        # save model
        self.model.save(config['STORE']+"unet")
        # test model
        results = self.model.predict(data_test,30,verbose=1)
        print(results)
        return results

    def predictedLung(self,img,size_output=224,threshold=0.3,show=False):
        predict_pixel,img=self.predict(img,threshold)
        predict_pixel=MorphologicalTransf(predict_pixel)
        predict_pixel=cv2.resize(predict_pixel,(size_output,size_output), interpolation = cv2.INTER_AREA)
        img=cv2.resize(img,(size_output,size_output), interpolation = cv2.INTER_AREA)
        inter_img=cv2.multiply(img,predict_pixel)
        if show:
            plot([inter_img,img,predict_pixel])
        return removeSmallestContour(inter_img)

    def predict(self,img_path,threshold=0.3):
        if not self.model:
            self.model=self.load()
        img=self.__LoadImage(img_path)
        img_pre=AjustImage(img)

        sized_img=cv2.resize(img_pre,(256,256), interpolation = cv2.INTER_AREA)
        predict_pixel=binarization(self.model.predict(np.expand_dims(sized_img, axis=0))[0],threshold)
        img=img*255
        img=img.astype(np.uint8)
        predict_pixel=predict_pixel.astype(np.uint8)
        return predict_pixel,img


    def predictList(self,imgs,size_output=224,threshold=0.3):
        if not self.model:
            self.model=self.load()
        cont_imgs=len(imgs)
        imgs=np.array([cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(256,256),interpolation = cv2.INTER_AREA) for img in imgs])
        
        if cont_imgs>int(config['MAX_PREDICTION']):
            raise Exception("Number of images large then limit number")
        imgs_pre=np.array([AjustImage(img) for img in imgs])
        predict_pixels=self.model.predict(imgs_pre)
        predict_pixels=np.array([binarization(cv2.resize(predict_pixel,(size_output,size_output), interpolation = cv2.INTER_AREA),threshold) for predict_pixel in predict_pixels])

        imgs=imgs*255
        imgs=imgs.astype(np.uint8)
        predict_pixels=predict_pixels.astype(np.uint8)
        lung_imgs=[]
        for index in range(cont_imgs):
            lung_imgs.append(self.mergeLung(imgs[index],predict_pixels[index],size_output=size_output))
        return lung_imgs

    def mergeLung(self,img,predict_img,size_output=224):
        predict_img=MorphologicalTransf(predict_img)
        img=cv2.resize(img,(size_output,size_output), interpolation = cv2.INTER_AREA)
        inter_img=cv2.multiply(img,predict_img)

        return removeSmallestContour(inter_img)

  

    def __LoadImage(self,inp):
        #load image
        img=pathOrDataImage(inp)
        return img

    def modelGenerator(self,input_size = (256,256,1),nb_layer=5):
        inputs,kernal,layers = Input(input_size),32,[]
        poolLayer=inputs
        #generate convolutionals layers
        for l in range(1,nb_layer+1):
            kernal*=2
            convLayer,poolLayer,dropLayer=self.convolutionalGenerator(poolLayer,_kernel=kernal,withPool=False if l==nb_layer else True,withDrop=True if l-nb_layer==1 else False)
            layers.append(dropLayer if l-nb_layer==2 else convLayer)
        _inp=layers.pop()
        layers.reverse()

        #generate deconvolutionals layers
        for index,layer in enumerate(layers):
            kernal/=2
            _inp=self.upSamplingGenerator(layer,_inp,kernal) 

        #output of layers
        _inp = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(_inp)
        _inp = Conv2D(1, 1, activation = 'sigmoid')(_inp)   

        model = Model(inputs,_inp)
        model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

        return model    


    def convolutionalGenerator(self,inputs,_kernel=64,_filter=3,_activation = 'relu',_padding = 'same',_kernel_initializer = 'he_normal',_pool_size=(2,2),withPool=True,withDrop=False,_dropout=0.5):
        poolLayer,dropLayer=None,None
        convLayer = Conv2D(_kernel, _filter, activation = _activation, padding = _padding, kernel_initializer = _kernel_initializer)(inputs)
        convLayer = Conv2D(_kernel, _filter, activation = _activation, padding = _padding, kernel_initializer = _kernel_initializer)(convLayer)
        if withPool:
            poolLayer = MaxPooling2D(pool_size=_pool_size)(convLayer)
        if withDrop:
            dropLayer = Dropout(_dropout)(poolLayer if poolLayer else convLayer)
        return (convLayer,poolLayer,dropLayer)  

    def upSamplingGenerator(self,input_1,input_2,_kernel=512,_filter=3,_activation = 'relu',_padding = 'same',_kernel_initializer = 'he_normal',_pool_size=(2,2),LastUpSem=False):
        #deconvolutional
        input_2 = Conv2D(_kernel, _filter-1, activation = _activation, padding = _padding, kernel_initializer = _kernel_initializer)(UpSampling2D(size = _pool_size)(input_2))
        #skip connection
        input = concatenate([input_1,input_2], axis = 3)
        input = Conv2D(_kernel, _filter, activation = _activation, padding = _padding, kernel_initializer = _kernel_initializer)(input)
        input = Conv2D(_kernel, _filter, activation = _activation, padding = _padding, kernel_initializer = _kernel_initializer)(input)
        return input    



    def load(self):
        self.model = tf.keras.models.load_model(config['STORE']+"unet")
        return self.model    

def pathOrDataImage(inp):
    if type(inp) == str:
        return cv2.imread(inp)
    return inp

def binarization(img,th):
    img[img >= th] = 1
    img[img <= th] = 0
    return img


def plot(imgs):
    fig = plt.figure()
    for index,img in enumerate(imgs):
        ax = fig.add_subplot(1,len(imgs),index+1)
        ax.imshow(img)
    plt.show()
