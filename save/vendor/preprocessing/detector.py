from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from dotenv import dotenv_values
from model.segmentor import Lung
import numpy as np
ln = Lung()
config = dotenv_values(".env")
def DataGenerator(size):
    data_gen_args = dict(rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest') 
    #generatation data useing LungSeparator as preprocessing image for extract only lung
    train=ImageDataGenerator(**data_gen_args,preprocessing_function=onlyLung)
    test=ImageDataGenerator(preprocessing_function=onlyLung)

    ds_train=train.flow_from_directory(config['GLOBAL_TRAIN_DATA_PATH'],color_mode='grayscale',target_size=size,batch_size=4,class_mode="categorical")
    ds_test=test.flow_from_directory(config['GLOBAL_TEST_DATA_PATH'],color_mode='grayscale',target_size=size,batch_size=4,class_mode="categorical")
    return ds_train,ds_test

def onlyLung(img):
    img_lung=ln.predictedLung(img,size_output=512)
    return img_lung[:,:,np.newaxis]/255



    