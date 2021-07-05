from vendor.preprocessing.image import AjustStuckImage
from dotenv import dotenv_values
from model.segmentor import Lung
import matplotlib.pyplot as plt

from progress.bar import IncrementalBar as Bar
import cv2
import os
config = dotenv_values(".env")
ln=Lung()


data_train_path=config['GLOBAL_TRAIN_DATA_PATH']
data_test_path=config['GLOBAL_TEST_DATA_PATH']

onlyLung_data_train_path=config['GLOBAL_TRAIN_ONLYLUNG_DATA_PATH']
onlyLung_data_test_path=config['GLOBAL_TEST_ONLYLUNG_DATA_PATH']

def saveOnlyLungFile(filePath,save_to_path,size_output=512):
    img=cv2.imread(filePath,0)
    img_lung=ln.predictedLung(img,size_output=512)
    img_lung=AjustStuckImage(img_lung)
    cv2.imwrite(save_to_path+"img_lung.jpg", img_lung)

def saveOnlyLung(data_path,save_to_path,size_output=512):
    for class_path in os.listdir(data_path):
        imgs_path=os.listdir(data_path+"/"+class_path)
        bar = Bar('Saving in '+class_path+'...', max=len(imgs_path))

        for index,img_path in enumerate(imgs_path):
            bar.next()
            new_file_name=save_to_path+class_path+"/"+class_path+"_"+str(index)+".jpg"
            if os.path.exists(new_file_name):
                continue
            img_full_path=data_path+class_path+"/"+img_path
            img=cv2.imread(img_full_path,0)
            img_lung=ln.predictedLung(img,size_output=size_output)
            img_lung=AjustStuckImage(img_lung)
            if not os.path.exists(save_to_path+class_path):
                os.makedirs(save_to_path+class_path)
            cv2.imwrite(new_file_name, img_lung)
        bar.finish()

def plot(imgs):
    fig = plt.figure()
    for index,img in enumerate(imgs):
        ax = fig.add_subplot(1,len(imgs),index+1)
        ax.imshow(img)
    plt.show()

# saveOnlyLung(data_train_path,onlyLung_data_train_path,size_output=224)
# saveOnlyLung(data_test_path,onlyLung_data_test_path)
saveOnlyLung("./dataset/val/global/","./dataset/val/lung/",size_output=224)


