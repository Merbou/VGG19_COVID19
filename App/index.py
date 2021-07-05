from tkinter.filedialog import askopenfilename
# from vendor.preprocessing.detector import DataGenerator
# from folderManager import saveOnlyLungFile
from model.detector import Classfier
import matplotlib.pyplot as plt 
from tkinter import Tk
size=224

# train_dataset,test_dataset=DataGenerator((size,size))
clsf=Classfier(size=size)
# clsf.train(train_dataset,test_dataset,_epochs=12)

Tk().withdraw()
filename = askopenfilename()
print(clsf.predict(filename))
