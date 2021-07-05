import cv2
import numpy as np
import matplotlib.pyplot as plt

def AjustImage(img_gray,removeCntr=False):
    '''
        @input (N,M,1) one channel
        normalization,equalization and remove Obfuscation from the image
        @output (N,M,1) 
    '''
    ## extract only the biggest element from image (ITC LUNG) 
    if removeCntr:
        img_gray=removeSmallestContour(img_gray)
    #Filter image by gussian filter
    img_pre=cv2.bilateralFilter(img_gray,5,20,100,borderType=cv2.BORDER_CONSTANT)
    #equalization intesety of the image
    img_pre=cv2.equalizeHist(img_pre.astype('uint8'))
    #normalization image
    return img_pre[:,:,np.newaxis]/255

def AjustStuckImage(img_gray):
    img_flr=cv2.bilateralFilter(img_gray,5,20,100,borderType=cv2.BORDER_CONSTANT)
    img_equal=cv2.equalizeHist(img_gray.astype('uint8'))
    return np.dstack((img_gray,img_flr,img_equal))


def removeSmallestContour(img_gray):
    ''' @input (N,M,1) one channel
        extract only the biggest element from image
        @output (N,M,1) 
    '''
    #binarisation then Morphological Transformation 
    img_mask=MorphologicalTransf(binarisation(img_gray))

    #extraction all objcs
    cnts, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # dim=cnts[0].shape[0]
    # out=False
    # for cnt in cnts:
    #     print(cnt.shape)
    #         out=True
    #         break
    # if not out:
        # return img_gray
    # return 2 biggest contours c.a.d lung
    for nbr in range(2):
        if len(cnts)==0:
            return img_gray
        cnt = max(cnts, key=cv2.contourArea)
        for i,cn in enumerate(cnts): 
            if cn.shape==cnt.shape:
                cnts.pop(i)

    for cnt in cnts:
    # create mask from contour finded
        mask = np.ones(img_gray.shape, dtype=img_gray.dtype)
        cv2.drawContours(mask, [cnt], -1, (0), -1)
        img_gray = cv2.multiply(img_gray,mask)
    # intersction between mask and origin image
    return img_gray

def binarisation(img):
    Vmin=np.min(img)
    Vmax=np.max(img)
    # threshold=Vmin+.9*(Vmax-Vmin)
    ret,th = cv2.threshold(img,Vmin+1,Vmax,cv2.THRESH_BINARY)  

    return th

def MorphologicalTransf(img):
    kernel=np.ones((5,5),np.uint8)
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    img=cv2.dilate(img,kernel,iterations=1)
    return img.astype('uint8')

def plot(imgs):
    fig = plt.figure()
    for index,img in enumerate(imgs):
        ax = fig.add_subplot(1,len(imgs),index+1)
        ax.imshow(img)
    plt.show()

