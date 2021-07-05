from keras.preprocessing.image import ImageDataGenerator
from vendor.preprocessing.image import AjustImage

def adjustMask(mask):
    #separation intensity of the mask
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask

def DataGenerator(batch_size,path,image_folder,mask_folder,aug_dict,color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                 target_size = (256,256),seed = 1):

    #generatation data useing data augmentation and AjustImage as preprocessing image 
    image_datagen = ImageDataGenerator(**aug_dict,preprocessing_function=AjustImage)
    mask_datagen = ImageDataGenerator(**aug_dict)
    #batch_size to avoid high memory usage
    #class_mode use mask as a label instead of classes folder
    image_generator = image_datagen.flow_from_directory(
        path,
        classes = [image_folder],
        class_mode = None,
        color_mode = color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_prefix  = mask_save_prefix,
        seed = seed)
    #before generation ajust mask 
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        mask = adjustMask(mask)
        yield (img,mask)
