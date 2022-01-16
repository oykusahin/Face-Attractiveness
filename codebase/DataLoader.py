import os
from PIL import Image
import numpy as np
import random
#A method to load data from .jpg to pixels
def data_loader(file_dir, num_data):
    images = []
    labels = []
    count  = 0
    
    all_files = os.listdir(file_dir)

    r = list(range(num_data))
    random.shuffle(r)
    for i in r:
        print("Getting the image #" + str(i))
        im = Image.open(file_dir+all_files[i],"r").convert('LA')#getting the image
        label = all_files[i].split(".jpg")[0].split("_")[0]     #getting the label from name
        temp_data = np.asarray(im.getdata())                    #converting the image -> pixels
        pix_val = np.resize(temp_data,(80, 80, 3))              #the data already in expected size but just in case
        pix_val = np.true_divide(pix_val, 255)                  #normalization of pixels
        images = np.append(images, pix_val)                     #putting all pixels to one array, will be reshaped later
        labels = np.append(labels, label)                       #putting all labels to one array, will be reshaped later 
        count = count + 1                                       #counting the number of samples


    images = np.reshape(images, (count, 80,80,3))               #pixels reshaped
    images = images.astype('float32')
    labels = np.reshape(labels, (count, 1))                     #labels reshaped
    labels = labels.astype('float32')

    print("shape for images: " + str(images.shape))             #just to make sure everything is as we wanted
    print("shape for labels: " + str(labels.shape))             

    return images, labels




