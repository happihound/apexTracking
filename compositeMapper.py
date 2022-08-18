from PIL import Image                                                            
import numpy as np                                                                    
import glob
import matplotlib.pyplot as plt
import cv2
from os import listdir
from os.path import isfile, join
tileFolderPath = 'tiles/'
tilePath = glob.glob(tileFolderPath + '/*.png') 
mapFolderPath = 'map/'
#template = cv2.imread('map/outimage_150.jpeg')
all_images = []
for file in glob.glob(tileFolderPath+ '/*.png'):
    all_images.append(cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB))
    print(file)
h = 289
Verti = 0
for image in all_images:
    if h > 0:
        h = h - 17
        Hori = np.concatenate((all_images[h],all_images[h+1], all_images[h+2], all_images[h+3], all_images[h+4], all_images[h+5], all_images[h+6], all_images[h+7], all_images[h+8], all_images[h+9], all_images[h+10], all_images[h+11], all_images[h+12], all_images[h+13], all_images[h+14], all_images[h+15],all_images[h+16]), axis=1)
        if h==289-17:
            Verti=Hori.copy()
            continue
    Verti = np.concatenate((Hori, Verti), axis=0)
    if h <= 0:
        break
#plt.figure(figsize=(32,32)) # specifying the overall grid size
#for i in range(256):
  #  plt.axis('off')
  #  plt.adjustable='datalim'
  #  plt.subplot(16,16,i+1,)    # the number of images in the grid is 5*5 (25)
 #   plt.subplots_adjust(wspace=0, hspace=0)
#    plt.imshow(all_images[i])

#plt.axis('off')
#plt.subplots_adjust(left=0.1)
plt.bbox_inches="tight"
#plt.subplots_adjust(bottom=0.10)
plt.imsave('mapOLY.png',Verti)
