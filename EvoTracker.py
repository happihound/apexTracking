import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import glob
import multiprocessing
import easyocr
from tqdm import tqdm
from statsmodels.nonparametric.smoothers_lowess import lowess


plt.switch_backend('TKAgg')
plt.bbox_inches="tight"
def evoTracker(pathToImages,queuedImage):
    matchCount = 0
    loopNumber = 0
    imageCount = 0
    foundData= []
    imageNumber= []
    all_Images = []
    foundData.append(0)
    imageNumber.append(0)
    #Load all images into an array
    for file in glob.glob(pathToImages+ '/*.png'):
        all_Images.append(cv2.imread(file))
        imageCount =imageCount+1
    print('Number of evo images to match ' +str(imageCount))
    reader= easyocr.Reader(['en'])
    for file in tqdm(glob.glob(pathToImages+ '/*.png')):
        loopNumber = loopNumber + 1
        image =  all_Images[loopNumber-1]
        #Define permitted characters
        result = reader.readtext(image, allowlist ='0123456789',paragraph=False)
        for (bbox, text, prob) in result:            
            evoFound = int(text)
            #Needs at 98% or great probability of being accurate 
            if(prob>0.98):
                #Max value of an EVO 
                if(evoFound>751):
                    print('bad match, over 750')
                    continue
                print('PICTURE NUMBER ' + str(loopNumber))
                print('Match')
                print(str(evoFound) +' ' + str(result[0][2]))
                matchCount = matchCount + 1
                foundData.append(evoFound)
                imageNumber.append(loopNumber)
                queuedImage.put(image)
    imageNumber.append((imageNumber[-1]+1))
    foundData.append(0)
    foundData.append(0)
    foundData = filterValues(foundData)
    imageNumber.append(imageCount)
    print('found ' +str(matchCount) +' total matches')
    save(foundData,imageNumber)
    graph(foundData,imageNumber)


def save(matchedImages,matchingImages):
    np.save('outputdata/evoData.npy', np.vstack((matchedImages,matchingImages)))
    
def filterValues(values):
    loopNumber = 2
    newValues = values.copy()
    for i in values:
        loopNumber = loopNumber + 1
        if loopNumber > len(values)-2:
            break
                
        if values[loopNumber-1]==values[loopNumber+1]:
            newValues[loopNumber]=values[loopNumber-1]
    return newValues


def graph(foundData,imageNumber):
    total = imageNumber[-1]
    x,y=(smoothing(foundData,imageNumber))
    plt.plot(x,y)
    plt.xlabel("Time")
    plt.ylabel("Damage")
    plt.xlim([0, total])
    plt.show()
    plt.pause(99999)
    
def smoothing(y,x):
    lowess_frac = 0.0001  # size of data (%) for estimation =~ smoothing window
    lowess_it = 0
    x_smooth = x
    y_smooth = lowess(y, x, is_sorted=False, frac=lowess_frac, it=lowess_it, return_sorted=False)
    return x_smooth, y_smooth



def display(queuedImage):
    cv2.namedWindow('damageTracker')
    cv2.imshow("damageTracker",cv2.imread('inputData/default.png'))
    while True:
        if not queuedImage.empty():
            cv2.imshow("damageTracker",queuedImage.get())
            continue
        cv2.waitKey(1) 
        
if __name__ == '__main__':
    print('Starting evo tracker')
    queuedImage = multiprocessing.Queue()
    pathToImages = 'inputData/playerEvo/'
    process1 = multiprocessing.Process(target=evoTracker, args=(pathToImages,queuedImage,))
    process2 = multiprocessing.Process(target=display, args=(queuedImage,))
    process1.start()
    process2.start()
    
