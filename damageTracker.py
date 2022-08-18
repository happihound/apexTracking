import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import glob
import multiprocessing
import easyocr
from tqdm import tqdm
from statsmodels.nonparametric.smoothers_lowess import lowess
#VERY SIMILAR TO EVO TRACKER,
#SEE EVO TRACKER FOR COMMENTS

plt.switch_backend('TKAgg')
plt.bbox_inches="tight"
def damageTracker(pathToImages,queuedImage):
    matchCount = 0
    loopNumber = 0
    imageCount = 0
    foundData= []
    imageNumber= []
    all_Images = []
    foundData.append(0)
    imageNumber.append(0)
    for file in glob.glob(pathToImages+ '/*.png'):
        all_Images.append(cv2.imread(file))
        imageCount =imageCount+1
    print('Number of damage images to match ' +str(imageCount))
    reader= easyocr.Reader(['en'])
    for file in tqdm(glob.glob(pathToImages+ '/*.png')):
        loopNumber = loopNumber + 1
        image =  all_Images[loopNumber-1]
        result = reader.readtext(image, allowlist ='0123456789',paragraph=False)
        for (bbox, text, prob) in result:            
            damageFound = int(text)
            if(prob>0.93):
                if((foundData[-1] > damageFound) or ((foundData[-1]+400) < damageFound)):
                    print('bad match, found '+str(damageFound) +' but expected closer to ' +str(foundData[-1]))
                    continue
                print('PICTURE NUMBER ' + str(loopNumber))
                print('Match')
                print(result[0][1])
                matchCount = matchCount + 1
                foundData.append(damageFound)
                imageNumber.append(loopNumber)
                queuedImage.put(image)
    print('found ' +str(matchCount) +' total matches')
    save(foundData,imageNumber)
    
    graph(foundData,imageNumber)


def save(foundData,imageNumber):
    np.save('outputdata/DamageData.npy', np.vstack((foundData,imageNumber)))


def graph(foundData,imageNumber):
    x,y=(smoothing(foundData,imageNumber))
    plt.plot(x,y)
    plt.xlabel("Time")
    plt.ylabel("Damage")
    plt.show()
    plt.pause(100)
    
def smoothing(y,x):
    lowess_frac = 0.06  # size of data (%) for estimation =~ smoothing window
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
    print('Starting damage tracker')
    queuedImage = multiprocessing.Queue()
    pathToImages = 'inputData/playerDamage/'
    process1 = multiprocessing.Process(target=damageTracker, args=(pathToImages,queuedImage,))
    process2 = multiprocessing.Process(target=display, args=(queuedImage,))
    process1.start()
    process2.start()
    
