import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import glob
import multiprocessing
import easyocr
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
#Similar to evo tracker

plt.switch_backend('TKAgg')
plt.bbox_inches="tight"
def killTracker(pathToImages,queuedImage):
    matchCount = 0
    loopNumber = 0
    imageCount = 0
    foundData= []
    imageNumber= []
    all_Images = []
    foundData.append(0)
    imageNumber.append(0)
    rollingArray = [0,0,0,1,1,1,1,1,1,1]
    for file in glob.glob(pathToImages+ '/*.png'):
        all_Images.append(cv2.imread(file))
        imageCount =imageCount+1
    print('Number of kill images to match ' +str(imageCount))
    reader= easyocr.Reader(['en'])
    for file in tqdm(glob.glob(pathToImages+ '/*.png')):
        loopNumber = loopNumber + 1
        image =  all_Images[loopNumber-1]
        image = imageBinarizer(image)
        result = reader.readtext(image, allowlist ='0123456789',paragraph=False)
        queuedImage.put(image)
        print('PICTURE NUMBER ' + str(loopNumber))
        if len(result):
            sequenceOfTextFound = 0
            for (bbox, text, prob) in result:
                if text == '':
                    continue
                text = int(text)
                sequenceOfTextFound = sequenceOfTextFound +1
                if sequenceOfTextFound > 1 or prob < 0.9:
                    continue
                checkValue = int((np.sum(rollingArray[-3:-1])/3))
                print('checkValue: ' +str(checkValue))
                rollingArray.append(text)
                if((checkValue > text) or ((checkValue+5) < text)):
                    print('bad match, found '+str(text) +' but expected closer to ' +str(checkValue))
                    continue
                print(str(text) +' ' + str(prob))
                matchCount = matchCount + 1
                foundData.append(text)
                imageNumber.append(loopNumber)
                queuedImage.put(image)
    foundData = filterValues(foundData)
    print('found ' +str(matchCount) +' total matches')
    save(foundData,imageNumber)
    
    graph(foundData,imageNumber)




def imageBinarizer(image):
    width = int(image.shape[1] * 5)
    height = int(image.shape[0] * 5)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # Color segmentation
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    lower_red = np.array([0, 0, 220])
    upper_red = np.array([200, 45, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(resized, resized, mask=mask)
    return res

def save(matchedImages,matchingImages):
    np.save('outputdata/killData.npy', np.vstack((matchedImages,matchingImages)))
    
def filterValues(values):
    loopNumber = 8
    newValues = values.copy()
    for i in values:
        loopNumber = loopNumber + 1
        if loopNumber > len(values)-4:
            break
        
        newValues[loopNumber] = most_frequent(values[loopNumber-8:loopNumber+4])     
    return newValues

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

def graph(foundData,imageNumber):
    total = imageNumber[-1]
    x,y=(smoothing(foundData,imageNumber))
    plt.locator_params(axis='x', nbins=120)
    plt.plot(x,y)
    plt.xlabel("Time")
    plt.ylabel("kills")
    plt.xlim([0, total])
    plt.show()
    plt.pause(99999)
    
def smoothing(y,x):
    lowess_frac = 0.00001  # size of data (%) for estimation =~ smoothing window
    lowess_it = 0
    x_smooth = x
    y_smooth = lowess(y, x, is_sorted=False, frac=lowess_frac, it=lowess_it, return_sorted=False)
    return x_smooth, y_smooth



def display(queuedImage):
    cv2.namedWindow('killTracker')
    cv2.imshow("killTracker",cv2.imread('inputData/default.png'))
    while True:
        if not queuedImage.empty():
            cv2.imshow("killTracker",queuedImage.get())
            continue
        cv2.waitKey(1) 
        
if __name__ == '__main__':
    print('Starting kill tracker')
    queuedImage = multiprocessing.Queue()
    pathToImages = 'inputData/playerKills/'
    process1 = multiprocessing.Process(target=killTracker, args=(pathToImages,queuedImage,))
    process2 = multiprocessing.Process(target=display, args=(queuedImage,))
    process1.start()
    process2.start()
    
