import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import glob
import multiprocessing
import jellyfish
import easyocr
from tqdm import tqdm
plt.switch_backend('TKAgg')
plt.bbox_inches = "tight"
#To-do increase size of killfeedbox read area


def killFeedFinder(pathToImages, queuedImage):
    matchCount = 0
    loopNumber = 0
    imageCount = 0
    all_Images = []
    imageNumber = []
    foundData = []
    reader = easyocr.Reader(['en'])
    #Load all images into an array
    for file in glob.glob(pathToImages + '/*.png'):
        all_Images.append(cv2.imread(file))
        imageCount = imageCount + 1
    print('Number of killFeed images to match ' + str(imageCount))
    for file in tqdm(glob.glob(pathToImages + '/*.png')):
        image = all_Images[loopNumber]
        #Define targets and exclusions for the OCR to find or ignore 
        targetString = 'happihound'
        otherTargets = ['ttv','twitch','_tv','TV','ttv']
        excludedTargets = ['reviving','entered']
        queuedImage.put(image)
        #perform OCR
        result = reader.readtext(image, paragraph=False)
        hasPrimaryTarget = False
        hasSecondaryTarget = True
        #Read the tuple from the OCR output
        for (bbox, text, prob) in result:
            killFeedText = text
            if(similar(killFeedText, targetString) > 0.8) and hasPrimaryTarget == False:
                hasPrimaryTarget = True
            for targets in otherTargets:
                if((similar(killFeedText, targets) > 0.7) or targets in killFeedText) and hasSecondaryTarget == False:
                    hasSecondaryTarget = True
                    break
            for excluded in excludedTargets:
                if(excluded in killFeedText):
                    hasPrimaryTarget = False
                    break
        if hasPrimaryTarget == True and hasSecondaryTarget == True:
            cv2.imwrite('other/otherOutput/' + str(loopNumber)+'.jpeg',image)
            for (bbox, text, prob) in result:
                killFeedText = text
                print(text)
                matchCount = matchCount + 1
                foundData.append(text)
                imageNumber.append(loopNumber)
                queuedImage.put(image)
        loopNumber = loopNumber + 1
    print(foundData)
    save(foundData, imageNumber) 
    print('DONE MATCHING')


def save(foundData, imageNumber):
    np.save('other/otheroutput/KillFeedData.npy', np.vstack((foundData, imageNumber)))
    pass

def similar(inputStringOne, inputStringTwo):
    #Check similarity between targets and results 
    otherTargets = ['happi','happl','bappi','hound','appi','appl']
    similarity = jellyfish.jaro_winkler_similarity(inputStringOne, inputStringTwo)
    if len(inputStringOne) == len(inputStringTwo) or len(inputStringOne) == len(inputStringTwo)-1 or len(inputStringOne) == len(inputStringTwo)+1 or len(inputStringOne) == len(inputStringTwo)-2 or len(inputStringOne) == len(inputStringTwo)+2:
        similarity = similarity + 0.15
    for targets in otherTargets:
        if((jellyfish.jaro_winkler_similarity(inputStringOne, targets) > 0.7)):
            similarity = similarity + 0.2
            break
    return  similarity

def display(queuedImage):
    cv2.namedWindow('killFeed')
    cv2.imshow("killFeed", cv2.imread('inputData/default.png'))
    while True:
        if not queuedImage.empty():
            cv2.imshow("killFeed", queuedImage.get())
            continue
        cv2.waitKey(1)


if __name__ == '__main__':
    print('Starting Kill Feed Elimination Finder')
    queuedImage = multiprocessing.Queue()
    pathToImages = 'other/bulkimages/'
    process1 = multiprocessing.Process(
        target=killFeedFinder, args=(pathToImages, queuedImage,))
    process2 = multiprocessing.Process(target=display, args=(queuedImage,))
    process1.start()
    process2.start()
