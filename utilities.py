# Contains all the methods used in the program in their generic form

import cv2
import numpy as np

"""
Contour method: We input colored image to our function and the function will apply
processes required to find the contour and gives output where all the objects are
in a  nicely formatted way.
"""

def getContours(img, cannyThreshold=[100,100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting image to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # Applying blur to grayscale image
    # In above, 5,5 is size of kernel and 1 is the value of sigma
    imgCanny = cv2.Canny(imgBlur, cannyThreshold[0], cannyThreshold[1])

    # To be on a safe side, now we'll apply dilation and erosion
    kernel = np.ones((5, 5))
    imgDialate = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThreshold = cv2.erode(imgDialate, kernel, iterations=2)
    if showCanny:
        cv2.imshow('Canny', imgThreshold)

    contours, heirarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours=[]
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True ) #To find corner points
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key = lambda x:x[1], reverse=True) #Sorting on basis of area

    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0,0,255), 3)

    return img, finalContours

'''
Using the obtained 4 corner points to work on our image
'''

def warpImg(img, points, w, h, pad=20):
    # print(points)
    points = reorder(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrx = cv2.getPerspectiveTransform(pts1, pts2)

    imgWarp = cv2.warpPerspective(img, matrx, (w, h))
    imgWarp = imgWarp[pad: imgWarp.shape[0]-pad, pad: imgWarp.shape[1]-pad] #To remove extra area that's not part of paper
    return imgWarp



# Reordering the points
def reorder(myPoints):
    print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints= myPoints.reshape((4,2))
    add=myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)] #Finding point 1
    myPointsNew[3] = myPoints[np.argmax(add)] #Finding point 4
    diff = np.diff(myPoints, axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


