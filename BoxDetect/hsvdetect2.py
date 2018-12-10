import math
import sys
from time import sleep

import cv2
import numpy as np
import os
import time
import imutils

def contour2Points(contour):
    for point in contour:
        point = point.tolist()
        #print(point)
        #print(point[0][0])
    #print("\n")

def getDistance(pixlwidth, actLength):
    focal = 559.542
    #actLength = 35
    #print("Length: " + str(pixlwidth))
    #print("Focal: " + str(focal))
    #print("ActualLength: " + str(actLength))
    t1 = (pixlwidth * focal)
    #print("Length * Focal = " + str(t1))
    t2 = t1 / actLength
    #print("... / actualLength => " + str(t2))

    return (int) ((actLength * focal)/pixlwidth)




def getBoxDist(outerP, outerY):
    # Yellow-Side: 20/12.5 = 1.6
    # pink-Side: 34.5/12.5 = 2.76
    if(outerY is not None):
        sideLengthY, sideWdithY = getContourLength(outerY)
        yRatio = sideLengthY / sideWdithY
        print("yRatio => " + str(yRatio))
    if (outerP is not None):
        sideLengthP, sideWdithP = getContourLength(outerP)
        pRatio = sideLengthP / sideWdithP
        print("pRatio => " + str(pRatio))

    if (outerY is not None and (yRatio >= 1.5 and yRatio <= 1.8)):
        actualLength = 20
        distance = getDistance(sideLengthY, actualLength)
        # print("Length: " + str(sideLength))
        print("Distance: " + str(distance))
        return distance
    elif (outerP is not None and (pRatio >= 2.5 and pRatio <= 3)):
        actualLength = 35
        distance = getDistance(sideLengthP, actualLength)
        print("Distance: " + str(distance))
        return distance
    return None


def getContourLength(outerPoints):
    [uL, uR, lL, lR] = outerPoints

    length1 = getDist(uL, uR)
    length2 = getDist(lL, lR)

    height1 = getDist(lL, uL)
    height2 = getDist(lR, uR)

    return length1, height1



def getOuterPoints(contour, center):
    #print("Center:\t" + str(center))
    points = cv2.convexHull(contour, clockwise=True)
    #print("Points: " + str(points))

    uLeft = []
    uRight = []
    lLeft = []
    lRight = []

    for point in points:
        tPoint = point.tolist()[0]
        #print("orientation tPoint: " + str(tPoint))
        if(tPoint[0] < center[0]):
            if (tPoint[1] < center[1]):
                uLeft.append(tPoint)
            else:
                lLeft.append(tPoint)
        else:
            if (tPoint[1] < center[1]):
                uRight.append(tPoint)
            else:

                lRight.append(tPoint)

    #print("Upper Left")
    #print(str(uLeft))

    #print("Lower Left")
    #print(str(lLeft))

    #print("Upper Right")
    #print(str(uRight))

    #print("Lower Right")
    #print(str(lRight))

    if(uLeft.__len__() > 0):
        upperLeft = uLeft[0]

    if (lLeft.__len__() > 0):
        lowerLeft = lLeft[0]

    if (uRight.__len__() > 0):
        upperRight = uRight[0]

    if (lRight.__len__() > 0):
        lowerRight = lRight[0]

    i = 1
    while(i < uLeft.__len__()):
        tPoint = uLeft[i]
        #print('upperLeft: ' + str(tPoint))
        if(getDist(tPoint, center) > getDist(upperLeft, center)):
            upperLeft = tPoint
        i += 1

    i = 1
    while (i < lLeft.__len__()):
        tPoint = lLeft[i]
        #print('lowerLeft: ' + str(tPoint))
        if (getDist(tPoint, center) > getDist(lowerLeft, center)):
            lowerLeft = tPoint
        i += 1

    i = 1
    while (i < uRight.__len__()):
        tPoint = uRight[i]
        #print('upperRight: ' + str(tPoint))
        if (getDist(tPoint, center) > getDist(upperRight, center)):
            upperRight = tPoint
        i += 1

    i = 1
    while (i < lRight.__len__()):
        tPoint = lRight[i]
        #print('lowerRight: ' + str(tPoint))
        if (getDist(tPoint, center) > getDist(lowerRight, center)):
            lowerRight = tPoint
        i += 1

    return [upperLeft, upperRight, lowerLeft, lowerRight]

def getDist(point1, point2):
    #print('Point1: ' + str(point1))
    #print('point2: ' + str(point2))

    return(math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2))

def getContourCenter(contour):
    if not contour.any():
        return None
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    #print("cX = " + str(cX) + " | cY = " + str(cY))
    return (cX, cY)

def colorFiltering(initialized):
    # named ites for easy reference
    barsWindow = 'Bars'
    hl = 'H Low'
    hh = 'H High'
    sl = 'S Low'
    sh = 'S High'
    vl = 'V Low'
    vh = 'V High'

    # uncomment to see coloring window
    #cv2.namedWindow(barsWindow, flags=cv2.WINDOW_AUTOSIZE)

    # create the sliders

    if(not initialized):
        cv2.createTrackbar(hl, barsWindow, 0, 179, nothing)
        cv2.createTrackbar(hh, barsWindow, 0, 179, nothing)
        cv2.createTrackbar(sl, barsWindow, 0, 255, nothing)
        cv2.createTrackbar(sh, barsWindow, 0, 255, nothing)
        cv2.createTrackbar(vl, barsWindow, 0, 255, nothing)
        cv2.createTrackbar(vh, barsWindow, 0, 255, nothing)

        # set initial values for sliders
        cv2.setTrackbarPos(hl, barsWindow, 14)
        cv2.setTrackbarPos(hh, barsWindow, 53)
        cv2.setTrackbarPos(sl, barsWindow, 60)
        cv2.setTrackbarPos(sh, barsWindow, 123)
        cv2.setTrackbarPos(vl, barsWindow, 150)
        cv2.setTrackbarPos(vh, barsWindow, 200)
        initialized = True

    # Yellow
    # read trackbar positions for all
    hul = cv2.getTrackbarPos(hl, barsWindow)
    huh = cv2.getTrackbarPos(hh, barsWindow)
    sal = cv2.getTrackbarPos(sl, barsWindow)
    sah = cv2.getTrackbarPos(sh, barsWindow)
    val = cv2.getTrackbarPos(vl, barsWindow)
    vah = cv2.getTrackbarPos(vh, barsWindow)

    return hul, huh, sal, sah, val, vah, initialized

def getSpecificColMask(lower, upper, frame, hsv, width, height):
    colorLow = np.array(lower)
    colorHigh = np.array(upper)
    colorMask = cv2.inRange(hsv, colorLow, colorHigh)

    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    color = cv2.erode(colorMask, kernel, iterations=4)
    color = cv2.dilate(colorMask, kernel, iterations=1)
    colorFrame = cv2.bitwise_and(frame, frame, mask=colorMask)
    return color

def printContours(printed, contoursY, contoursP, height, width):
    if(not printed):
        print("Contoursobject is of type:" + str(type(contoursY)))
        print("ContoursYellow is of type:" + str(type(contoursY)))
        print("ContoursPink is of type:" + str(type(contoursP)))
        print("FRAME WIDTH: " + str(width) + "\tFRAME HEIGHT: " + str(height))

        i = 0
        while(i < len(contoursP)):
            area = cv2.contourArea(contoursP[i])
            print("\nContour number: " + str(i) + " has: " + str(len(contoursP[i])) + " coordinates")
            print("Area: " + str(area))
            i += 1

        i = 0
        while (i < len(contoursY)):
            M = cv2.moments(contoursY[i])
            epsilon = 0.01 * cv2.arcLength(contoursY[i], True)
            approx = cv2.approxPolyDP(contoursY[i], epsilon, True)
            area = cv2.contourArea(contoursY[i])

            contPoints = np.vstack(contoursY[i].squeeze())
            #print("\nContour Info: " + str(contPoints))




            print("Area: " + str(area))
            i += 1

        printed = True

    #del contoursP[0:4]
    #cv2.drawContours(frame, contoursP, -1, (0, 0, 255), 6)
    #cv2.drawContours(frame, contoursY, -1, (255, 0, 0), 6)


    #drawConts(frame, contoursP, (0, 0, 255))
    #drawConts(frame, contoursY, (255, 0, 0))



def removeFrameCont(contours, frameWidth, frameHeight):
    i = 0
    while (i < len(contours)):
        area = cv2.contourArea(contours[i])
        #print('Ext: ' + str(area))
        if ((area <= (frameWidth * frameHeight) and area >= 30000) or area <= 150):
            #print(area)
            #print("\n" + str(area))
            #print('EXCEPTION FOUND')
            npCont = np.array(contours)
            #contours.remove(contours[i])
            del contours[i]
            i = i - 1
        i += 1
    return contours


def drawConts(windowframe, contours, color):
    cv2.drawContours(windowframe, contours, -1, color, 6)

def getContourBoxPoints(contour):
    rc = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rc)
    return box

def printContour(cont):
    print('For Contour wth Area: ' + str(cv2.contourArea(cont)))
    for p in cont:
        print(p)

def contors2Points(conts):
    pt = []
    for cont in conts:
        for p in cont:
            pt.append(p)
    #print('internal single: ' + str(pt))
    return np.int0(pt)





def getConvex(frame, hull, points):
    hull.append(cv2.convexHull(points, hull, True))
    return hull

# optional argument for trackbars
def nothing(x):
    pass

def calcContours(frame, hsv, width, height):
    # Yellow

    # Interior night
    #yellowLow = np.array([0, 165, 136])
    #yellowHigh = np.array([76, 255, 255])

    # Interior day
    yellowLow = np.array([0, 101, 155])
    yellowHigh = np.array([33, 186, 255])

    yellow = getSpecificColMask(yellowLow, yellowHigh, frame, hsv, width, height)

    # Pink
    # Interior night
    #pinkLow = np.array([161, 70, 103])
    #pinkHigh = np.array([179, 191, 255])
    pinkLow = np.array([122, 70, 64])
    pinkHigh = np.array([179, 255, 184])
    pink = getSpecificColMask(pinkLow, pinkHigh, frame, hsv, width, height)

    ret, threshold = cv2.threshold(yellow, 100, 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    _, contoursY, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    ret, threshold = cv2.threshold(pink, 100, 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    _, contoursP, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if (len(contoursY) > 0):
        contoursY = removeFrameCont(contoursY, width, height)
    if (len(contoursP) > 0):
        contoursP = removeFrameCont(contoursP, width, height)

    return (contoursY, yellow, contoursP, pink)


cap = None
initilaized = False


def setup(picTest):
    cwd = os.getcwd()
    path = cwd + '/hsvTest/box2.jpg'
    test = os.path.isfile(path)

    printed = False

    if test and picTest:
        frame = cv2.imread(path)
        height, width, _ = frame.shape
        cap = None
    else:
        cap = cv2.VideoCapture(1)
        ret, frame = cap.read()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    return (frame, cap, height, width)





def main():
    # Picture instead
    picTest = False

    #cap set correctly in setup method
    frame, cap, height, width = setup(picTest)

    initialized = False

    start = time.time()
    #The time delta in which I want to print
    delta = 3
    contIni = False

    while(True):

        if(picTest == False):
            ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        #Do HSV stuff
        hul, huh, sal, sah, val, vah, initialized = colorFiltering(initialized)
        if initialized == False:
            initialized = True
        HSVLOW = np.array([hul, sal, val])
        HSVHIGH = np.array([huh, sah, vah])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)
        maskedFrame = cv2.bitwise_and(frame, frame, mask=mask)

        contoursY, yellow, contoursP, pink = calcContours(frame, hsv, width, height)

        compP = contors2Points(contoursP)
        compY = contors2Points(contoursY)

        now = time.time()
        if (contIni == False):
            outerP = None
            outerY = None

        if (compP.any()):
            hullP = cv2.convexHull(compP)
            cv2.drawContours(frame, [hullP], -1, (0, 0, 255), 2)

            pCX, pCY = getContourCenter(hullP)
            cv2.circle(frame, (pCX, pCY), 5, (0, 0, 255), 3)

            if (outerP):
                for point in outerP:
                    cv2.circle(frame, (point[0], point[1]), 4, (127, 0, 127), 1)
        else:
            outerP = None

        if (compY.any()):
            hullY = cv2.convexHull(compY)
            cv2.drawContours(frame, [hullY], -1, (255, 0, 0), 2)

            yCX, yCY = getContourCenter(hullY)


            if (outerY):
                for point in outerY:
                    cv2.circle(frame, (point[0], point[1]), 4, (0, 255, 0), 1)
        else:
            outerY = None

        if (now - start >= delta):
            if compY.any():
                outerY = getOuterPoints(hullY, [yCX, yCY])
            elif compP.any():
                outerP = getOuterPoints(hullP, [pCX, pCY])
            distance = getBoxDist(outerP, outerY)
            print("Distance from Kamera to Box = " + str(distance))
            start = time.time()

            contIni = True




        composite = cv2.addWeighted(pink, 1.1, yellow, 0.6, 0)


        if (display):
            cv2.imshow('Camera', frame)
            cv2.imshow('comp', composite)
        # cv2.imshow('result', maskedFrame)

        # check for q to quit program with 5ms delay
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        #sleep(0.25)
    cap.release()
    cv2.destroyAllWindows()


display = True
main()


