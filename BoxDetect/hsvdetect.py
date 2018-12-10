import cv2
import numpy as np
import os


# optional argument for trackbars
def nothing(x):
    pass


# named ites for easy reference
barsWindow = 'Bars'
hl = 'H Low'
hh = 'H High'
sl = 'S Low'
sh = 'S High'
vl = 'V Low'
vh = 'V High'

# set up for video capture on camera 0
cap = cv2.VideoCapture(1)

# create window for the slidebars
cv2.namedWindow(barsWindow, flags=cv2.WINDOW_AUTOSIZE)

# create the sliders
cv2.createTrackbar(hl, barsWindow, 0, 179, nothing)
cv2.createTrackbar(hh, barsWindow, 0, 179, nothing)
cv2.createTrackbar(sl, barsWindow, 0, 255, nothing)
cv2.createTrackbar(sh, barsWindow, 0, 255, nothing)
cv2.createTrackbar(vl, barsWindow, 0, 255, nothing)
cv2.createTrackbar(vh, barsWindow, 0, 255, nothing)



# set initial values for sliders
cv2.setTrackbarPos(hl, barsWindow, 0)
cv2.setTrackbarPos(hh, barsWindow, 41)
cv2.setTrackbarPos(sl, barsWindow, 123)
cv2.setTrackbarPos(sh, barsWindow, 255)
cv2.setTrackbarPos(vl, barsWindow, 157)
cv2.setTrackbarPos(vh, barsWindow, 255)

kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((3,3),np.uint8)


#Picture instead
picTest = False
cwd = os.getcwd()
path = cwd + '/hsvTest/box2.jpg'
test = os.path.isfile(path)

printed = False

while (True):
    if(picTest and test):
        frame = cv2.imread(path)
        height, width, _ = frame.shape
    else:
        ret, frame = cap.read()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float





    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gaus = cv2.adaptiveThreshold(gray, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # convert to HSV from BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    #Yellow
    # read trackbar positions for all
    hul = cv2.getTrackbarPos(hl, barsWindow)
    huh = cv2.getTrackbarPos(hh, barsWindow)
    sal = cv2.getTrackbarPos(sl, barsWindow)
    sah = cv2.getTrackbarPos(sh, barsWindow)
    val = cv2.getTrackbarPos(vl, barsWindow)
    vah = cv2.getTrackbarPos(vh, barsWindow)

    # make array for final values
    HSVLOW = np.array([hul, sal, val])
    HSVHIGH = np.array([huh, sah, vah])



    # apply the range on a mask
    mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)




    #mask = cv2.dilate(mask, kernel2, iterations=3)
    #mask = cv2.dilate(mask, kernel2, iterations=3)

    maskedFrame = cv2.bitwise_and(frame, frame, mask=mask)

    erodeDil = cv2.erode(mask, kernel, iterations=4)
    erodeDil = cv2.dilate(mask, kernel2, iterations=1)

    # Yellow
    #yellowLow = np.array([9, 47, 70])
    #yellowHigh = np.array([31, 118, 144])
    yellowLow = np.array([0, 87, 178])
    yellowHigh = np.array([30, 193, 255])
    yellowMask = cv2.inRange(hsv, yellowLow, yellowHigh)

    yellow = cv2.erode(yellowMask, kernel, iterations=4)
    yellow = cv2.dilate(yellowMask, kernel2, iterations=1)
    yellowFrame = cv2.bitwise_and(frame, frame, mask=yellowMask)


    # Pink
    pinkLow = np.array([109, 134, 60])
    pinkHigh = np.array([179, 255, 255])
    pinkMask = cv2.inRange(hsv, pinkLow, pinkHigh)

    pink = cv2.erode(pinkMask, kernel, iterations=4)
    pink = cv2.dilate(pinkMask, kernel2, iterations=1)
    pinkFrame = cv2.bitwise_and(frame, frame, mask=pinkMask)

    composite = cv2.addWeighted(pink, 1.1, yellow, 0.6, 0)

    #ret, threshold = cv2.threshold(composite, 0, 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #_, contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    ret, threshold = cv2.threshold(yellow, 0, 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    _, contoursY, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    ret, threshold = cv2.threshold(pink, 100, 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    _, contoursP, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contours = contoursY
    contours.append(contoursP)



    pixelPoints = np.transpose(np.nonzero(composite))
    plist = cv2.findNonZero(composite)

    hull = []
    for a in contours[0]:
        point = a[0]
        point = str(point)
        if(point.startswith(' ')):
            del point[0]
        point = point.split()
        #point = [int(point[0]), int(point[1])]
        #print(point[0])
    #print hull

    #cv2.drawContours(maskedFrame, contours, -1, (0,0,255), 6)


    #cv2.drawContours(frame, contours, -1, (0,0,255), 6)
    #cv2.drawContours(frame, contoursY, -1, (0, 0, 255), 6)

    # print(len(contours))

    # cv2.imshow('ErodeDil1', erodeDil1)
    # cv2.imshow('ErodeDil2', erodeDil2)
    # display the camera and masked images
    #cv2.imshow('Mask', mask)
    cv2.imshow('result', maskedFrame)





    #cv2.imshow('gaus', gaus)

    cv2.imshow('Camera', frame)
    #cv2.imshow('Composite', composite)



    # check for q to quit program with 5ms delay
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# clean up our resources
cap.release()
cv2.destroyAllWindows()
