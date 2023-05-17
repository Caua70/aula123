import cv2

video = cv2.VideoCapture(0)

mog = cv2.createBackgroundSubtractorMOG2(history = 200, varThreshold = 10, detectShadows = False)
knn = cv2.createBackgroundSubtractorKNN(history = 300, dist2Threshold = 200, detectShadows = True)

while True:
    ret, frame = video.read()
    mask1 = mog.apply(frame)
    mask2 = knn.apply(frame)

    cv2.imshow("mog2", mask1)
    cv2.imshow("knn", mask2)

    if cv2.waitKey(2) == 32:
        break

    
cv2.destroyallwindows()
video.release()