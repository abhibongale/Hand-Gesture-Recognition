"""
builder.py helps to create custom dataset.
"""
import cv2, time, pyautogui, os
import numpy as np

path = '../data/'
if not os.path.exists(path):
    os.makedirs(path)

test_path = '../data/test'

# create camera object
capture = cv2.VideoCapture(0)

# decrease frame size
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

# total outputs
outputs = 10
while 1:
    if cv2.waitKey(2) == 99:
        for i in range(1, 501):
            ret, frame = capture.read()
            cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
            crop_frame = frame[100:300, 100:300]
            gauss_blur = cv2.GaussianBlur(crop_frame, (3, 3), 0)
            hsv = cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2HSV)

            mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))
            kernel_sq = np.ones((11, 11), np.uint8)
            kernel_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            median_blur = cv2.medianBlur(mask2, 5)
            cv2.imshow('main', frame)
            cv2.imshow('masked', median_blur)
            resize = cv2.resize(median_blur, (50, 50))
            cv2.imwrite(os.path.join(path, "gesture" + str(outputs)
                                     + "_" + str(i)+".jpg").resize)
            cv2.imshow('resize', resize)
            time.sleep(0.05)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        else:
            ret, frame = capture.read()
            cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
            crop_frame = frame[100:300, 100:300]
            gauss_blur = cv2.GaussianBlur(crop_frame, (3, 3), 0)
            hsv = cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2HSV)
            # create a binary image with where white will be skin colours and rest is black
            mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

            kernel_sq = np.ones((11, 11), np.uint8)
            kernel_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            median_blur = cv2.medianBlur(mask2, 5)
            cv2.imshow('main', frame)
            cv2.imshow('masked', median_blur)
            # close the output video by processing  'ESC'
            k = cv2.waitKey(2) & 0xFF
            if k == 27:
                break
capture.release()
cv2.destroyAllWindows()
