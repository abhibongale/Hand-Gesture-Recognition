"""
gesture.py is the main file used to for live detection
"""
import tensorflow as tf
import numpy as np
import cv2

c_frame = -1
p_frame = -1

# Setting threshold for number of frames to compare
threshold_frames = 50

# loading up the model
session = tf.Session()
# recreate  the network graph,
saver = tf.train.import_meta_graph('../model/train_model.meta')
# restoring  the weights
saver.restore(session, tf.train.latest_checkpoint('../model'))
# default graph
graph = tf.get_default_graph()

# Now, let's get hold of the operation that we can be processed to get the output.

# y is the tensor that is the prediction of the network
y = graph.get_tensor_by_name("y: 0")

# feed the images to the input placeholders
X = graph.get_tensor_by_name("X: 0")
y_true = graph.get_tensor_by_name("y_true: 0")
y_test_imgs = np.zeros((1, 10))


# live detection
def live(frame, y_test_images):
    """
    live detection using webcam
    :param frame: webcam frame
    :param y_test_images: test images
    :return: predicted output
    """
    img_size = 50
    n_channels = 3
    images = []
    image = frame
    cv2.imshow('test', image)
    # preprocessing step.
    # resize images
    img = cv2.resize(image, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)
    images.append(img)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)

    # The input to the network is of shape [None, img_size, img_size, n_channels].
    X_batch = images.reshape(1, img_size, img_size, n_channels)

    # Creating the feed_dict that is required to be feed to calculate y
    feed_dict_testing = {X: X_batch, y_true: y_test_images}
    result = session.run(y, feed_dict=feed_dict_testing)
    return np.array(result)


# camera object
capture = cv2.VideoCapture(0)

# frame dim (4 * 4)
capture.set(4, 700)
capture.set(4, 700)


i = 0
while i < 100000000:
    ret, frame = capture.read()
    cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
    crop_frame = frame[100:300, 100:300]
    # blur the images
    blur = cv2.GaussianBlur(crop_frame, (3, 3), 0)
    # convert to HSV colour space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # create a binary image with where white will be skin colour and rest in black
    mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))
    median_blur = cv2.medianBlur(mask2, 5)
    # displaying frames
    cv2.imshow('main', frame)
    cv2.imshow('masked', median_blur)
    # resize the image
    memedian_blur = cv2.resize(median_blur, (50, 50))
    # making it 3 channel
    median_blur = np.stack((median_blur, )*3)
    # making it 3 channel
    median_blur = np.stack((median_blur, )*3)
    # adjusting rows, columns as per X
    median_blur = np.rollaxis(median_blur, axis=1, start=0)
    median_blur = np.rollaxis(median_blur, axis=2, start=0)
    # rotating and flipping correctly as per training images
    rotate_img = cv2.getRotationMatrix2D((25, 25), 270, 1)
    median_blur = cv2.wrapAffine(median_blur)
    median_blur = np.fliplr(median_blur)
    # exponent to float
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})

    # print index of maximum probability value
    answer = live(median_blur, y_test_imgs)

    # compare 50 continuous frames
    c_frame = np.argmax(max(answer))
    if c_frame == p_frame:
        counter = + 1
        p_frame = c_frame
        if counter == threshold_frames:
            print(answer)
            print("Answer: "+str(c_frame))
            counter = 0
            i = 0
    else:
        p_frame = c_frame
        counter = 0

    # close the output video by pressing 'ESC'
    k = cv2.waitKey(2) & 0xFF
    if k == 27:
        break
    i = + 1

capture.release()
cv2.destroyAllWindows()
