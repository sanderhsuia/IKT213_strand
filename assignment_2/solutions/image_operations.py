import cv2
import numpy as np

def padding(image, border_width=100):
    padded_img = cv2.copyMakeBorder(image, top=border_width, bottom=border_width, left=border_width, right=border_width, borderType=cv2.BORDER_REFLECT)
    return padded_img

def crop(image, x_0, x_1, y_0, y_1):
    return image[y_0:y_1, x_0:x_1]

def resize(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def copy(image, emptyPictureArray):
    emptyPictureArray[:] = image[:]
    return emptyPictureArray

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hue_shifted(image, emptyPictureArray, hue=50):
    shifted = image.astype(np.int16) + hue
    emptyPictureArray[:] = np.clip(shifted, 0, 255).astype(np.uint8)
    return emptyPictureArray

def smoothing(image, ksize=(15, 15)):
    return cv2.GaussianBlur(image, ksize, 0, borderType=cv2.BORDER_DEFAULT)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Invalid rotation angle")