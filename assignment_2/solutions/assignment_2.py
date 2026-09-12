import numpy as np

import image_operations as img_op
import cv2

def perform_padding(img):
    res = img_op.padding(img, border_width=100)
    out_path = "padded_iris.png"
    #cv2.imwrite(out_path, res)
    print(f"Original image shape: {img.shape}.")
    print(f"Padded image shape: {res.shape}.")

def perform_crop(img):
    h, w = img.shape[:2]
    x_0 = 200
    y_0 = 200
    x_1 = w - 130
    y_1 = h - 130
    res = img_op.crop(img, x_0, x_1, y_0, y_1)
    out_path = "cropped_iris.png"
    cv2.imwrite(out_path, res)

def perform_resize(img):
    res = img_op.resize(img, 200, 200)
    out_path = "resized_iris.png"
    cv2.imwrite(out_path, res)

def perform_copy(img):
    h, w, c = img.shape
    emptyPictureArray = np.zeros((h, w, c), dtype=np.uint8)
    manual_copy = img_op.copy(img, emptyPictureArray)
    out_path = "copied_iris.png"
    cv2.imwrite(out_path, manual_copy)

def perform_grayscale(img):
    gray = img_op.grayscale(img)
    out_path = "grayscale_iris.png"
    cv2.imwrite(out_path, gray)

def perform_hsv(img):
    hsv_img = img_op.hsv(img)
    out_path = "hsv_iris.png"
    cv2.imwrite(out_path, hsv_img)

def perform_hueshift(img):
    h, w, c = img.shape
    emptyPictureArray = np.zeros((h, w, c), dtype=np.uint8)
    shifted_img = img_op.hue_shifted(img, emptyPictureArray, hue=50)
    cv2.imwrite("shifted_iris.png", shifted_img)

def perform_smoothing(img):
    smoothed_img = img_op.smoothing(img, ksize=(15, 15,))
    out_path = "smoothed_iris.png"
    cv2.imwrite(out_path, smoothed_img)

def perform_rotation(img):
    rotated_img = img_op.rotation(img, 180)
    out_path = "rotated_iris.png"
    cv2.imwrite(out_path, rotated_img)

if __name__ == "__main__":
    img = cv2.imread("iris.png")
    if img is None:
        raise FileNotFoundError(f"Could not load image file.")

    #perform_padding(img)
    #perform_crop(img)
    #perform_resize(img)
    #perform_copy(img)
    #perform_grayscale(img)
    #perform_hsv(img)
    #perform_hueshift(img)
    #perform_smoothing(img)
    #perform_rotation(img)
