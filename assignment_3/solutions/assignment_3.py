import cv2
import numpy as np

def sobel_edge_detection(image):
    blur = cv2.GaussianBlur(image, (3,3), 0)
    sobel = cv2.Sobel(blur, cv2.CV_64F,1,1,ksize=1)
    return cv2.convertScaleAbs(sobel)

def canny_edge_detection(image, threshold_1=50, threshold_2=50):
    blur = cv2.GaussianBlur(image, (3,3), 0)
    return cv2.Canny(blur, threshold_1, threshold_2)

def template_match(image, template):
    threshold = 0.9
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marked_img = image.copy()
    else:
        gray_image = image
        marked_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if len(template.shape) == 3:
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray_template = template

    h, w = gray_template.shape[:2]
    res = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    locs = np.where(res >= threshold)

    list(map(lambda point: cv2.rectangle(marked_img, point, (point[0] + w, point[1] + h), (0, 0, 255), 2), zip(*locs[::-1])))

    return marked_img

def resize(image, scale_factor: int, up_or_down: str):
    curr_img = image.copy()
    if up_or_down.lower() == "up":
        for _ in range(scale_factor): curr_img = cv2.pyrUp(curr_img)
    elif up_or_down.lower() == "down":
        for _ in range(scale_factor): curr_img = cv2.pyrDown(curr_img)
    else:
        raise ValueError("up_or_down must be either 'up' or 'down'")

    return curr_img

if __name__ == "__main__":
    lambo = cv2.imread("lambo.png")
    if lambo is not None:
        sobel_res = sobel_edge_detection(lambo)
        cv2.imwrite("sobel_edges.png", sobel_res)

        canny_res = canny_edge_detection(lambo, 50, 50)
        cv2.imwrite("canny_edges.png", canny_res)

    shapes_img = cv2.imread("shapes-1.png")
    shapes_template = cv2.imread("shapes_template.jpg")
    if shapes_img is not None and shapes_template is not None:
        matched_res = template_match(shapes_img, shapes_template)
        cv2.imwrite("matched_shapes.png", matched_res)

    pyramid_img = lambo
    if pyramid_img is not None:
        downscaled = resize(pyramid_img, scale_factor=2, up_or_down="down")
        upscaled = resize(pyramid_img, scale_factor=2, up_or_down="up")
        cv2.imwrite("pyramid_down.png", downscaled)
        cv2.imwrite("pyramid_up.png", upscaled)