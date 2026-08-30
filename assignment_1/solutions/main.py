import os
import cv2

def print_image_information(image):
    size = image.size
    data_type = image.dtype

    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    print(f"Height = {height}, Width = {width}, Channels = {channels}.")
    print(f"Size = {size}")
    print(f"Data Type = {data_type}")

def save_webcam_information(path):
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Webcam not found")
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture.release()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

    print(f"Camera outputs have been saved to {path} successfully.")

def main():
    img_name = "iris-1.jpg"
    img = cv2.imread(img_name)
    if img is not None:
        print_image_information(img)
    else:
        print("Img not found")

    output_path = os.path.expanduser("~/Documents/IKT213_strand/assignment_1/solutions/camera_outputs.txt")
    save_webcam_information(output_path)
if __name__ == "__main__":
    main()