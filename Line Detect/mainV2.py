import cv2
import numpy as np

def find_line(img):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    gray_img = cv2.convertScaleAbs(gray_img, alpha=1.4, beta=0)

    # Edge detection using combined gradient and color-based edge detection
    edges = cv2.Canny(gray_img, 50, 150)

    # Define region of interest
    roi_img = find_roi(edges)

    # Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(roi_img, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw lines on the original image
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    
    combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    return combo_image

def find_roi(img):
    mask = np.zeros_like(img)
    height, width = img.shape
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.45), int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi_img = cv2.bitwise_and(img, mask)
    return roi_img

def make_vid():
    vid = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\AISC2024 IOTMLLAB\Line Detect\lane.mp4")
    success = True
    count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.avi', fourcc, 20, (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while success:
        success, frame = vid.read()
        if success:
            count += 1
            print(f'Frame {count}')
            result = find_line(frame)
            video.write(result)
    
    vid.release()
    video.release()
    cv2.destroyAllWindows()

def make_img():
    img_path = r"c:\Users\Lenovo\Downloads\Lane.png"
    img = cv2.imread(img_path)
    result = find_line(img)
    cv2.imshow("Find line", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def play_vid():
    cap = cv2.VideoCapture('video.avi')
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Uncomment to run
if __name__ == "__main__":
    # make_img()
    # make_vid()
    play_vid()

