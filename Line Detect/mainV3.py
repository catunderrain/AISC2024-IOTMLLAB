import cv2
import numpy as np
import time
def find_line(img):
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel=np.ones((3,3),np.uint8))
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=-80)
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R,G,B = cv2.split(lab_image)
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.imshow('', R)
    cv2.waitKey(0)
    # gray_img = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    gray_img = R
    # gray_img = cv2.GaussianBlur(gray_img, (5,5), 0 )
    
    # gray_img = cv2.convertScaleAbs(gray_img, alpha=1.4, beta=-30)
    cv2.imshow('B', gray_img)
    cv2.waitKey(0)
    # gray_img = cv2.erode(gray_img, kernel=np.ones((5, 5), np.uint8))
    
    edges = cv2.Canny(gray_img, 100, 200)

    roi_vertices = [(270, 670), (600, 400), (1127, 712)]
    
    # roi_img = find_roi(edges)
    roi_img = roi(edges, np.array([roi_vertices], np.int32))
    # contours, _ = cv2.findContours(roi_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(roi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lines = cv2.HoughLinesP(roi_img, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    cv2.drawContours(img, contours, -1, (0,0,255), 2)
    def draw_lines(image, hough_lines):

        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return image


    final_img = draw_lines(img, lines)
    
    
    return img
    return final_img

def roi(image, vertices):
    mask = np.zeros_like(image)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


def find_roi(img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    shape = img.shape
    center = [shape[1] // 2, shape[0] // 2]
    left_point = [0, shape[0]]
    right_point = [shape[1], shape[0]]
    points = np.array([center, left_point, right_point])
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], color=(255, 255, 255))
    roi_img = cv2.bitwise_and(img, img, mask=mask)
    return roi_img

# def find_roi(img):
#     mask = np.zeros_like(img)
#     height, width = img.shape
#     polygon = np.array([[
#         (int(width * 0.1), height),
#         (int(width * 0.9), height),
#         (int(width * 0.55), int(height * 0.6)),
#         (int(width * 0.45), int(height * 0.6))
#     ]], np.int32)
#     cv2.fillPoly(mask, polygon, 255)
#     roi_img = cv2.bitwise_and(img, mask)
#     return roi_img

def make_vid():
    vid = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\AISC2024 IOTMLLAB\Line Detect\lane.mp4")
    success = True
    count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.avi', fourcc, 30, (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while success:
        success, frame = vid.read()
        if success:
            count += 1
            print(f'Frame {count}')
            result = find_line(frame)
            video.write(result)
    print('Done')
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
            time.sleep(0.02)
            if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    make_img()
    # make_vid()
    while True:
        play_vid()
        
