import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_line(img):
    kernel1 = np.array([[-1, -0, -1],
                    [-2,9,-2],
                    [-1,-0,-1]])
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=np.ones((3,3),np.uint8))
    gray_img = cv2.convertScaleAbs(gray_img, alpha=1.4, beta=0)
    gray_img = cv2.filter2D(src=gray_img, ddepth=-1, kernel=kernel1)
    # gray_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=np.ones((3,3),np.uint8))
    gray_img = cv2.dilate(gray_img, kernel=np.ones((5, 5), np.uint8))
    # gauss_img = cv2.GaussianBlur(gray_img, (5,5), 0 )
    # gauss_img = cv2.morphologyEx(gray_img,cv2.MORPH_OPEN, kernel=np.ones((3,3),np.uint8))
    canny_img = cv2.Canny(gray_img, 200, 255)
    # canny_img = cv2.Canny(gauss_img, 75, 150)
    # plt.subplot(1,2,1)
    # plt.imshow(gray_img, cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(gauss_img, cmap='gray')
    # plt.subplot(1,3,3)
    # plt.imshow(canny_img, cmap='gray')
    # plt.show()
    # cv2.waitKey(0)
    roi_img = find_roi(canny_img)
    contours, _ = cv2.findContours(roi_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,0,255), 2)
    return img
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


def make_vid():
    vid = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\AISC2024 IOTMLLAB\Line Detect\lane.mp4")
    print(vid.read()[1].shape)
    success = 1
    count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('video.avi', fourcc, 30, (vid.read()[1].shape[1], vid.read()[1].shape[0]))
    while success:
        success, frame = vid.read()
        if success:
            count += 1
            print(f'Frame {count}')
            result = find_line(frame)
            video.write(result)
    cv2.destroyAllWindows()
    video.release()


def make_img():
    img_path = r"c:\Users\Lenovo\Downloads\Lane.png"
    img = cv2.imread(img_path)
    result = find_line(img)
    cv2.imshow("Find line", result)
    cv2.waitKey(0)


def play_vid():
    cap = cv2.VideoCapture('video.avi')
    if (cap.isOpened()== False):
        print("Error opening video file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(0) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    
    

# if __name__ == "__main__":
#     # make_img()
#     # make_vid()
    
cap = cv2.VideoCapture('video.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()