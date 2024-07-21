import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines, color=[0, 255, 0], thickness=10):
    if lines is None:
        return img
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), color, thickness)
    
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def process_image(image):
    height, width = image.shape[:2]
    region_of_interest_vertices = [
        (0, height),
        (width, height),
        (int(width / 2), int(height * 0.6))
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    lines = hough_lines(cropped_image, rho=1, theta=np.pi/180, threshold=50,
                        min_line_len=100, max_line_gap=50)
    image_with_lines = draw_lines(image, lines)
    return image_with_lines

def detect_road_lines(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
    processed_image = process_image(image)
    cv2.imshow('Road Line Detection', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_road_lines(r'C:\Users\Lenovo\Desktop\AISC2024 IOTMLLAB\Line Detect\Lane.png')