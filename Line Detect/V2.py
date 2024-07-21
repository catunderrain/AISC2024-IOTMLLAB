import cv2
import numpy as np

def detect_road_lines(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
    # Resize the image to a fixed size for consistency
    image = cv2.resize(image, (1280, 720))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Define a mask to focus on the region of interest
    mask = np.zeros_like(edges)
    height, width = edges.shape
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.45), int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Use HoughLinesP to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)
    
    # Create a copy of the original image to draw lines on
    line_image = np.copy(image)
    
    # Draw lines on the image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    
    # Combine the original image with the line image
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    # Show the final output
    cv2.imshow('Road Line Detection', combo_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_road_lines(r'C:\Users\Lenovo\Desktop\AISC2024 IOTMLLAB\Line Detect\Lane.png')
