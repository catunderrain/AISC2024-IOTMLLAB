import cv2
import numpy as np
import dlib
from math import hypot
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
i = 0

# Loading Camera and images
cap = cv2.VideoCapture(0)
glass_image = cv2.imread(os.path.join(dir_path, "glass.png"))
nose_image = cv2.imread(os.path.join(dir_path, "pignose.png"))

_, frame = cap.read()
rows, cols, _ = frame.shape
glass_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(dir_path, "shape_predictor_68_face_landmarks.dat"))

while True:
    _, frame = cap.read()
    glass_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray_frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        # Glass coordinates
        center_glass = (landmarks.part(27).x, landmarks.part(27).y)
        left_glass = (landmarks.part(0).x, landmarks.part(0).y)
        right_glass = (landmarks.part(16).x, landmarks.part(16).y)
        glass_width = int(hypot(left_glass[0] - right_glass[0], left_glass[1] - right_glass[1]) * 1.05)
        glass_height = int(glass_width * 2 / 5.7)

        # Nose coordinates
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        nose_width = int(hypot(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) * 2)
        nose_height = int(nose_width * 0.77)

        # New glass position
        bot_left = (int(center_glass[0] - glass_width / 2), int(center_glass[1] - glass_height / 2))
        top_left = (int(center_nose[0] - nose_width / 2), int(center_nose[1] - nose_height / 2))

        # Adding the new nose
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
        nose_area = frame[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_pig)
        frame[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = final_nose

        # Adding the new glass
        glass_pig = cv2.resize(glass_image, (glass_width, glass_height))
        glass_pig_gray = cv2.cvtColor(glass_pig, cv2.COLOR_BGR2GRAY)
        _, glass_mask = cv2.threshold(glass_pig_gray, 0, 255, cv2.THRESH_BINARY_INV)
        glass_area = frame[bot_left[1]: bot_left[1] + glass_height, bot_left[0]: bot_left[0] + glass_width]
        glass_area_no_glass = cv2.bitwise_and(glass_area, glass_area, mask=glass_mask)
        final_glass = cv2.add(glass_area_no_glass, glass_pig)
        frame[bot_left[1]: bot_left[1] + glass_height, bot_left[0]: bot_left[0] + glass_width] = final_glass

    cv2.imshow("Frame", cv2.flip(frame, 1))
    key = cv2.waitKey(1)
    if key == 99:
        if not os.path.exists(os.path.join(dir_path, 're')):
            os.makedirs(os.path.join(dir_path, 're'))
        cv2.imwrite(os.path.join(dir_path, f're/img{i}.png'), cv2.flip(frame, 1))
        i += 1
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
