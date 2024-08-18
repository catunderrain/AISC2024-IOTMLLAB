import mediapipe as mp
import cv2


# khởi tạo model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils


def find_hands(img):
    results = hands.process(img)
    multiLanmarks = results.multi_hand_landmarks
    return multiLanmarks

