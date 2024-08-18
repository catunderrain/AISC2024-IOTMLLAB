import cv2
import time
import math
from Hand_Detect_Module import *
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import pyautogui
import pydirectinput
import mouse


#khởi tạo camera
camera = cv2.VideoCapture(0)
width_cam , height_cam = 800, 600
camera.set(width_cam, 4)
camera.set(height_cam, 3)
previous_time = 0
mode = ""
id_mode = 0


# #cofig volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = -20
maxVol = volRange[1]
dis_min = 30
dis_max = 150
volBar = 400
volPer = 0
vol = 0

# cofig cursor
pyautogui.FAILSAFE = False
w_scr, h_scr = pyautogui.size()
print(w_scr, h_scr)

#xac dinh ngon tay
finger_index = [4,8,12,16,20]

while True:
    ret, frame = camera.read()
    current_time = time.time()
    # Doi sang he mau rgb và flip frame
    frame = cv2.flip(frame, 1)
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    multiLanmarks = find_hands(new_frame)
    # tiếp tục xử lí nếu nhận diện được bàn tay
    if multiLanmarks:
        hand_lst = []
        for handLms in multiLanmarks:
            # vẽ lên frame
            mpDraw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            for idx, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_lst.append((cx, cy))
        for point in hand_lst:
            cv2.circle(frame, point, 5, (0, 0, 255), cv2.FILLED)
        # print(hand_lst[8])
        # print(len(hand_lst))
        hand = []
        if len(hand_lst) > 0:
            if hand_lst[finger_index[0]][0] < hand_lst[finger_index[0] - 1][0]:
                hand.append(1)
            else:
                hand.append(0)
        for id in range(1, 5):
            if hand_lst[finger_index[id]][1] < hand_lst[finger_index[id] - 2][1]:
                hand.append(1)
            else:
                hand.append(0)
        print(hand)
        if hand == [0,0,0,0,0]:
            mode = " "
        if (hand == [1,1,0,0,0]):
            mode = "Volume"
        elif (hand == [1,1,1,0,0] or hand == [0,1,1,0,0]):
            mode = "Cursor"
        elif (hand == [0,1,0,0,0] or hand == [1, 0, 0, 0, 0]):
            mode = "Srcoll"


        if mode == "Volume":
            x4, y4 = hand_lst[4][0], hand_lst[4][1]
            x8, y8 = hand_lst[8][0], hand_lst[8][1]
            xc, yc = (x4+x8)//2, (y4+y8)//2
            distance = math.sqrt(pow(x4-x8,2)+pow(y4-y8,2))
            cv2.circle(frame, (xc,yc), 5, (0,255,255),cv2.FILLED)
            cv2.line(frame, (x4,y4), (x8,y8), (0,255,255),1)
            # print(distance)

            # nội suy âm thanh, % âm thanh và dãy âm thanh
            vol = np.interp(distance, [dis_min, dis_max], [minVol, maxVol])
            volBar = np.interp(vol, [minVol, maxVol], [400, 150])
            volPer = np.interp(vol, [minVol, maxVol], [0, 100])
            print(vol)
            volN = int(vol)
            if volN % 4 != 0:
                volN = volN - volN % 4
                if volN >= 0:
                    volN = 0
                elif volN <= -64:
                    volN = -64
                elif vol >= -11:
                    volN = vol

            #    print(int(length), volN)
            volume.SetMasterVolumeLevel(vol, None)

            cv2.rectangle(frame, (30, 150), (55, 400), (209, 206, 0), 3)
            cv2.rectangle(frame, (30, int(volBar)), (55, 400), (215, 255, 127), cv2.FILLED)
            cv2.putText(frame, f'{int(volPer)}%', (25, 430), cv2.FONT_HERSHEY_COMPLEX, 0.9, (209, 206, 0), 3)
            time.sleep(0.05)



        if mode == "Cursor":
            # lấy tọa độ ngón trỏ
            x8, y8 = hand_lst[8][0], hand_lst[8][1]

            # tạo vùng di chuyển chuột
            cv2.rectangle(frame, (300, 120), (600, 240), (255,0,0),2)

            if 300 <x8< 600 and 120<y8<240:
                X = int(np.interp(x8, [300, 600], [0, w_scr - 1]))
                Y = int(np.interp(y8, [120, 240], [0, h_scr - 1]))
                if X%2 != 0 or Y %2 != 0:
                    X = X - X%2
                    Y = Y - Y%2
                pydirectinput.moveTo(X,Y)

                if hand == [0,1,1,0,0]:
                    cv2.circle(frame, (hand_lst[4][0], hand_lst[4][1]), 10, (0, 255, 255), cv2.FILLED)
                    pydirectinput.click()


        if mode == "Srcoll":
            if hand == [0,1,0,0,0]:
                time.sleep(0.3)
                mouse.wheel(-1)
            elif hand == [1,0,0,0,0]:
                time.sleep(0.3)
                mouse.wheel(1)

    cv2.putText(frame, f"Mode: {mode}", (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    if not ret:
        continue
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    text = f"FPS:{fps:.0f}"
    cv2.putText(frame, text, (100,100), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,0),1)
    cv2.imshow("Final Project", frame)
    if (cv2.waitKey(1) == ord("q")):
        break
camera.release()
cv2.destroyAllWindows()
