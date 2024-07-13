import cv2
import numpy as np
import dlib

dir_path = __file__[:len(__file__)-len(__file__.split('\\')[-1])]

tile = 100
count = 0
talk = True

datfile = rf"{dir_path}shape_predictor_68_face_landmarks.dat"

cap = cv2.VideoCapture(0)
_, frame = cap.read()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(datfile)

def pytago(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

while True:
    count += 1
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)

    for face in faces:
        landmarks = predictor(gray_frame, face)

        def notes():
            for i in range(68):
                dotx = (landmarks.part(i).x, landmarks.part(i).y)
                cv2.circle(frame, dotx, 0, (255, 0, 0), 5)
        notes()

        def dot(x):
            return (landmarks.part(x).x, landmarks.part(x).y)

        def basewidth():
            return pytago(dot(28), dot(29))

        def eyes():
            lew = pytago(dot(38), dot(40))
            rew = pytago(dot(43), dot(47))
            base = basewidth()
            cv2.putText(frame, 'L: ' + str(round(lew, 2)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
            cv2.putText(frame, 'R: ' + str(round(rew, 2)), (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
            cv2.putText(frame, 'B: ' + str(round(base, 2)), (10, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
            if lew < (base * 7.5 / 18):
                cv2.putText(frame, 'Left eye closed', (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
            else:
                cv2.putText(frame, 'Left eye opened', (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

            if rew < (base * 7.5 / 18):
                cv2.putText(frame, 'Right eye closed', (450, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
            else:
                cv2.putText(frame, 'Right eye opened', (450, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
        eyes()

        cv2.putText(frame, 'Time: ' + str(round(count, 2)), (10, 210), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))

        def smile(count, tile):
            smi = pytago(dot(48), dot(54))
            basesm = pytago(dot(7), dot(9))
            if count == 20:
                tile = smi / basesm

            if smi > tile * basesm:
                cv2.putText(frame, 'Smiling', (255, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            return tile
        tile = smile(count, tile)

        def mouth():
            mid = pytago(dot(62), dot(66))
            cv2.putText(frame, 'M: ' + str(round(mid, 2)), (10, 130), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
            if mid > 8 and talk:
                cv2.putText(frame, 'Talking', (255, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        mouth()

        def face():
            if pytago(dot(2), dot(30)) / pytago(dot(30), dot(13)) >= 1.5:
                cv2.putText(frame, 'Right look', (355, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            elif pytago(dot(13), dot(30)) / pytago(dot(30), dot(2)) >= 1.5:
                cv2.putText(frame, 'Left look', (155, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            else:
                if pytago(dot(8), dot(33)) / pytago(dot(33), dot(27)) >= 1.5:
                    cv2.putText(frame, 'Up look', (255, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
                else:
                    cv2.putText(frame, 'Straight look', (255, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        face()

    cv2.imshow('App', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
