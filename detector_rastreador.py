import cv2
import sys
from random import randint


tracker = cv2.TrackerCSRT_create()

video = cv2.VideoCapture('./videos/walking.avi')

if not video.isOpened():
    print("Não foi possivel carregar o video")
    sys.exit()

ok, frame = video.read()

if not ok:
    print("Não foi possivel ler o frame")
    sys.exit()
    
detector = cv2.CascadeClassifier('./cascade/fullbody.xml')

def detectar():
    while True:
        ok, frame = video.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detectMultiScale(frame_gray)

        for (x, y, l, a) in detections[1:]:
            if x > 0:
                print('Detecção efetuada pelo haar cascade')
                return x, y, l, a


bbox = detectar()

ok = tracker.init(frame, bbox)

color = (randint(0,255), randint(0,255), randint(0,255))

while True:

    ok, frame = video.read()

    if not ok:
        break

    ok, bbox = tracker.update(frame)

    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2, 1)
    else:
        cv2.putText(frame, 'Falha no rastreamento, recuperando com Haar Cascade', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)
        bbox = detectar()
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)


    cv2.imshow('tracking', frame) 

    if cv2.waitKey(1) & 0xFF == 27:
        break