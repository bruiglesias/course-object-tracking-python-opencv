import cv2
import sys
from random import randint

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# print(major_ver, minor_ver, subminor_ver)

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):

    if trackerType == tracker_types[0]:
        tracker = cv2.TrackerBoosting_create()
    if trackerType == tracker_types[1]:
        tracker = cv2.TrackerMIL_create()
    if trackerType == tracker_types[2]:
        tracker = cv2.TrackerKCF_create()
    if trackerType == tracker_types[3]:
        tracker = cv2.TrackerTLD_create()
    if trackerType == tracker_types[4]:
        tracker = cv2.TrackerMedianFlow_create()
    if trackerType == tracker_types[5]:
        tracker = cv2.TrackerMOSSE_create()
    if trackerType == tracker_types[6]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None 
        print('Incorrect name')
        print('avaiable trackers: ' + str(tracker_types))

    return tracker


video = cv2.VideoCapture('./videos/race.mp4')

if not video.isOpened():
    print("Não foi possivel carregar o video")
    sys.exit()

ok, frame = video.read()

if not ok:
    print("Não foi possivel ler o frame")
    sys.exit()

bboxs = []
colors = []

while True:
    
    bbox = cv2.selectROI('Multitracker',frame)
    bboxs.append(bbox)
    colors.append((randint(0,255), randint(0,255), randint(0,255)))
    print('Pressione Q para sair e começar a rastrear')
    print('Pressione qualquer outra tecla para continuar')
    
    k = cv2.waitKey(0) & 0xFF
    
    if (k == 113):
        break

trackerType = 'CSRT'

multitracker = cv2.MultiTracker_create()

for bbox in bboxs:
    multitracker.add(createTrackerByName(trackerType), frame, bbox)

while video.isOpened():

    ok, frame = video.read()

    if not ok:
        break


    ok, boxes = multitracker.update(frame)
    if ok:
        for i, newbox in enumerate(boxes):  
            (x, y, w, h) = [int(v) for v in newbox]
            cv2.rectangle(frame, (x,y), (x+w, y+h), colors[i], 2, 1)
            cv2.putText(frame, trackerType + 'Traker', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 170, 50), 2)
    else:
        cv2.putText(frame, 'Falha no rastreamento', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)
    
    cv2.imshow('tracking', frame) 
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()

