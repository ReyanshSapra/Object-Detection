import cv2

wCam, hCam = 640, 480

#img = cv2.imread(r"C:\Users\ms080\OneDrive\reyansh\The owner\Regular Python\OpenCV\objdetection\image.png")
cap = cv2.VideoCapture(0)
cap.set(3,1000)
cap.set(4,1000)

classNames = []
classFile = r"C:\Users\ms080\OneDrive\reyansh\The owner\Regular Python\OpenCV\objdetection\coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = r"C:\Users\ms080\OneDrive\reyansh\The owner\Regular Python\OpenCV\objdetection\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = r"C:\Users\ms080\OneDrive\reyansh\The owner\Regular Python\OpenCV\objdetection\frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(255,0,0), thickness=2)
            cv2.putText(img,classNames[classId-1], (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)


    cv2.imshow("output",img)
    cv2.waitKey(1)