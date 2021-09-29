import numpy as np
import cv2
import os 

# Load the cascade
face_cascade = cv2.CascadeClassifier('frontalface.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

# load age pred model from pytorch
onnx_model_path = "agepred.onnx"
net =  cv2.dnn.readNetFromONNX(onnx_model_path) 

# labels for predictions
labels = os.listdir("test")

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_color = img[y:y + h, x:x + w]
        
        save_dest = "DetectedFaces/" + str(w) + str(h) + '_faces.jpg'
        cv2.imwrite(save_dest, roi_color)

        face = cv2.imread(save_dest)
        face_resized = cv2.resize(face, (32,32))
        input_blob = cv2.dnn.blobFromImage(face_resized, 1, (32,32), swapRB=False, crop=False)
        net.setInput(input_blob)
        preds = net.forward()
        biggest_pred_index = np.array(preds)[0].argmax()
        cv2.putText(img, labels[biggest_pred_index],
        (x + w,y + h), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0),2 )
        os.remove(save_dest)

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()