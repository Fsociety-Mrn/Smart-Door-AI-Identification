from Jolo_Recognition.JoloRecognition import JoloRecognition
import cv2

image = cv2.imread('Images/AFJ.jpg')
# resulta = JoloRecognition().Face_Multiple_Compare(image)

face_results = JoloRecognition().Face_Multiple_Compare(image)
for result in face_results:
    print("Result Name:", result[0])
    print("Result box:", result[1])
