from Jolo_Recognition.JoloRecognition import JoloRecognition
import cv2
import time

# image = cv2.imread('Images/AFJ.jpg')
# # resulta = JoloRecognition().Face_Multiple_Compare(image)

# face_results = JoloRecognition().Face_Multiple_Compare(image)
# for result in face_results:
#     print("Result Name:", result[0])
#     print("Result box:", result[1])

# ======================= Live streaming detection ======================= #



# Load the Haar cascade XML file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video capture object for the camera (index 0)
cap = cv2.VideoCapture(0)


# Padding ratio to increase the size of the cropped face
top_padding_ratio = 0.3
side_padding_ratio = 0.2
bottom_padding_ratio = 0.1

# Timer variables
start_time = time.time()
interval = 2  # Time interval in seconds for face recognition

# recognize person
recognized_persons ={}

while True:
    # Read the current frame from the video capture
    ret, frame = cap.read()
    
    frame = cv2.flip(frame,1)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    # Process each detected face
    for (x, y, w, h) in faces:
        
        # Calculate padding values
        top_padding = int(h * top_padding_ratio)
        side_padding = int(w * side_padding_ratio)
        bottom_padding = int(h * bottom_padding_ratio)

        # Apply padding to the face region
        xs = max(0, x - side_padding)
        ys = max(0, y - top_padding)
        ws = min(frame.shape[1], w + 2 * side_padding)
        hs = min(frame.shape[0], h + top_padding + bottom_padding)

        # Crop the face from the frame
        face_crop = frame[ys:ys+hs, xs:xs+ws]
        
 
        # Perform face recognition every 2 seconds
        current_time = time.time()
        if current_time - start_time >= interval:
            # Face recognition
            result = JoloRecognition().Face_Compare(face=face_crop)

            # Print the result
            print("Result Name:", result[0])
            recognized_persons[(x, y, w, h)] = result[0]
            
            # Reset the timer
            start_time = current_time

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # label the person
        # Label the person
        if (x, y, w, h) in recognized_persons:
            label = recognized_persons[(x, y, w, h)]
            cv2.putText(frame, label, (x, y+h+ 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),1)
            
            # Check if the label duration has passed
            if current_time - start_time >= 5:
                # Remove the label from recognized_persons
                del recognized_persons[(x, y, w, h)]
        
    # Display the cropped face image
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()

