from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import playsound

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
EAR_THRESHOLD = 0.25
FRAME_CHECK = 20
ALARM_SOUND = "alarm.wav"

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Extract the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize the video stream and sleep counter
cap = cv2.VideoCapture(0)
sleep_counter = 0

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Loop over the detected faces
    for face in faces:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the eye aspect ratio for each eye
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Visualize the eye regions
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the threshold
        if ear < EAR_THRESHOLD:
            sleep_counter += 1

            # If the eyes have been closed for a sufficient number of frames, sound the alarm
            if sleep_counter >= FRAME_CHECK:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                playsound.playsound(ALARM_SOUND)  # Play alarm sound

        else:
            sleep_counter = 0

    # Display the resulting frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Exit the loop if the 'q' key is pressed
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()
