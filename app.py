from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Pose and Hands classes
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()

# Initialize MediaPipe drawing utils
mp_drawing = mp.solutions.drawing_utils


# Function to detect peace sign gesture
def detect_peace_sign(hand_landmarks):
    # Define the index, middle, ring, and pinky finger tip landmarks
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Check if the index and middle fingers are extended and the others are curled
    index_extended = index_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_extended = middle_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_curled = ring_finger_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_curled = pinky_finger_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

    if index_extended and middle_extended and ring_curled and pinky_curled:
        return True
    return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed', methods=['POST'])
def video_feed():
    # Receive the image from the client
    nparr = np.frombuffer(request.data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the image from BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for pose estimation and hand tracking
    pose_results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # Initialize gesture variable
    gesture = 'No gesture detected'

    # Draw the pose landmarks on the original frame
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )

    # Draw hand landmarks and detect gestures
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            # Check for the peace sign gesture
            if detect_peace_sign(hand_landmarks):
                gesture = 'Peace Sign Detected'
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Encode the frame back to JPEG to send back to the client
    ret, jpeg = cv2.imencode('.jpg', frame)
    frame_encoded = jpeg.tobytes()

    # Send the gesture as a response header
    response = Response(frame_encoded, mimetype='image/jpeg')
    response.headers['Gesture'] = gesture

    return response


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

