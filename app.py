from mediapipe.framework.formats import landmark_pb2
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import math

app = Flask(__name__)
camera = cv2.VideoCapture(0)
# the drawing utilities
mp_drawing = mp.solutions.drawing_utils
# import the pose solution model
mp_pose = mp.solutions.pose


def generate_frames():
    while camera.isOpened():

        # read the camera frame
        success, frame = camera.read()

        # When the camera unable to read the frame
        if not success:
            break

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # 1. Detection
            results = pose.process(frame)

            # 2. Add the renders by OpenCV
            try:

                nose, right_shoulder, left_shoulder, right_hip, left_hip = extract_landmarks(
                    results.pose_landmarks.landmark)

                # 2.2 Get the distance based on drawable
                right_side_distance = get_distance(right_shoulder, right_hip)
                left_side_distance = get_distance(left_shoulder, left_hip)
                distance = (right_side_distance + left_side_distance)/2
                # print("distance")

                displayed_distance = str(f'{distance:.4f}')

                # Show massage
                # we getting those to change them to the size of the cam feed
                IMG_HEIGHT, IMG_WIDTH = frame.shape[:2]

                cv2.putText(frame, displayed_distance,
                            tuple(np.multiply(
                                nose[:2], [IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass

            # Adding drawable
            landmark_subset = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    # results.pose_landmarks.landmark[0],
                    results.pose_landmarks.landmark[11],
                    results.pose_landmarks.landmark[12],
                    results.pose_landmarks.landmark[23],
                    results.pose_landmarks.landmark[24],
                ]
            )
            # print(landmark_subset.landmark)
            mp_drawing.draw_landmarks(
                # frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,)
                frame, landmark_list=landmark_subset)

            """draw the subset
            poses = landmark_subset.landmark
            for i in range(0, len(poses)-1, 2):
                start_idx = [
                    poses[i].x,
                    poses[i].y
                ]
                end_idx = [
                    poses[i+1].x,
                    poses[i+1].y
                ]
                IMG_HEIGHT, IMG_WIDTH = frame.shape[:2]
                # print(start_idx)

                cv2.line(frame,
                         tuple(np.multiply(start_idx[:2], [
                               IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                         tuple(np.multiply(end_idx[:2], [
                               IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                         (255, 0, 0), 9)
                """

        # Render the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Process image


def process_image(frame, pose):
    # 1. Formatting input to MediaPipe, recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # 2. Detection
    results = pose.process(image)
    image.flags.writeable = True
    yield results

# Extract landmarks


def extract_landmarks(landmarks):
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
            landmarks[mp_pose.PoseLandmark.NOSE.value].y,
            landmarks[mp_pose.PoseLandmark.NOSE.value].z]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]

    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z]

    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]

    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].z]
    return nose, right_shoulder, left_shoulder, right_hip, left_hip

# Get the distance


def get_distance(a, b):
    return math.sqrt(
        (a[0]-b[0])**2 +
        (a[1]-b[1])**2 +
        0.5*(a[2]-b[2])**2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()
