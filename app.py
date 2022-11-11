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

            # 2. Add the result by OpenCV
            # 4.1 Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]

                # 4.2 Get the distance based on drawable
                right_side_distance = math.sqrt(
                    (right_shoulder[0]-right_hip[0])**2 + (right_shoulder[1]-right_hip[1])**2)
                left_side_distance = math.sqrt(
                    (left_shoulder[0]-left_hip[0])**2 + (left_shoulder[1]-left_hip[1])**2)
                distance = (right_side_distance + left_side_distance)/2
                # print(distance)

                displayed_distance = str(f'{distance:.4f}')

                # Show massage
                # we process the elbow coordinates to change them to the size of the cam feed
                IMG_HEIGHT, IMG_WIDTH = frame.shape[:2]

                cv2.putText(frame, displayed_distance,
                            tuple(np.multiply(
                                nose, [IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                            255, 255), 2, cv2.LINE_AA
                            )
            except:
                pass

            # Adding drawable
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,)

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

    # 3. Formatting output to OpenCV, recolor back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()
