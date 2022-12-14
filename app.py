import math
from flask import Flask, render_template, Response
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

app = Flask(__name__)
camera = cv2.VideoCapture(1)


def generate_frames():
    while camera.isOpened():

        # read the camera frame
        success, frame = camera.read()

        # When the camera unable to read the frame
        if not success:
            break

        # Resize image
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
        input_img = tf.cast(img, dtype=tf.int32)

        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[
            :, :, :51].reshape((6, 17, 3))

        # Render keypoints and landmarks
        loop_through_people(frame, keypoints_with_scores, EDGES, 0.3)

        # Render the frame
        buffer = cv2.imencode('.jpg', frame)[1]
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to loop through each person detected and render


def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_distance(frame, person, confidence_threshold)


# [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]
# [0,5,6,11,12]

EDGES = {
    (5, 6): 'shoulders',
    (5, 11): 'right shoulder',
    (6, 12): 'left shoulder',
    (11, 12): 'hips',
}


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)),
                     (int(x2), int(y2)), (0, 0, 255), 4)


# Get the distance


def draw_distance(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    nose = shaped[0]
    right_shoulder = shaped[5]
    left_shoulder = shaped[6]
    right_hip = shaped[11]
    left_hip = shaped[12]

    # 2.2 Get the distance based on drawable
    #
    if (right_shoulder[-1] > confidence_threshold) & (left_shoulder[-1] > confidence_threshold) & (right_hip[-1] > confidence_threshold) & (left_hip[-1] > confidence_threshold):
        right_side_distance = get_distance(right_shoulder, right_hip)
        left_side_distance = get_distance(left_shoulder, left_hip)
        distance = (right_side_distance + left_side_distance)/2
        displayed_distance = str(f'{distance:.2f}')
    else:
        displayed_distance = "UnKnown"

    # Show massage
    ky, kx, kp_conf = nose
    if kp_conf > confidence_threshold:
        cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
        cv2.putText(frame, displayed_distance,
                    (int(kx), int(ky)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


def get_distance(a, b):
    return math.sqrt(
        (a[0]-b[0])**2 +
        (a[1]-b[1])**2)


@ app.route('/')
def index():
    return render_template('index.html')


@ app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()
