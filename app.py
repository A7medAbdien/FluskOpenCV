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
        # draw_keypoints(frame, person, confidence_threshold)

# Draw Keypoints


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)


# [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]
# [0,5,6,11,12]

EDGES = {
    # (0, 1): 'c',
    # (0, 2): 'c',
    # (1, 3): 'm',
    # (2, 4): 'c',
    # (0, 5): 'm',
    # (0, 6): 'c',
    # (5, 7): 'm',
    # (7, 9): 'm',
    # (6, 8): 'c',
    # (8, 10): 'c',
    (5, 6): 'shoulders',
    (5, 11): 'right shoulder',
    (6, 12): 'left shoulder',
    (11, 12): 'hips',
    # (11, 13): 'm',
    # (13, 15): 'm',
    # (12, 14): 'c',
    # (14, 16): 'c'
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
