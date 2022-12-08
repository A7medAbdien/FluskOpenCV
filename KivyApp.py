# Import Kivy dependencies
# App Layer
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# to make continuous updates to our app
from kivy.clock import Clock
# we should convert our OpenCV image to Texture and set our image equal to that texture
from kivy.graphics.texture import Texture
# to show some matrices
from kivy.logger import Logger

import cv2
import mediapipe as mp
import numpy as np
import math
# the drawing utilities
mp_drawing = mp.solutions.drawing_utils
# import the pose solution model
mp_pose = mp.solutions.pose


class CamApp(App):

    state = "Okay"
    distance = 0

    def build(self):

        # layout component
        self.web_cam = Image(size_hint=(1, .8))
        self.distance_label = Label(
            text=str(self.distance), size_hint=(1, .1))  # the distance
        self.state_label = Label(
            text=self.state, size_hint=(1, .1))  # the state

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.distance_label)
        layout.add_widget(self.state_label)

        # get image form webcam
        self.capture = cv2.VideoCapture(1)
        # schedule_interval: will do this event every x seconds
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    ###############
    # Run continuously to get webcam feed
    # convert raw OpenCV image -> OpenGL texture can be rendered
    def update(self, *args):
        # Read frame form OpenCV

        # ---------------- Capturing image -------------------
        ret, frame = self.capture.read()

        # ---------------- Processing image ------------------
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

            # 1. Formatting input to MediaPipe, recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 2. Detection
            results = pose.process(image)

            # 3. Formatting output to OpenCV, recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 4. Render the result by OpenCV
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
                IMG_HEIGHT, IMG_WIDTH = image.shape[:2]

                cv2.putText(image, displayed_distance,
                            tuple(np.multiply(
                                nose, [IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                            255, 255), 2, cv2.LINE_AA
                            )
            except:
                pass

            # Adding drawable
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,)

            frame = image
        # --------------- Rendering image---------------------
        frame = frame[0:frame.shape[0], 0:frame.shape[0], :]

        # Flip horizontally and convert image to texture
        # save cv2 image
        buf = cv2.flip(frame, 0).tobytes()

        # Create image texture with this frame sizes and color format (bgr is the color format of opencv images)
        img_texture = Texture.create(
            size=(frame.shape[0], frame.shape[1]), colorfmt='bgr')

        # Convert image  to Texture
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        # Assign the texture to the webcam
        self.web_cam.texture = img_texture


if __name__ == '__main__':
    CamApp().run()
