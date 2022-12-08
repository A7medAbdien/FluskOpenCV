# Flask and OpenCV

Set-ExecutionPolicy Unrestricted -Scope Process

.\script\activate

pip install -r .\requirements.txt

#

this the [MedeaPipe repository code](https://github.com/google/mediapipe/blob/340d7651af8caca795220b81124b2a3e557f4784/mediapipe/python/solutions/drawing_utils.py#L120) for `draw_landmarks` method

```py
if connections:
    num_landmarks = len(landmark_list.landmark)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:

        start_idx = connection[0]
        end_idx = connection[1]

        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
            raise ValueError(f'Landmark index is out of range. Invalid connection '
                            f'from landmark #{start_idx} to landmark #{end_idx}.')

        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            # I didn't try to deg in this line, but it seems to be used to get the global color and thickness
            drawing_spec = connection_drawing_spec[connection] if isinstance(connection_drawing_spec, Mapping) else connection_drawing_spec
            """
            idx_to_coordinates is list of landmark_px
            and let us say that landmark_px is kind of tubules of (landmark.x, landmark.y, image_cols, image_rows), so it has x and y
            then they used to draw the opencv line
            """
            # cv2.line(image, start_point, end_point, color, thickness)
            cv2.line(image, idx_to_coordinates[start_idx],idx_to_coordinates[end_idx], drawing_spec.color,drawing_spec.thickness)

```

so we can draw them manually using our landmark_subset

```py
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
# print(landmark_subset)
mp_drawing.draw_landmarks(
    # frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,)
    frame, landmark_list=landmark_subset)

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
                # here we change coordinates to fit to the camera feed
                tuple(np.multiply(start_idx[:2], [
                    IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                tuple(np.multiply(end_idx[:2], [
                    IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                (255, 0, 0), 9)
```

