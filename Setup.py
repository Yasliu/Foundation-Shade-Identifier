import cv2
import pandas as pd
import os
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage.color import deltaE_ciede2000


# Loading the image from the directory using imread
model_path = 'face_landmarker.task'
df = pd.read_csv('master_foundation_db.csv')

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    min_face_detection_confidence=0.5,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# we obtain image as RGB
def find_comparison(image):
    target_lab = []

    print("Cropping image...")
    h, w, _ = image.shape

    #extract landmarks

    # loading the image here
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = landmarker.detect(mp_image)
            
    if detection_result.face_landmarks:
        # Get the first face detected
        face = detection_result.face_landmarks[0]

        # we are going for the left cheek polygon
        cheek_indices = [117, 118, 101, 121, 47, 126, 209]

        polygon_points = np.array([
            [int(face[i].x * w), int(face[i].y * h)] for i in cheek_indices
        ], dtype=np.int32)

        #creation of mask and pixels
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], 255)

        # bitwise_and to get the pixels of the polygon
        cheek_pixels = image[mask > 0] # Array of RGB pixels

        # Color conversion to LAB
        cheek_pixels_float = cheek_pixels.astype(np.float32) / 255.0
        cheek_pixels_reshaped = cheek_pixels_float.reshape(1, -1, 3)
        lab_pixels = cv2.cvtColor(cheek_pixels_reshaped, cv2.COLOR_RGB2LAB)[0]

        # sort based on lightness
        sorted_indices = lab_pixels[:, 0].argsort()
        sorted_labs = lab_pixels[sorted_indices]

        # dropping the first and last 20%
        num_pixels = len(sorted_labs)
        lower_bound = int(num_pixels * 0.2)
        upper_bound = int(num_pixels * 0.8)
        trimmed_labs = sorted_labs[lower_bound:upper_bound]

        # averaging the middle
        target_lab = trimmed_labs.mean(axis=0)

        best_matches = find_my_match(target_lab)

    else:
        print("Error: No face detected in the image.")
        return None
    

    return best_matches

def find_my_match(target_lab):
    foundation_matrix = df[['L_val', 'a_val', 'b_val']].values.astype(float)

    temp_df = df.copy()
    temp_df['dist'] = deltaE_ciede2000(target_lab, foundation_matrix)
    temp_df = temp_df.sort_values('dist')

    # Ensuring non-duplicate brands for variation
    unique_brands_df = temp_df.drop_duplicates(subset=['brand'], keep='first')

    top_3 = unique_brands_df.head(3)[['brand','product','name','hex']].copy()

    return top_3    