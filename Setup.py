import cv2
import pandas as pd
import os
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.cluster import KMeans
from skimage.color import deltaE_cie76


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

        # we are going for the left cheek (index 117)
        landmark_indices = [10, 4, 152, 117, 346]

        patch_size = 10
        half_size = patch_size // 2

        all_patches = []

        for idx in landmark_indices:
            landmark = face[idx]

            center_x, center_y = int(landmark.x * w), int(landmark.y * h)

            y1 = max(0, center_y - half_size)
            y2 = min(h, center_y + half_size)
            x1 = max(0, center_x - half_size)
            x2 = min(w, center_x + half_size)

            patch_rgb = image[y1:y2 , x1:x2]

            all_patches.append(patch_rgb)

        # Convert all patches to LAB and find the dominant color
        five_centroids = []
        for patch in all_patches:
            lab_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
            centroid = np.mean(lab_patch, axis=(0,1))
            five_centroids.append(centroid)

        five_centroids = np.array(five_centroids)

        sorted_labs = five_centroids[five_centroids[:, 0].argsort()]
        trimmed_labs = sorted_labs[1:4]

        target_lab = np.mean(trimmed_labs, axis=0)
        best_matches = find_my_match(target_lab)

    else:
        print("Error: No face detected in the image.")
        return None
    

    return best_matches

def find_my_match(target_lab):
    foundation_matrix = df[['L_val', 'a_val', 'b_val']].values.astype(float)

    temp_df = df.copy()
    temp_df['dist'] = deltaE_cie76(target_lab, foundation_matrix)
    temp_df = temp_df.sort_values('dist')

    # Ensuring non-duplicate brands for variation
    unique_brands_df = temp_df.drop_duplicates(subset=['brand'], keep='first')

    top_3 = unique_brands_df.head(3)[['brand','product','name','hex']].copy()

    return top_3    