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
        cheek_point = face[10]

        center_x = int(cheek_point.x * w)
        center_y = int(cheek_point.y * h)
    
        patch_size = 20
        half_size = patch_size // 2

        y1 = max(0, center_y - half_size)
        y2 = min(h, center_y + half_size)
        x1 = max(0, center_x - half_size)
        x2 = min(w, center_x + half_size)

        patch_rgb = image[y1:y2 , x1:x2]

        lab_img = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB)
        # km = kmeans
        km_lab = lab_img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        kmeans.fit(km_lab)

        labels=kmeans.labels_
        centroids=kmeans.cluster_centers_

        # This provides the 3 rows of LAB values from which it sorts and chooses the middle value

        target_lab = centroids[centroids[:, 0].argsort()][1]

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