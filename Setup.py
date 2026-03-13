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
        left_cheek_indices = [117, 118, 101, 121, 47, 126, 209]
        right_cheek_indices = [346, 347, 330, 350, 277, 355, 429]

        def get_cheek_lab(indices):
            polygon = np.array([[int(face[i].x * w), int(face[i].y * h)] for i in indices], dtype=np.int32)
            mask = np.zeros((h,w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            pixels = image[mask > 0]

            pixels_float = pixels.astype(np.float32) / 255.0
            pixel_reshaped = pixels_float.reshape(1, -1, 3)
            return cv2.cvtColor(pixel_reshaped, cv2.COLOR_RGB2LAB)[0]

        left_lab = get_cheek_lab(left_cheek_indices)
        right_lab = get_cheek_lab(right_cheek_indices)

        left_mean_L = left_lab[:, 0].mean()
        right_mean_L = right_lab[:, 0].mean()

        if right_mean_L > left_mean_L:
            best_cheek_lab = right_lab
        else:
            best_cheek_lab = left_lab

        # sort based on lightness
        sorted_indices = best_cheek_lab[:, 0].argsort()
        sorted_labs = best_cheek_lab[sorted_indices]

        num_pixels = len(sorted_labs)
        lower_bound = int(num_pixels * 0.40)
        upper_bound = int(num_pixels * 0.95)
        trimmed_labs = sorted_labs[lower_bound:upper_bound]

        # averaging the middle
        target_lab = trimmed_labs.mean(axis=0)
        target_lab[0] = np.clip(target_lab[0] + 0, 0, 100)
        target_lab[1] = target_lab[1] - 4.0
        target_lab[2] = target_lab[2] - 8.0

        best_matches = find_my_match(target_lab)

    else:
        print("Error: No face detected in the image.")
        return None
    

    return best_matches

def find_my_match(target_lab):
    foundation_matrix = df[['L_val', 'a_val', 'b_val']].values.astype(float)

    target_matrix = np.tile(target_lab, (len(foundation_matrix), 1))

    temp_df = df.copy()
    temp_df['dist'] = deltaE_ciede2000(target_matrix, foundation_matrix)
    temp_df = temp_df.sort_values('dist')

    # Ensuring non-duplicate brands for variation
    unique_brands_df = temp_df.drop_duplicates(subset=['brand'], keep='first')
    anchors = unique_brands_df.head(3)

    final_results = []
    for _, anchor in anchors.iterrows():
        # Ensuring same brand and product
        brand_shades = df[(df['brand'] == anchor['brand']) & (df['product'] == anchor['product'])]

        # sorting by lightness 
        lighter_shades = brand_shades[brand_shades['L_val'] > anchor['L_val']].sort_values('L_val')

        # finding the closest match
        if not lighter_shades.empty:
            lighter_match = lighter_shades.iloc[0]
            final_results.append(lighter_match[['brand', 'product', 'name', 'hex']])
        else:
            final_results.append(anchor[['brand', 'product', 'name', 'hex']])

    top_3 = pd.DataFrame(final_results).to_dict(orient='records')

    return top_3    