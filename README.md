# Foundation Shade Identifier

An end-to-end computer vision and machine learning pipeline designed to find the mathematically perfect foundation makeup match from a single selfie. 

Instead of relying on basic Hex/RGB comparisons which fail to accurately represent human skin, this application leverages the LAB color space and Delta-E algorithms to match extracted skin tones against a database of over 5,000 real-world cosmetic products.

## The Architecture Pipeline
1. **Facial Mapping:** Utilizes Google's MediaPipe FaceLandmarker to detect facial topology and specifically crop a 20x20 pixel patch from the user's cheek.
2. **Pigment Extraction:** Runs K-Means clustering (k=3) on the cropped patch to separate highlights, shadows, and the True Base skin tone.
3. **Mathematical Translation:** Converts the True Base RGB values into LAB color space, which maps directly to how the human eye perceives color differences.
4. **The Matching Engine:** Computes the Delta-E (CIE76) distance between the user's LAB coordinates and a pre-cleaned dataset of foundation shades, filtering for the top 3 matches across unique brands.

## Tech Stack
* **Frontend:** Streamlit
* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** Scikit-Learn (K-Means clustering)
* **Math & Distances:** Scikit-Image (Delta-E CIE76)
* **Data Engineering:** Pandas, NumPy

## Live Demo
**[https://foundation-shade-identifier.streamlit.app/]**
