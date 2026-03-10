---
title: Lumina Engine
emoji: 💄
colorFrom: gray
colorTo: gray
sdk: docker
pinned: false
---

# Lumina: Foundation Shade Identifier

An end-to-end computer vision and machine learning pipeline designed to find the mathematically perfect foundation makeup match from a single selfie.

Instead of relying on basic Hex/RGB comparisons which fail to accurately represent human skin, this application leverages the LAB color space, Delta-E algorithms, and dynamic spatial averaging to match extracted skin tones against a database of over 5,000 real-world cosmetic products.

## The Architecture Pipeline

1. **Multi-Point Facial Mapping:** Utilizes Google's MediaPipe FaceLandmarker to detect facial topology and extract 5 specific micro-crops (Forehead, Nose Tip, Chin, Left Cheek, Right Cheek).
2. **Spatial Averaging & Outlier Rejection:** Calculates the perceived luminance of each extracted zone. It automatically rejects the absolute brightest (specular glare) and darkest (shadows) patches, mathematically averaging the remaining mid-tones to isolate the True Base skin color.
3. **Mathematical Translation:** Converts the True Base RGB values into the LAB color space, which maps directly to how the human eye actually perceives color differences.
4. **The Matching Engine:** Computes the Delta-E (CIE76) distance between the user's LAB coordinates and a pre-cleaned dataset of foundation shades, filtering for the top 3 matches strictly across unique cosmetic brands.

## Tech Stack

- **Frontend:** HTML, CSS, Vanilla JavaScript (UI/UX codebase fully generated and handled by AI Agent). Hosted on Vercel.
- **Backend & API:** FastAPI, Uvicorn, Python. Containerized via Docker and hosted on Hugging Face Spaces.
- **Computer Vision:** OpenCV, MediaPipe.
- **Data Engineering & Math:** Pandas, NumPy, Scikit-Image.

## Contributors

- **Yiso** - Lead Engineer: Backend API architecture, DevOps deployment, and computer vision integration
- **Anood** - Co-Creator & Product Strategy: Architected the 5-point spatial averaging algorithm, defined the cross-brand recommendation logic, served as the Domain Expert on cosmetic color theory, and led UI/UX testing

## Live Demo

**[https://lumina-foundation.vercel.app/]**
