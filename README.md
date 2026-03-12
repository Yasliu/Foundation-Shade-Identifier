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

## The Problem We Solve

Current virtual shade-matching tools fail consumers across several critical technical pain points. Lumina was engineered to directly solve these issues:

* **Environmental & Technical Barriers:** Many users report that a shade suggested by an app in one lighting condition looks completely different in another [1][2]. 
  * *The Lumina Solution:* Implements a 5-point facial spatial averaging algorithm that calculates luminance to automatically reject specular glare (camera flash) and environmental shadows, isolating the true skin tone.
* **Skin Complexity & Undertone Confusion:** Existing tools often fail to account for complex skin conditions. Furthermore, a major pain point is the neglect of olive undertones, which can be particularly difficult to match as they often appear "muted" or "greenish" rather than just yellow or pink [3][4].
  * *The Lumina Solution:* Abandons subjective "warm, cool, or neutral" marketing labels. By converting RGB pixels to the CIELAB color space, the engine calculates the mathematical Delta-E distance of exact pigments. 
* **Data Bias & Product Waste:** Algorithms are frequently not trained on diverse skin tones [5]. This lack of accuracy leads to "shade hoarding," where consumers purchase multiple bottles of foundation in different shades, hoping one will work [6]. This results in financial loss and significant environmental waste.
  * *The Lumina Solution:* Matches user data against a robust matrix of over 5,000 physical products, outputting the top 3 mathematical matches across unique brands to guarantee diverse options and inclusivity.

---
**Market Research References:**
1. Reddit Community Discussions: [Questions about lighting when matching foundation.](https://www.reddit.com/r/OliveMUA/comments/7fqbp6/questions_about_lighting_when_matching_foundation/)
2. Reddit Community Discussions: [Why does my foundation look like the right match in some lighting but then totally wrong in others?](https://www.reddit.com/r/AustralianMakeup/comments/wnisgr/why_does_my_foundation_look_like_the_right_match/)
3. GirlGetGlamorous (2019): [How To Know Your Skin + Foundation Undertone.](https://girlgetglamorous.com/how-to-know-your-foundation-undertone-cool-warm-neutral-peach/)
4. Reddit Community Discussions: [Having trouble finding a foundation shade, cool makes me...](https://www.reddit.com/r/beauty/comments/1pxfsg3/having_trouble_finding_a_foundation_shade_cool/)
5. Very Good Light (2022): [The foundation shade range inclusivity problem.](https://www.verygoodlight.com/2022/08/29/inclusive-shade-ranges/)
6. Reddit Community Discussions: [Trying to find my foundation shade has led me to hoard 15...](https://www.reddit.com/r/MakeupRehab/comments/hww7xb/trying_to_find_my_foundation_shade_has_led_me_to/)

## The Architecture Pipeline

1. **Multi-Point Facial Mapping:** Utilizes Google's MediaPipe FaceLandmarker to detect facial topology and extract 5 specific micro-crops (Forehead, Nose Tip, Chin, Left Cheek, Right Cheek).
2. **Spatial Averaging & Outlier Rejection:** Calculates the perceived luminance of each extracted zone. It automatically rejects the absolute brightest (specular glare) and darkest (shadows) patches, mathematically averaging the remaining mid-tones to isolate the True Base skin color.
3. **Mathematical Translation:** Converts the True Base RGB values into the LAB color space, which maps directly to how the human eye actually perceives color differences.
4. **The Matching Engine:** Computes the Delta-E (CIE76) distance between the user's LAB coordinates and a pre-cleaned dataset of foundation shades, filtering for the top 3 matches strictly across unique cosmetic brands.

## Tech Stack

- **Frontend:** HTML, CSS, Vanilla JavaScript. Hosted on Vercel.
- **Backend & API:** FastAPI, Uvicorn, Python. Containerized via Docker and hosted on Hugging Face Spaces.
- **Computer Vision:** OpenCV, MediaPipe.
- **Data Engineering & Math:** Pandas, NumPy, Scikit-Image.

## Contributors

- **Yasin** - Lead Engineer: Backend API architecture, DevOps deployment, and computer vision integration
- **Anood** - Co-Creator & Product Strategy: Architected the 5-point spatial averaging algorithm, defined the cross-brand recommendation logic, served as the Domain Expert on cosmetic color theory, and led UI/UX testing

## Live Demo

**[https://lumina-foundation.vercel.app/]**
