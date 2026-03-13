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

Instead of relying on basic Hex/RGB comparisons which fail to accurately represent human skin, this application leverages the CIELAB color space, advanced CIE2000 Delta-E algorithms, and dynamic spatial averaging to match extracted skin tones against a database of over 5,000 real-world cosmetic products.

## The Problem We Solve

Current virtual shade-matching tools fail consumers across several critical technical pain points. Lumina was engineered to directly solve these issues:

* **Environmental & Technical Barriers:** Many users report that a shade suggested by an app in one lighting condition looks completely different in another [1][2]. 
  * *The Lumina Solution:* Implements dynamic lighting compensation by extracting multi-point polygons from both cheeks, automatically discarding the shadowed side of the face. It then applies an asymmetric statistical trim to reject specular glare (camera flash) and micro-shadows, mathematically isolating the true, well-lit base skin tone.
* **Skin Complexity & Undertone Confusion:** Existing tools often fail to account for complex skin conditions and neglect olive undertones, which can be particularly difficult to match as they often appear "muted" or "greenish" rather than strictly yellow or pink [3][4].
  * *The Lumina Solution:* Abandons subjective "warm, cool, or neutral" marketing labels. By converting pixels to the CIELAB color space, the engine calculates the $\Delta E_{00}$ (CIE2000) distance, an advanced perceptual algorithm that explicitly penalizes hue/undertone mismatches heavier than lightness mismatches. 
* **Data Bias & Product Waste:** Algorithms are frequently not trained on diverse skin tones [5]. This lack of accuracy leads to "shade hoarding," where consumers purchase multiple bottles of foundation in different shades, hoping one will work [6]. This results in financial loss and significant environmental waste.
  * *The Lumina Solution:* Matches user data against a robust matrix of over 5,000 physical products, outputting the top 3 mathematical matches strictly across unique brands to guarantee diverse formulas and inclusivity.

---
**Market Research References:**
1. Reddit Community Discussions: [Questions about lighting when matching foundation.](https://www.reddit.com/r/OliveMUA/comments/7fqbp6/questions_about_lighting_when_matching_foundation/)
2. Reddit Community Discussions: [Why does my foundation look like the right match in some lighting but then totally wrong in others?](https://www.reddit.com/r/AustralianMakeup/comments/wnisgr/why_does_my_foundation_look_like_the_right_match/)
3. GirlGetGlamorous (2019): [How To Know Your Skin + Foundation Undertone.](https://girlgetglamorous.com/how-to-know-your-foundation-undertone-cool-warm-neutral-peach/)
4. Reddit Community Discussions: [Having trouble finding a foundation shade, cool makes me...](https://www.reddit.com/r/beauty/comments/1pxfsg3/having_trouble_finding_a_foundation_shade_cool/)
5. Very Good Light (2022): [The foundation shade range inclusivity problem.](https://www.verygoodlight.com/2022/08/29/inclusive-shade-ranges/)
6. Reddit Community Discussions: [Trying to find my foundation shade has led me to hoard 15...](https://www.reddit.com/r/MakeupRehab/comments/hww7xb/trying_to_find_my_foundation_shade_has_led_me_to/)

## The Architecture Pipeline

1. **Dynamic Facial Mapping:** Utilizes Google's MediaPipe FaceLandmarker to detect facial topology and extract precise, multi-point polygon regions of both the left and right cheeks.
2. **Lighting Compensation & Asymmetric Trim:** Evaluates the perceived luminance of both extracted regions, automatically discarding the polygon located in shadow. The system then flattens the well-lit pixels and applies an asymmetric statistical trim (dropping the bottom 40% of shadow/contour pixels and the top 5% of specular glare) to isolate the pure True Base skin color.
3. **Mathematical Translation:** Converts the extracted, trimmed RGB array into the CIELAB color space, which maps directly to how the human eye actually perceives color differences.
4. **The Matching Engine:** Computes the highly optimized $\Delta E_{00}$ (CIE2000) distance between the user's isolated LAB coordinates and a pre-cleaned dataset of foundation shades, filtering for the top 3 perceptually perfect matches across unique cosmetic brands.

## Tech Stack

- **Frontend:** HTML, CSS, Vanilla JavaScript. Hosted on Vercel.
- **Backend & API:** FastAPI, Uvicorn, Python. Containerized via Docker and hosted on Hugging Face Spaces.
- **Computer Vision:** OpenCV, MediaPipe.
- **Data Engineering & Math:** Pandas, NumPy, Scikit-Image.

## Contributors

- **Yasin** - Lead Engineer: Backend API architecture, DevOps deployment, dynamic computer vision integration, and statistical mathematics pipeline.
- **Anood** - Lead Designer & Co-Creator: Architected the Single Page Application (SPA) UI/UX, led frontend web development, and served as the Domain Expert on cosmetic color theory and user testing.

## Live Demo

**[https://lumina-foundation.vercel.app/]**

**[https://lumina-foundation.vercel.app/]**
