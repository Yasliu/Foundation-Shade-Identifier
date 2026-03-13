import cv2
import numpy as np
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from Setup import find_comparison
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Status": "Server is awake!"}

@app.post("/analyze-shade")
async def analyze_shade(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # bytes to numpy
        nparr = np.frombuffer(contents, np.uint8)
        # decode image into opencv (this turns it into BGR)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Could not decode image. Ensure you are sending a valid JPEG/PNG.", "traceback": ""})

        # my function takes the RGB values
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = find_comparison(img_rgb)

        return result

    except Exception as e:
        tb = traceback.format_exc()
        print("=== LUMINA 500 ERROR ===")
        print(tb)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb}
        )
