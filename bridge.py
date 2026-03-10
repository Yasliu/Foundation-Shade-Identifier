import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from Setup import find_comparison

app = FastAPI()

@app.get("/")
def read_root():
    return {"Status": "Server is awake!"}

@app.post("/analyze-shade")
async def analyze_shade(file: UploadFile = File(...)):
    contents = await file.read()

    # bytes to numpy
    nparr = np.frombuffer(contents, np.uint8)
    # decode image into opencv (this turns it into BGR)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # my function takes the RGB values
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = find_comparison(img_rgb)

    final_data = result.to_dict(orient="records")

    return {"results": final_data}

