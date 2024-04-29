import uvicorn
import os
from fastapi import FastAPI
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from PIL import Image

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./Models/finalmodel.h5", compile=False)

# Load the labels
class_names = open("./Models/labelskeras.txt", "r").readlines()


app = FastAPI()

@app.get("/skillestim")
async def root(discription: str ="you left me empty" ):
    result={"skills":["skill1","skill2","skill3"]}
    return result
@app.get("/assigntalent")
async def root(projectid: str ="you left me empty" , duration: int = 7 ):
    sugestedtalent={"talentid":"765028745"}
    
    return  sugestedtalent

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get('PORT', 8000)), log_level="info")