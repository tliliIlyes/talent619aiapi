import uvicorn
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware






# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./Models/kerasrefinedfinal.h5", compile=False)

# Load the labels
class_names = open("./Models/refinedkeraslabels.txt", "r").readlines()
def make_square(im, min_size=224, fill_color=(255, 255, 255)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

app = FastAPI()
origins = [
    "http://talent619.com",
    "https://talent619.com",
    "https://www.talent619.com",
    
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/skillestim")
async def root(discription: str ="you left me empty" ):
    result={"skills":["skill1","skill2","skill3"]}
    return result
@app.get("/assigntalent")
async def root(projectid: str ="you left me empty" , duration: int = 7 ):
    sugestedtalent={"talentid":"765028745"}
    
    return  sugestedtalent

@app.post('/verify_cin')
def upload(file: UploadFile = File()):
    try:        
        im = Image.open(file.file)
        # resize and save Image
        im = Image.fromarray(np.asarray(make_square(im)))
        im.save('out.png')
        # convert Image to array
        im = np.asarray(im)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
        imtosave=Image.fromarray(im)
        imtosave.save('out.png')

        # Make the image a numpy array and reshape it to the models input shape.
        im = np.asarray(im, dtype=np.float32).reshape(1, 224, 224, 3)
        im = (im / 127.5) - 1
        prediction = model.predict(im)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        return{"Class:":class_name[2:],"Confidence Score:":str(np.round(confidence_score * 100))[:-2]}
    
    except Exception as error:
        print(error)
        raise HTTPException(status_code=500, detail='Something went wrong')



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get('PORT', 8080)), log_level="info")