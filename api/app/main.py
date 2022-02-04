 # Main.py
    
import io
import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from typing import List


from enum import Enum

class FileType(Enum):
    audio: str = 'AUDIO'
    
BASE_PATH = '/Users/johanvekeman/Documents/Datasets/Home_Assignment/Audio'
AUDIO_PATH = os.path.join(BASE_PATH, FileType.audio.value)
os.makedirs(AUDIO_PATH, exist_ok=True)
  
app = FastAPI()
   
# Load registered model
  
loaded_model=tf.keras.models.load_model('my_model.h5')
  
# Predict if a defect occurred or not, based on an sound file of 1s in length
   
def predict_audio_file(file):
    audio, sr = librosa.load(file, 22000)
    Spectrogram=librosa.feature.melspectrogram(audio)
    # The input shape for a CNN in Tensorflow should be in the format (batch, height, width, channels)
    Spectrogram=Spectrogram.reshape(1, Spectrograms[0].shape[0],Spectrograms[0].shape[1], 1)
    prediction = model.predict(Spectrogram)
    prediction = np.argmax(model.predict(Spectrogram), axis=-1)
    index=prediction[0]
    labels = ['No defect is detected', 'A defect is detected']
    return {'Prediction': labels[index]}
  
app.add_middleware
    CORSMiddleware,
    allow_origins=["*"]
    allow_credentials=True
    allow_methods=["*"]
    allow_headers=["*"]
 
@app.get("/")
def root():
    return {"message":"Hello World"}
  
@app.get("/audio", response_model=List[str])
def getAllAudio():
    return os.listdir(AUDIO_PATH)
  
@app.get("/audio/{filename}", response_class=FileResponse)
def getAudioById(filename: str):
    try:
        file = os.path.join(AUDIO_PATH, filename)
        return predict_audio_file(file)
    except Exception as e:
        return HTTPException(500, f'Something went wrong while trying to return file {filename}')
  
@app.post("/upload")
def uploadFile(file: UploadFile = File(...), type: FileType = FileType.default):
    print(f"Upload type: {type}")
    print(f"Upload type: {type.value}")
    fileLocation = os.path.join(BASE_PATH, type.value, file.filename)
    try:
        open(fileLocation, 'wb').write(file.file.read())
    except Exception as e:
        print(e)
        return HTTPException(500, f'Something went wrong while trying to upload file {file.filename}')
   
    return Response(f'Uploaded file to {fileLocation}', 200)