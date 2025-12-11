from PIL import Image as PILImage
import os
import sys
import io
import base64
#from utils.helper import pil_to_texture, texture_to_pil, format_measurements
from pipeline.scan_frame_processor import USFrameProcess
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Patient(BaseModel):
    Name: str
    PatientDir: str

result: dict[str, list[str]] = {}

class AppLoader:
    def load_app(self, patient:Patient):
        patient_folder = patient.PatientDir
        """Handles patient folder selection and initializes components."""
        # Initialize paths
        image_folder = patient_folder#os.path.join(patient_folder, "Images")
        if not os.path.exists(image_folder):
            #Clock.schedule_once(lambda dt: self.show_error_popup("Image folder does not exist!"), 0)
            return
        xml_files = [f for f in os.listdir(patient_folder) if f.endswith('.xml')]
        self.xml_path = os.path.normpath(os.path.join(patient_folder, xml_files[0])) if xml_files else None

        if not os.path.exists(image_folder) or not self.xml_path:
            #Clock.schedule_once(lambda dt: self.show_error_popup("Path is incorrect!"), 0)
            return
        # Simulate progress while loading the pipeline
        #self.progress = 0
        #self.loading_steps = len(os.listdir(image_folder)) // 2
        #self.loading_interval = 0.4
        #Clock.schedule_interval(self.update_progress, self.loading_interval)

        # Load processing pipeline in a separate thread
        self.pipeline = USFrameProcess(image_folder)
       # patient.ClassifiedImages = self.pipeline.classified_images
        print("------*********Count:" ,len(self.pipeline.classified_images))       
        # return self.SaveClassifiedImages(patient_folder)
        return  self.SaveClassifiedImagesBase64()
    
    def SaveClassifiedImages(self,patient_folder):
        classifiedImagesDir = os.path.join(patient_folder, "ClassifiedImages")
        os.makedirs(classifiedImagesDir, exist_ok=True)

        for key, values in self.pipeline.classified_images.items():
            class_folder = os.path.join(classifiedImagesDir, key)
            os.makedirs(class_folder, exist_ok=True)
            for idx, item in enumerate(values):
                if hasattr(item, "save"):
                    img = item

                # If item is a tuple: assume first element is the image
                elif isinstance(item, tuple) and hasattr(item[0], "save"):
                    img = item[0]

                else:
                    raise TypeError(f"Cannot extract image from: {item}")

                img_path = os.path.join(class_folder, f"{idx}.png")
                img.save(img_path, "PNG")
                relative_path = os.path.relpath(img_path, start=patient_folder)
                url_path = f"/uploads/{relative_path.replace(os.sep, '/')}"
                #relative_path = os.path.relpath(img_path, start=os.getcwd())               
                #relative_path = "/" + relative_path.replace("\\", "/")
                result.setdefault(key, []).append(url_path)
        return result
        
    def SaveClassifiedImagesBase64(self):
        resultData = {}

        for key, values in self.pipeline.classified_images.items():
            resultData[key] = []

            for idx, item in enumerate(values):

                # Extract image (same logic as original)
                if hasattr(item, "save"):
                    img = item
                elif isinstance(item, tuple) and hasattr(item[0], "save"):
                    img = item[0]
                else:
                    raise TypeError(f"Cannot extract image from: {item}")

                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)

                base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

                # Add classification info
                resultData[key].append({
                    "index": idx,
                    "base64": base64_str
                })
        return resultData

Loader=AppLoader()
@app.get("/GetImageData/{category}/{imageidx}")
def get_image_data(category: str,imageIdx: int):
    images = self.pipeline.classified_images.get(key)
    if images and index < len(images):
        value = images[index]
    else:
        value = None
    
    return 

@app.post("/LoadPatientDir/")
def LoadPatientDir(patient:Patient):
    return Loader.load_app(patient)