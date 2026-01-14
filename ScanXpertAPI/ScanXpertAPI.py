from PIL import Image as PILImage
import numpy as np
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

class MeasurementData(BaseModel):
    base64: str
    biometry: dict[str, str]

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
                # Convert to grayscale 2D array
                img_gray = img.convert("L")        # shape: (H, W)
                arr = np.array(img_gray,dtype=np.float64)

                # Add classification info
                # resultData[key].append({
                #     "index": idx,
                #     "base64": base64_str
                # })
                 # Convert to RGB numpy array
                # img_rgb = img.convert("RGB")
                # arr = np.array(img_rgb, dtype=np.uint8)  # (H, W, 3)

                # # RGB → 24-bit decimal
                # decimal_2d = (
                #     (arr[:, :, 0].astype(np.uint32) << 16) |
                #     (arr[:, :, 1].astype(np.uint32) << 8)  |
                #     arr[:, :, 2].astype(np.uint32)
                # ).astype(np.float64)
                resultData[key].append({
                "index": idx,
                "width": arr.shape[1],
                "height": arr.shape[0],
                "data": arr.tolist(),
                "base64": base64_str           
            })

        return resultData
    
    def GetBrainImagesBase64(self):
        resultData = {}
        try:
            for key, values in self.pipeline.brain_images.items():
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
                    # Convert to grayscale 2D array
                    img_gray = img.convert("L")        # shape: (H, W)
                    arr = np.array(img_gray,dtype=np.float64)

                    # Add classification info
                    # resultData[key].append({
                    #     "index": idx,
                    #     "base64": base64_str
                    # })
                    # Convert to RGB numpy array
                    # img_rgb = img.convert("RGB")
                    # arr = np.array(img_rgb, dtype=np.uint8)  # (H, W, 3)

                    # # RGB → 24-bit decimal
                    # decimal_2d = (
                    #     (arr[:, :, 0].astype(np.uint32) << 16) |
                    #     (arr[:, :, 1].astype(np.uint32) << 8)  |
                    #     arr[:, :, 2].astype(np.uint32)
                    # ).astype(np.float64)
                    resultData[key].append({
                    "index": idx,
                    "width": arr.shape[1],
                    "height": arr.shape[0],
                    "data": arr.tolist(),
                    "base64": base64_str           
                })
        except Exception as e:
            print("Error On GetBrainImages" , e)
            return
    
        return resultData
    
    def perform_measurement(self,category: str,imageIdx: int,thresh: int):
        """
        Performs automated fetal biometry measurements and displays results.

        This function takes pixel spacing and threshold values as inputs, processes the ultrasound image through a measurement pipeline,
        and outputs the image with a fitted ellipse along with calculated biometry parameters.
        
        Note:
            The function internally uses the measurement pipeline to process
            the image and calculate biometric measurements.
        """
        image = None
        if(category != "Femur"):
            image, _  = self.pipeline.brain_images[category][imageIdx]
        else:
            image, _  = self.pipeline.classified_images[category][imageIdx]

        if not self.pipeline:
            print("Coding Pipeline is not set properly! \n Solution: Ensure correct directory path is set!")
            return
        
        #plane = self.pipeline.find_plane(image)
        #if plane != "Trans-thalamic":
        #    print(f"The Selected Image is {plane} plane. \n Choose Trans-thalamic view image to obtain Brain measurement!")
        #    return
        
        ruler = 1 #self.auto_measurement_screen.get_ruler_scale()
        pixel_spacing = 1 #self.auto_measurement_screen.get_pixel_spacing()
        if pixel_spacing == "0":
            pixel_spacing = self.pipeline.find_pixel_spacing(image, float(ruler))
            #self.auto_measurement_screen.ids.pixel.text = str(pixel_spacing)
        try:
            biometryData, measured_image = self.pipeline.measure(image, pixel_spacing, thresh)

            buffer = io.BytesIO()
            measured_image.save(buffer, format="PNG")
            buffer.seek(0)

            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return MeasurementData( base64 = base64_str, biometry = biometryData )
        except TypeError:
            print("Not enough points to fit an ellipse! \n Solution: Try Reducing the threshold!!!")
            return
        except Exception as e:
             # General handler for any other unexpected errors
            print(f"error occurred on Measurment: {e}")
            return
        # if self.image_for_measurement.texture:
        #     image = texture_to_pil(self.image_for_measurement.texture)
        #     plane = self.pipeline.find_plane(image)
        #     if plane != "Trans-thalamic":
        #         self.show_error_popup(f"The Selected Image is {plane} plane. \n Choose Trans-thalamic view image to obtain Brain measurement!")
        #         return
            
        #     ruler = self.auto_measurement_screen.get_ruler_scale()
        #     pixel_spacing = self.auto_measurement_screen.get_pixel_spacing()
        #     if pixel_spacing == "0":
        #         pixel_spacing = self.pipeline.find_pixel_spacing(image, float(ruler))
        #         self.auto_measurement_screen.ids.pixel.text = str(pixel_spacing)
        #         if not pixel_spacing:
        #             self.show_error_popup("Error in finding the pixel spacing. \n Issue occurred because the algorithm cannot detect the Ruler. \n Solution: Use manual tool to depict 1cm spacing using pencil icon.")
            
        #     thresh = int(self.auto_measurement_screen.get_threshold())
        #     try:
        #         biometry, measured_image = self.pipeline.measure(image, pixel_spacing, thresh)

        #         report_screen = self.root.ids.content_manager.get_screen("Report")
        #         report_screen.ids.bpd.text = biometry["BPD"]
        #         report_screen.ids.ofd.text = biometry["OFD"]
        #         report_screen.ids.hc.text = biometry["HC"]
                
        #         if isinstance(measured_image, PILImage.Image):
        #             texture = pil_to_texture(measured_image)
        #             output_image = self.auto_measurement_screen.output_image()
        #             output_image.texture = texture
        #             self.auto_measurement_screen.add_output(texture)
                    
        #             if hasattr(self, 'confidence_image_instance'):
        #                 self.confidence_image_instance.bind_to_external_image(output_image)

        #         self.auto_measurement_screen.get_biometry_label().text = format_measurements(biometry)
        #     except TypeError:
        #         self.show_error_popup("Not enough points to fit an ellipse! \n Solution: Try Reducing the threshold!!!")
        # else:
        #     self.show_error_popup("Select a Valid Image to perform measurement")

Loader=AppLoader()
@app.get("/AutoMeasurement/{category}/{imageIdx}/{thresh}",response_model=MeasurementData)
def AutoMeasurement(category: str,imageIdx: int,thresh: int):    
    return Loader.perform_measurement(category,imageIdx,thresh)

@app.get("/GetBrainImages/")
def GetBrainImages():    
    return Loader.GetBrainImagesBase64()

@app.post("/LoadPatientDir/")
def LoadPatientDir(patient:Patient):
    return Loader.load_app(patient)