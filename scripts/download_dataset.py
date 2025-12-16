import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("learnai-t0pyb").project("box_detection_tracking")
version = project.version(2)
dataset = version.download("yolov11")
                