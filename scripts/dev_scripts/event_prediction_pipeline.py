
from dotenv import load_dotenv
import sys
sys.path.append("/home/maria/HuggingMouse/src/HuggingMouse/pipelines")
from pipeline_tasks import pipeline

load_dotenv()

pipe = pipeline("event-prediction")
