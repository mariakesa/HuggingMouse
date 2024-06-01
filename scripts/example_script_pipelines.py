from HuggingMouse.pipelines.pipeline_tasks import pipeline
from dotenv import load_dotenv
from HuggingMouse.pipelines.single_trial_fs import MovieSingleTrialRegressionAnalysis
from transformers import ViTModel, CLIPVisionModel
from sklearn.linear_model import Ridge

model = ViTModel.from_pretrained('google/vit-base-patch16-224')
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
regr_model = Ridge(10)

pipe = pipeline("neural-activity-prediction",
                model=model,
                regression_model=regr_model,
                single_trial_f=MovieSingleTrialRegressionAnalysis(),
                test_set_size=0.25)

# 511511001 is an experiment container ID.
pipe(511511001).dropna().scatter_movies().heatmap()
pipe(646959386).dropna().scatter_movies().heatmap()
