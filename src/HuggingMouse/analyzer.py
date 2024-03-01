from transformers import ViTImageProcessor, ViTModel
from regressors import VisionEmbeddingToNeuronsRegressor
from sklearn.linear_model import Ridge
from allen_api_utilities import AllenExperimentUtility
import os

if __name__=="__main__":
    # Check if the code is being executed in a documentation build environment
        # Skip API call in documentation build environment
    regression_model=Ridge(10)
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    exps=AllenExperimentUtility()
    exps.view_all_imaged_areas()
    id=exps.experiment_container_ids_imaged_areas(['VISal'])[0]
    VisionEmbeddingToNeuronsRegressor(model,regression_model).execute(id)
