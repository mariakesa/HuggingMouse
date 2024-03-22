import os  # nopep8

os.environ["HGMS_ALLEN_CACHE_PATH"] = "/media/maria/DATA/AllenData"  # nopep8
os.environ["HGMS_TRANSF_EMBEDDING_PATH"] = "/media/maria/DATA/BrainObservatoryProcessedData"  # nopep8
os.environ["HGMS_REGR_ANAL_PATH"] = "/media/maria/DATA/BrainObservatoryProcessedData/analysis"  # nopep8

from HuggingMouse.allen_api_utilities import AllenExperimentUtility
from HuggingMouse.visualizers import VisualizerDimReduction
from HuggingMouse.trial_averaged_data import MakeTrialAveragedData
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import ViTImageProcessor, ViTModel
from HuggingMouse.regressors import VisionEmbeddingToNeuronsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score


info_utility = AllenExperimentUtility()
# These functions will print some information to help select the
# experimental container id to work on.
info_utility.view_all_imaged_areas()
info_utility.visual_areas_info()
# Let's grab the first eperiment container id in the VISal area.
id = info_utility.experiment_container_ids_imaged_areas(['VISal'])[0]

# Let's visualize the data in that experimental container.
# Let's run PCA on trial averaged data. The visualize function
# will take in the experimental session (see Allen documentation)
# and stimulus.
dim_reduction_model = PCA(n_components=3)
trial_averaged_data = MakeTrialAveragedData().get_data(id)
visualizer = VisualizerDimReduction(dim_reduction_model)
visualizer.info()
visualizer.visualize(trial_averaged_data,
                     'three_session_A', 'natural_movie_one')
# We can easily perform the same operation with TSNE or any other
# model with sklearn-like API.
dim_reduction_model2 = TSNE(n_components=3)
visualizer2 = VisualizerDimReduction(dim_reduction_model2)
visualizer2.visualize(trial_averaged_data,
                      'three_session_A', 'natural_movie_one')

# Now let's run a regression model on ViT model embeddings.
regression_model = Ridge(10)
metrics = [r2_score, mean_squared_error, explained_variance_score]
# Let's use the most popular Vision Transformer model from HuggingFace
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
VisionEmbeddingToNeuronsRegressor(
    regression_model, metrics, model=model).execute(id)

# Any of these models and metrics can be replaced similarly
# to what we did in the visualization. Strategy design pattern magic at work!
