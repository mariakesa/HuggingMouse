from transformers import ViTImageProcessor, ViTModel
from regressors import VisionEmbeddingToNeuronsRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from allen_api_utilities import AllenExperimentUtility
from visualizers import Visualizer
from trial_averaged_data import MakeTrialAveragedData
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE        

if __name__=="__main__":
    # Check if the code is being executed in a documentation build environment
        # Skip API call in documentation build environment
    regression_model=Ridge(10)
    model = ViTModel.from_pretrained('google/vit-base-patch32-384')
    exps=AllenExperimentUtility()
    exps.view_all_imaged_areas()
    id=exps.experiment_container_ids_imaged_areas(['VISal'])[0]
    VisionEmbeddingToNeuronsRegressor(model,regression_model).execute(id)
    dim_reduction_model=PCA(n_components=3)
    trial_averaged_data=MakeTrialAveragedData().get_data(id)
    visualizer=Visualizer(dim_reduction_model)
    visualizer.info()
    visualizer.visualize(trial_averaged_data,'three_session_A','natural_movie_one')
    dim_reduction_model2=TSNE(n_components=3)
    visualizer2=Visualizer(dim_reduction_model2)
    visualizer2.visualize(trial_averaged_data,'three_session_A','natural_movie_one')

