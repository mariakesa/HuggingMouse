# HuggingMouse

![alt text](https://github.com/mariakesa/HuggingMouse/blob/main/logo_CC0_attention.jpg)

HuggingMouse is a data analysis library that combines AllenSDK (Allen Brain Observatory), HuggingFace Transformers, Sklearn
and Pandas. 

You can install the library via pip:

    pip install HuggingMouse

or clone this repo and run:

    pip install .

The documentation of the project is at: https://huggingmouse.readthedocs.io/en/latest/

[Allen Brain Observatory](https://allensdk.readthedocs.io/en/latest/brain_observatory.html) is a massive repository of calcium imaging and NeuroPixel probe recordings from the mouse visual cortex during presentation of various visual stimuli ([see also](https://github.com/AllenInstitute/brain_observatory_examples/blob/master/Visual%20Coding%202P%20Cheat%20Sheet%20October2018.pdf)). Currently, HuggingMouse supports running regression analyses on neural data while mice are viewing [three different natural movies](https://observatory.brain-map.org/visualcoding/stimulus/natural_movies). 

Since version 0.1.0 HuggingMouse supports HuggingFace like pipelines. 

The following sections go through the code step by step. The scripts are avaiable here: 
https://github.com/mariakesa/HuggingMouse/blob/main/scripts/example_script_pipelines.py
https://github.com/mariakesa/HuggingMouse/blob/main/scripts/example_script.py

### Setting environment variables

To run the analyses three environment variables have to be set. These environmental variables are paths that are used to cache Allen data coming from the API, save HuggingFace model embeddings of experimental stimuli (natural movies)
and save the CSV files that come from regression analyses. 

These three environment variables are HGMS_ALLEN_CACHE_PATH, HGMS_TRANSF_EMBEDDING_PATH, HGMS_REGR_ANAL_PATH

There are two ways to set these variables. First, you can use the os module in Python:

    import os

    os.environ["HGMS_ALLEN_CACHE_PATH"] = ...Allen API cache path as string...
    os.environ["HGMS_TRANSF_EMBEDDING_PATH"] = ...stimulus embedding path as string...
    os.environ["HGMS_REGR_ANAL_PATH"] = ...path to store model metrics csv's as string...

Alternatively, you can save the same paths in a .env file like this:

    HGMS_ALLEN_CACHE_PATH=...path... 
    HGMS_TRANSF_EMBEDDING_PATH=...path...
    HGMS_REGR_ANAL_PATH=...path...

and call the dotenv library to read in these environment variables in your script that uses HuggingMouse:

    from dotenv import load_dotenv

    load_dotenv('.env')

Note that the dotenv library has to be installed for this to work.

### Fitting regression models

The new HuggingMouse pipelines functionality allows fitting custom regression models on all of the experimental trials
for an animal with just a few lines of code. This pipeline is highly customizable and you can write your own function 
for processing a single trial, and the pipeline applies it to all the trials. Train and test splits can also be customized.
The pipeline also supports method chaining to plot the variance explained for each trial in heatmaps and scatterplots. 

    from HuggingMouse.pipelines.pipeline_tasks import pipeline
    from dotenv import load_dotenv
    from HuggingMouse.pipelines.single_trial_fs import MovieSingleTrialRegressionAnalysis
    from transformers import ViTModel
    from sklearn.linear_model import Ridge

    model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    regr_model = Ridge(10)
    
    pipe = pipeline("neural-activity-prediction",
                    model=model,
                    regression_model=regr_model,
                    single_trial_f=MovieSingleTrialRegressionAnalysis(),
                    test_set_size=0.25)
    #511511001 and 646959386 are experiment container ID's.  
    pipe(511511001).dropna().scatter_movies().heatmap()
    pipe(646959386).dropna().scatter_movies().heatmap()

### Selecting experimental container for analysis

HuggingMouse has a helper class for choosing the experimental container (one transgenic animal).

    from HuggingMouse.allen_api_utilities import AllenExperimentUtility

    info_utility = AllenExperimentUtility()
    #These functions will print some information to help select the
    #experimental container id to work on. 
    info_utility.view_all_imaged_areas()
    info_utility.visual_areas_info()
    #Let's grab the first eperiment container id in the VISal area. 
    id = info_utility.experiment_container_ids_imaged_areas(['VISal'])[0]

### Visualizing trial averaged data

You can use any sklearn decomposition (PCA, NMF) or manifold method (TSNE, SpectralEmbedding) or
your own custom model with a fit_transform method to visualize the patterns in the neural data. 
This is currently possible with trial-averaged data. The visualize function takes the session and 
the stimulus that uniquely identify a sequence of trials and embeds the data
using the provided model. 

    from HuggingMouse.visualizers import VisualizerDimReduction
    from HuggingMouse.trial_averaged_data import MakeTrialAveragedData
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    dim_reduction_model = PCA(n_components=3)
    trial_averaged_data = MakeTrialAveragedData().get_data(id)
    visualizer = VisualizerDimReduction(dim_reduction_model)
    visualizer.info()
    visualizer.visualize(trial_averaged_data,
                        'three_session_A', 'natural_movie_one')

    dim_reduction_model2 = TSNE(n_components=3)
    visualizer2 = VisualizerDimReduction(dim_reduction_model2)
    visualizer2.visualize(trial_averaged_data,
                        'three_session_A', 'natural_movie_one')

### Enjoy!

This package is meant as a gateway for exploring the mysteries of the brain and vision-- this is what calcium imaging raw data looks like:

![Calcium Imaging](https://github.com/mariakesa/HuggingMouse/blob/main/calcium_movie.gif)

The gif is courtesy of [Andermann lab](https://www.andermannlab.com/), used with permission.



