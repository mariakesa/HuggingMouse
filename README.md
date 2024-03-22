# HuggingMouse

![alt text](https://github.com/mariakesa/HuggingMouse/blob/main/logo_CC0_attention.jpg)

HuggingMouse is a data analysis library that combines AllenSDK (Allen Brain Observatory), HuggingFace Transformers, Sklearn
and Pandas. 

You can install the library via pip:

    pip install HuggingMouse

or clone this repo and run:

    pip install .

The documentation of the project is at: https://huggingmouse.readthedocs.io/en/latest/

[Allen Brain Observatory](https://allensdk.readthedocs.io/en/latest/brain_observatory.html) is a massive repository of calcium imaging recordings from the mouse visual cortex during presentation of various visual stimuli ([see also](https://github.com/AllenInstitute/brain_observatory_examples/blob/master/Visual%20Coding%202P%20Cheat%20Sheet%20October2018.pdf)). Currently, HuggingMouse supports running regression analyses on neural data while mice are viewing [three different natural movies](https://observatory.brain-map.org/visualcoding/stimulus/natural_movies). The code uses the Strategy design pattern to make it easy to run regression analyses with any HuggingFace vision model that can turn images into embeddings (currently the code extracts CLS tokens). Any regression model that has a sklearn like API will work. The result of the regression is measured by a metric function from sklearn metrics module and trial-by-trial analyses are stored in a Pandas dataframe, which can be further processed with statistical analyses. 

### Setting environment variables

In order to run the analyses three environment variables have to be set. These environmental variables are paths that are used to cache Allen data comming from the API, save HuggingFace model embeddings of experimental stimuli (natural movies)
and save the csv files that come from regression analyses. 

These three environment variables are: HGMS_ALLEN_CACHE_PATH, HGMS_TRANSF_EMBEDDING_PATH, HGMS_REGR_ANAL_PATH

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

### Selecting experimental container for analysis

### Visualizing trial averaged data

### Fitting regression models



