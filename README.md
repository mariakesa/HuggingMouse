# HuggingMouse

![alt text](https://github.com/mariakesa/HuggingMouse/blob/main/logo_CC0_attention.jpg)

HuggingMouse is a data analysis library that combines AllenSDK (Allen Brain Observatory), HuggingFace Transformers, Sklearn
and Pandas. 

    pip install HuggingMouse

[Allen Brain Observatory](https://allensdk.readthedocs.io/en/latest/brain_observatory.html) is a massive repository of calcium imaging recordings from the mouse visual cortex during presentation of various visual stimuli ([see also](https://github.com/AllenInstitute/brain_observatory_examples/blob/master/Visual%20Coding%202P%20Cheat%20Sheet%20October2018.pdf)). Currently, HuggingMouse supports running regression analyses on neural data while mice are viewing [three different natural movies](https://observatory.brain-map.org/visualcoding/stimulus/natural_movies). The code uses the Strategy Design Pattern to make it easy to run regression analyses with any HuggingFace vision model that can turn images into embeddings (currently the code extracts CLS tokens). Any regression model that has a sklearn like API will work. The result of the regression is measured by a metric function from sklearn metrics module and trial-by-trial analyses are stored in a Pandas dataframe, which can be further processed with statistical analyses. 
