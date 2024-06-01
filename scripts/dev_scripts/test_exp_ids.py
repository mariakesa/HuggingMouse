from HuggingMouse.allen_api_utilities import AllenExperimentUtility

info_utility = AllenExperimentUtility()
exp_ids = info_utility.experiment_container_ids_imaged_areas(['VISpm'])
print(exp_ids)
exp_ids = info_utility.experiment_container_ids_imaged_areas(['VISam'])
print(exp_ids)
