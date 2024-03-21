import requests
from bs4 import BeautifulSoup
from huggingface_hub import HfApi, ModelFilter
import json
import time

'''
This script should run for about an hour since there are about 10000 image classification models 
on huggingface. 
'''
# Function to retrieve the "Downloads last month" statistic for a given model


def get_downloads_last_month(model_name):
    url = f"https://huggingface.co/{model_name}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # Locate the element containing the "Downloads last month" statistic
        downloads_last_month_element = soup.find(
            "dt", text="Downloads last month")
        if downloads_last_month_element:
            # Extract the next sibling (dd element) containing the actual statistic
            downloads_last_month_stat = downloads_last_month_element.find_next_sibling(
                "dd").text.strip()
            # Remove commas from the statistic
            downloads_last_month_stat = downloads_last_month_stat.replace(
                ",", "")
            return downloads_last_month_stat
    return 0


start = time.time()
# Retrieve metadata for all image recognition models
api = HfApi()
filter = ModelFilter(task="image-classification")
models = api.list_models(filter=filter)

# Collect model names
model_names = [model.modelId for model in models]

i = 0
# Get the "Downloads last month" statistic for each model
downloads_last_month_stats = {}
for model_name in model_names:
    print(i)
    downloads_last_month_stat = get_downloads_last_month(model_name)
    downloads_last_month_stats[model_name] = downloads_last_month_stat
    i += 1

# Sort models based on "Downloads last month" statistic
sorted_models = sorted(downloads_last_month_stats.items(),
                       key=lambda x: int(x[1]), reverse=True)

# Select top 10 models
top_10_models = sorted_models[:10]

# Print the top 10 models and their "Downloads last month" statistics
for model_name, downloads_last_month_stat in top_10_models:
    print(
        f"Model: {model_name}, Downloads last month: {downloads_last_month_stat}")

# Convert sorted models to dictionary
sorted_downloads_last_month_stats = dict(sorted_models)

# Save the sorted models and their "Downloads last month" statistics to a JSON file
output_file = "sorted_models_downloads.json"
with open(output_file, "w") as f:
    json.dump(sorted_downloads_last_month_stats, f, indent=4)

end = time.time()
print('Time taken to compile model downloads: ', end-start)
