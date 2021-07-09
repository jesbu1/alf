import os
import sys
import json
import pandas as pd

input_files = [
	"custom_eval_data_action_noise_percent.json",
	"custom_eval_data_each_perception_noise_percent.json",
	"custom_eval_data_topOff_harvester_train_test_percent.json"
]

#method_ordering = {
#    "global": 0,
#    "recurrent": 1,
#}

method_rows = [
    "global",
    "recurrent"
]

def process_json(filename, df_dict):
    with open(filename) as f:
        input_data = json.load(f)
    for key, performance in input_data.items():
        split_key = key.split(" ")
        method = split_key[0][:-1]
        env_name = split_key[1]
        all_env_name = f"{env_name} {split_key[-1].replace('suite_karel_env.load.', '')}"
        if env_name not in df_dict:
            df_dict[all_env_name] = [performance]
        else:
            df_dict[all_env_name].append(performance)


df_dict = {}
for input_file in input_files:
    process_json(input_file, df_dict)

df = pd.DataFrame(df_dict, index=method_rows)
df.to_csv("all_eval_results.csv")