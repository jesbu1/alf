import json
import numpy as np
import sys
#desired_changes= [
#    #"suite_karel_env.load.perception_noise_prob=0.0",
#    #"suite_karel_env.load.perception_noise_prob=0.25",
#    #"suite_karel_env.load.perception_noise_prob=0.5",
#    #"suite_karel_env.load.perception_noise_prob=0.75",
#    #"suite_karel_env.load.perception_noise_prob=1.0",
#    "suite_karel_env.load.action_noise_prob=0.0",
#    "suite_karel_env.load.action_noise_prob=0.25",
#    "suite_karel_env.load.action_noise_prob=0.5",
#    "suite_karel_env.load.action_noise_prob=0.75",
#    "suite_karel_env.load.action_noise_prob=1.0",
#]
baseline_filename="custom_eval_data_baseline_vals.json"
with open(baseline_filename) as f:
    baseline_json = json.load(f)
    baseline_map = {}
    for name in baseline_json.keys(): 
        baseline_map[name.split(" suite_karel_env")[0]] = baseline_json[name]
#baseline = desired_changes[0]

input_file = sys.argv[1]
final_results = {}
def process_reward(string):
    if type(string) == float:
        return string
    return float(string.split(" ")[0])
def process_sd(string):
    if type(string) == float:
        return string
    return float(string.split(" ")[1][1:-1])
with open(input_file, 'r') as f:
    results = json.load(f)
for key, reward in results.items():
    processed_key = key.split(" suite_karel_env")[0]
    if "_" in processed_key:
        processed_key = processed_key.split("_")[0]
    #query_key = processed_key + baseline
    #if query_key == key:
    #    continue
    baseline_reward = process_reward(baseline_map[processed_key])
    processed_reward = process_reward(reward)
    processed_std = process_sd(reward)
    # Calculate percent drop
    try:
        #if "stairClimber" in query_key and "harvester" in query_key:
        #    reward += 1.0
        #    baseline_reward += 1.0
        percent_drop = (processed_reward - baseline_reward)/baseline_reward * 100
    except ZeroDivisionError:
        percent_drop = float("nan")
    results[key] = f"{processed_reward:.2f} ({processed_std:.2f}) {percent_drop:.2f}%"

save_name = input_file.replace('.json', '_percent.json')
with open(save_name, "w") as f:
    json.dump(results, f, sort_keys=True, indent=4)