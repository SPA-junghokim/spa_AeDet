import json
import sys

try:
    json_path = sys.argv[1]
except:
    json_path = '/media/ddd/git/spa_AeDet/outputs/overfit/metrics_summary.json'
with open(json_path) as file:
    # Load the JSON data
    data = json.load(file)
    
print("mAP:", round(data['mean_ap'], 4))
print("mATE:", round(data['tp_errors']['trans_err'], 4))
print("mASE:", round(data['tp_errors']['scale_err'], 4))
print("mAOE:", round(data['tp_errors']['orient_err'], 4))
print("mAVE:", round(data['tp_errors']['vel_err'], 4))
print("mAAE:", round(data['tp_errors']['attr_err'], 4))
print("NDS:", round(data['nd_score'], 4))
