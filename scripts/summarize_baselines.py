import yaml
import re
import os
from collections import defaultdict
import numpy as np
import heapq
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp

def top_k_auc(arr, k, max_oracle_calls=10000):
    min_heap = []
    result = []
    for score in arr:
        heapq.heappush(min_heap, score)
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # Remove the smallest element if more than 10 elements
        result.append(np.mean(min_heap))  # Sort in descending order for top scores
    result.append((max_oracle_calls-len(result))*np.mean(min_heap))
    return np.mean(result)


def compute_scores(f):
    """
    We compute the following metrics from PMO [1]:
        Top 1: Score of best mol
        Top 10: Avg score of top 10
        Top 100: Avg score of top 100
        First: Score of best mol
        Second: Score of second best mol
        Third: Score of third best mol
        Top 1 AUC: Sum_(k=1,...,K) Score of best mol found by iter k
        Top 10 AUC: Sum_(k=1,...,K) Avg top 10 score of mols found by iter k
        Top 100 AUC: Sum_(k=1,...,K) Avg top 100 score of mols found by iter k
    
    [1] Gao, Wenhao, et al. "Sample efficiency matters: a benchmark for practical molecular optimization." Advances in neural information processing systems 35 (2022): 21342-21357.
    
    Note: 
    PMO computes Top k AUC in a slightly peculiar way, by tracking a running average of the avg top 10 score.
    Our implementation is more straight-forward, but there may be slight deviation in results.
    https://github.com/wenhao-gao/mol_opt/blob/2da631be85af8d10a2bb43f2de76a03171166190/main/optimizer.py#L30
    """
    res = yaml.safe_load(open(f))
    K = min(len(res), 10000)
    res = {r: res[r] for r in list(res)[:K]}
    scores = {'K': K}
    scores['top_1'] = list(res.values())[0][0]
    scores['top_10'] = np.mean(list(res.values())[:10], axis=0)[0]
    scores['top_100'] = np.mean(list(res.values())[:100], axis=0)[0]
    scores['first'] = scores['top_1']
    scores['second'] = list(res.values())[1][0]
    scores['third'] = list(res.values())[2][0]
    running = list(map(lambda x: x[0], sorted(res.values(), key=lambda x:x[1])))
    scores['top_1_auc'] = top_k_auc(running, 1, K)
    scores['top_10_auc'] = top_k_auc(running, 10, K)
    scores['top_100_auc'] = top_k_auc(running, 100, K)
    return scores
    

data_dir = 'data/pmo/'
# use synnet to extract supported metrics
metrics = []
for f in os.listdir(data_dir):
    if 'synnet' not in f:
        continue
    if 'similarity' not in f:
        continue    
    match = re.match('results_synnet_(\w+)_0.yaml', f)
    if match:
        metrics.append(match.groups()[0])

print(f"Supporting metrics:", metrics)

# now loop over all files
res = defaultdict(lambda: [[] for _ in metrics])
args = []
for f in tqdm(os.listdir(data_dir)):
    for i, metric in enumerate(metrics):
        match = re.match(f'results_(\w+)_{metric}_(\d).yaml', f)
        if match:
            baseline, seed = match.groups()
            arg = os.path.join(data_dir, f)
            args.append((arg, i, baseline, seed))
        
with mp.Pool(5) as p:
    all_scores = p.map(compute_scores, tqdm([a[0] for a in args]))
for (_, i, baseline, seed), scores in zip(args, all_scores):        
    scores['seed'] = int(seed)
    res[baseline][i].append(scores)

df = pd.DataFrame(index=list(res), columns=metrics)
for baseline in res:
    for i in range(len(metrics)):
        df_i = pd.DataFrame(res[baseline][i])
        if len(df_i) == 0:
            continue
        res[baseline][i] = dict(df_i.mean(axis=0)) # aggregate over seeds    
    for i, metric in enumerate(metrics):
        for col in res[baseline][i]:
            df.loc[baseline, f"{metric}_{col}"] = res[baseline][i][col]
path = os.path.join(data_dir, "results.csv")
df.to_csv(path)
print(path)
