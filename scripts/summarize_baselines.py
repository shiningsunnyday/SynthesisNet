import yaml
import re
import os
from collections import defaultdict
import numpy as np
import heapq
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
import argparse
from tdc import Oracle

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


def compute_scores(f, limit=10000):
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
    sa_oracle = Oracle(name="SA")
    for smi in res:
        sa = sa_oracle(smi)
        res[smi] = res[smi]+[sa]
    K = min(len(res), limit)
    res = {r: res[r] for r in list(res) if res[r][1] <= limit}
    scores = {'K': K}
    scores['top_1'] = list(res.values())[0][0]
    scores['top_1_sa'] = list(res.values())[0][-1]
    scores['top_10'] = np.mean(list(res.values())[:10], axis=0)[0]
    scores['top_10_sa'] = np.mean(list(res.values())[:10], axis=0)[-1]
    scores['top_100'] = np.mean(list(res.values())[:100], axis=0)[0]
    scores['top_100_sa'] = np.mean(list(res.values())[:100], axis=0)[-1]
    scores['second'] = list(res.values())[1][0]
    scores['second_sa'] = list(res.values())[1][-1]
    scores['third'] = list(res.values())[2][0]
    scores['third_sa'] = list(res.values())[2][-1]
    running = list(map(lambda x: x[0], sorted(res.values(), key=lambda x:x[1])))
    scores['top_1_auc'] = top_k_auc(running, 1, K)
    scores['top_10_auc'] = top_k_auc(running, 10, K)
    scores['top_100_auc'] = top_k_auc(running, 100, K)
    print("done")
    return scores


def compute_metrics(data_dir, baseline, seed, metric):    
    path = os.path.join(data_dir, f'results_{baseline}_qed_{seed}.yaml')
    out_path = os.path.join(data_dir, f'results_{baseline}_{metric}_{seed}.yaml')
    data = yaml.safe_load(open(path))
    result = {}
    if metric == 'SA':
        oracle = Oracle(metric)
        res = [oracle(smi) for smi in data]    
    for smi, score in zip(data, res):
        result[smi] = [score, data[smi][1]]
    yaml.dump(result, open(out_path, 'w'), default_flow_style=False)
    print(out_path)
    

def main(args):
    data_dir = args.data_dir
    # use synnet to extract supported metrics
    metrics = []
    baselines = defaultdict(list)
    for f in os.listdir(data_dir): 
        match = re.match('results_synnet_(\w+)_0.yaml', f)
        if match:
            metrics.append(match.groups()[0])
        match = re.match(f'results_(\w+)_qed_(\d).yaml', f)
        if match:
            baseline, seed = match.groups()
            baselines[baseline].append(seed)        

    # compute remaining metrics
    pargs = [(data_dir, baseline, seed, metric) for baseline in baselines \
             for seed in baselines[baseline] \
             for metric in args.metrics \
             if not os.path.exists(os.path.join(data_dir, f'results_{baseline}_{metric}_{seed}.yaml'))]
             
    if args.ncpu:
        with mp.Pool(args.ncpu) as p:
            p.starmap(compute_metrics, pargs)
    else:
        [compute_metrics(*parg) for parg in pargs]
        
    metrics += args.metrics
    print(f"Supporting metrics:", metrics)

    # now loop over all files
    res = defaultdict(lambda: [[] for _ in metrics])
    pargs = []    
    for f in tqdm(os.listdir(data_dir)):
        for i, metric in enumerate(metrics):
            match = re.match(f'results_(\w+)_{metric}_(\d).yaml', f)
            if match:
                baseline, seed = match.groups()
                arg = os.path.join(data_dir, f)
                pargs.append((arg, i, baseline, seed))                

    if args.ncpu:
        with mp.Pool(args.ncpu) as p:
            all_scores = p.map(compute_scores, tqdm([a[0] for a in pargs]))
    else:
        all_scores = [compute_scores(a[0]) for a in pargs]
    for (_, i, baseline, seed), scores in zip(pargs, all_scores):        
        scores['seed'] = int(seed)
        res[baseline][i].append(scores)

    df = pd.DataFrame(index=list(res))
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
    print(os.path.abspath(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default='data/pmo/')
    parser.add_argument("--ncpu", type=int, default=0)
    parser.add_argument("--metrics", nargs='+', choices=['SA'], help="Additional metrics to compute for each file", default=[])
    args = parser.parse_args()
    main(args)
    