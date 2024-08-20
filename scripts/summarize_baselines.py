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

LIMIT = 10000
category = {
    'synnet': 'synthesis',
    'pasithea': 'string',
    'dog_ae': 'synthesis',
    'smiles_vae_bo': 'string',
    'jt_vae_bo': 'graph',
    'moldqn': 'graph',
    'mars': 'graph',
    'selfies_lstm_hc': 'string',
    'gp_bo': 'graph',
    'smiles_ga': 'string',
    'mimosa': 'graph',
    'reinvent': 'string',
    'smiles_lstm_hc': 'string',
    'selfies_vae_bo': 'string',
    'dog_gen': 'synthesis',
    'stoned': 'string',
    'gflownet': 'graph',
    'reinvent_selfies': 'string',
    'graph_mcts': 'graph',
    'dst': 'graph',
    'selfies_ga': 'string',
    'gflownet_al': 'graph',
    'screening': 'N/A',
    'mol_pal': 'N/A',
    'graph_ga': 'graph',
    'Ours': 'synthesis'
}

oracles_order = ['qed','gsk3b','jnk3','drd2','median1','median2','celecoxib_rediscovery','osimertinib_mpo','fexofenadine_mpo','ranolazine_mpo','perindopril_mpo','amlodipine_mpo','sitagliptin_mpo','zaleplon_mpo','drd3','7l11','Avg']

def top_k_auc(arr, k, max_oracle_calls=LIMIT):
    min_heap = []
    result = []
    for score in arr:
        heapq.heappush(min_heap, score)
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # Remove the smallest element if more than 10 elements
        result.append(np.mean(min_heap))  # Sort in descending order for top scores
    result.append((max_oracle_calls-len(result))*np.mean(min_heap))
    return sum(result)/max_oracle_calls


def process_res(res, limit):
    sa_oracle = Oracle(name="SA")
    for smi in res:
        sa = sa_oracle(smi)
        res[smi] = res[smi]+[sa]
    K = min(len(res), limit)
    res = {r: res[r] for r in list(res) if res[r][1] <= limit}
    scores = {'K': K}
    scores['Top 1'] = list(res.values())[0][0]
    scores['Top 1 SA'] = list(res.values())[0][-1]
    scores['Top 10'] = np.mean(list(res.values())[:10], axis=0)[0]
    scores['Top 10 SA'] = np.mean(list(res.values())[:10], axis=0)[-1]
    scores['Top 100'] = np.mean(list(res.values())[:100], axis=0)[0]
    scores['Top 100 SA'] = np.mean(list(res.values())[:100], axis=0)[-1]    
    scores['2nd'] = list(res.values())[1][0]
    scores['2nd SA'] = list(res.values())[1][-1]
    scores['3rd'] = list(res.values())[2][0]
    scores['3rd SA'] = list(res.values())[2][-1]    
    running = list(map(lambda x: x[0], sorted(res.values(), key=lambda x:x[1])))
    scores['Top 1 AUC'] = top_k_auc(running, 1)
    scores['Top 10 AUC'] = top_k_auc(running, 10)
    scores['Top 100 AUC'] = top_k_auc(running, 100)    
    return scores    



def load_data(f):
    if f[-4:] == 'yaml':
        return yaml.safe_load(open(f))    
    else:
        df = pd.read_csv(f)
        vals = sorted(df.values, key=lambda x: x[-1], reverse=True)
        dic = {}
        for val in vals:
            smiles, index, score = tuple(val)
            dic[smiles] = [score, index+1]
        return dic



def compute_scores(f, limit=LIMIT):
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
    res = load_data(f)
    scores = process_res(res, limit)
    print("done")
    return scores


def compute_metrics(data_dir, baseline, seed, metric):    
    path = os.path.join(data_dir, f'results_{baseline}_qed_{seed}.yaml')
    out_path = os.path.join(data_dir, f'results_{baseline}_{metric}_{seed}.yaml')
    data = load_data(path)
    result = {}
    if metric == 'SA':
        oracle = Oracle(metric)
        res = [oracle(smi) for smi in data]    
    for smi, score in zip(data, res):
        result[smi] = [score, data[smi][1]]
    yaml.dump(result, open(out_path, 'w'), default_flow_style=False)
    print(out_path)


def compute_ours(path):
    df = pd.read_csv(path)
    df = df.loc[df["smiles"]==df["smiles"]]
    df.idx = np.arange(df.shape[0])
    vals = sorted(list(df.values), key=lambda x: x[-1], reverse=True)
    res = {smi: [score, ind+1] for smi, ind, score in vals}
    scores = process_res(res, limit=LIMIT)
    return scores


def sort_metrics(cols):
    inds = np.arange(len(cols))
    sorted_inds = sorted(inds, key=lambda ind: (oracles_order.index(cols[ind].split()[0]), ind))
    return [cols[i] for i in sorted_inds]



def main(args):
    data_dir = args.baselines_dir    
    metrics = []
    baselines = defaultdict(list)
    for f in os.listdir(data_dir): 
        # use synnet to extract supported metrics
        match = re.match('results_synnet_(\w+)_0.(?:yaml|csv)', f)
        if match:
            metric = match.groups()[0]
            if args.include_metrics and metric not in args.include_metrics:
                continue
            metrics.append(metric)  

    for f in os.listdir(data_dir):
        for metric in metrics:
            match = re.match(f'results_(\w+)_{metric}_(\d).(?:yaml|csv)', f)
            if match:                
                baseline, seed = match.groups()
                if int(seed) not in baselines[baseline]:
                    baselines[baseline].append(int(seed))

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
            match = re.match(f'results_(\w+)_{metric}_(\d).(?:yaml|csv)', f)
            if match:
                baseline, seed = match.groups()            
                # if not (metric in ['drd2','jnk3'] and 'smiles' in baseline): 
                #     continue # for debugging
                # if not (metric in ['drd3']):
                #     continue
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

    # get our metrics
    ours_dir = args.ours_dir 
    pargs = []   
    for i, metric in enumerate(metrics):
        path = os.path.join(ours_dir, f"{metric}.csv")        
        if os.path.exists(path):
            baseline = 'Ours'            
            pargs.append((path, i))

    if args.ncpu:
        with mp.Pool(args.ncpu) as p:
            all_scores = p.map(compute_ours, [a[0] for a in pargs])
    else:
        all_scores = [compute_ours(a[0]) for a in pargs]

    for (path, i), scores in zip(pargs, all_scores):
        res['Ours'][i].append(scores)
    
    df = pd.DataFrame(index=list(res))
    for baseline in res:
        for i in range(len(metrics)):
            df_i = pd.DataFrame(res[baseline][i])
            if len(df_i) == 0:
                continue
            res[baseline][i] = dict(df_i.mean(axis=0)) # aggregate over seeds
        for i, metric in enumerate(metrics):
            for col in res[baseline][i]:
                df.loc[baseline, f"{metric} {col}"] = res[baseline][i][col]

    # summarize over oracles
    pat = 'qed (.+)'
    suffixes = [re.match(pat,c).groups()[0] for c in df.columns if re.match(pat,c)]
    for suffix in suffixes:
        suffix_cols = [col for col in df.columns if col[-len(suffix):] == suffix \
                       and (df[col]==df[col]).all()\
                        and re.search('qed|7l11|drd3', col) is None]
        df[f'Avg {suffix}'] = np.mean([df[col] for col in suffix_cols], axis=0)

    for col in df:
        if col[-2:] == ' K':
            continue
        col_vals = df[col]
        col_vals = col_vals[col_vals == col_vals]
        col_vals = round(col_vals, 3)
        argsort = col_vals.rank(method='min', ascending='SA' in col)
        dic = dict(zip(range(len(col_vals)), map(int, argsort)))
        # group by category
        df_col = pd.DataFrame(col_vals)
        df_col['category'] = [category[method] for method in df_col.index]
        ranks = df_col.groupby('category').rank(method='min', ascending='SA' in col)
        dicc = dict(zip(range(len(col_vals)), map(int, ranks[col])))
        rank = [dic[i] for i in range(len(col_vals))]
        rank_category = [dicc[i] for i in range(len(col_vals))]
        df.loc[df.index[df[col] == df[col]], col] = [f"{score} ({r}|{rc})" for score, r, rc in zip(col_vals, rank, rank_category)]
    df['category'] = [category[method] for method in df.index]
    cand_metrics = df.columns.tolist()[:-1]
    cols = ['category'] + sort_metrics([m for m in cand_metrics if m.split()[0] in oracles_order])
    cols = [col for col in cols if 'seed' not in col and np.all(df[col] == df[col])]
    df = df[cols]    
    # include ours
    path = os.path.join(data_dir, "results.csv")
    df.to_csv(path)
    print(os.path.abspath(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines-dir", default='data/pmo/')
    parser.add_argument("--ours-dir", default='data/ours/')
    parser.add_argument("--ncpu", type=int, default=0)
    parser.add_argument("--include-metrics", nargs='+', help="If given, only consider these metrics")
    parser.add_argument("--metrics", nargs='+', choices=['SA'], help="Additional metrics to compute for each file", default=[])
    args = parser.parse_args()
    main(args)
