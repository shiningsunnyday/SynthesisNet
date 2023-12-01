from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet, Skeleton
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from networkx.drawing.nx_pydot import graphviz_layout

def skeleton2graph(skeleton):
    graph = nx.MultiDiGraph()
    count = {}
    lookup = {}
    for n in skeleton.nodes:
        name = n.smiles
        if n.smiles in count:
            name += f":{count[n.smiles]}"
        graph.add_node(name)
        count[n.smiles] = count.get(n.smiles, 0)+1
        lookup[n] = name
    for e in skeleton.edges:
        graph.add_edge(lookup[skeleton.nodes[e[0]]], lookup[skeleton.nodes[e[1]]])
    return graph, count[skeleton.root.smiles]



def vis_skeletons(args, skeletons):
    min_count = args.min_count
    max_i = args.num_to_vis
    fig_path = os.path.join(args.visualize_dir, 'skeletons.png')
    fig = plt.Figure(figsize=(30, 30))
    skeletons = {k:skeletons[k] for k in skeletons if len(skeletons[k]) >= min_count}
    for i in range(max_i):
        for j, sk in enumerate(skeletons):
            ax = fig.add_subplot(max_i, len(skeletons), i*len(skeletons)+j+1)
            G, root = skeleton2graph(skeletons[sk][i])
            pos = Skeleton.hierarchy_pos(G, root)
            # pos = graphviz_layout(G, prog="twopi")
            # pos = nx.circular_layout(G)
            node_sizes = [200 for _ in G.nodes()]
            node_sizes[list(G.nodes()).index(skeletons[sk][i].root.smiles)] *= 2
            nx.draw_networkx(G, pos=pos, ax=ax, node_size=node_sizes)
    
    # fig(f"{args.num_to_vis} representing {len(skeletons)} classes")    
    fig.savefig(fig_path)    
    print(f"visualized some skeletons at {fig_path}")


def count_skeletons(args, skeletons):
    fig_path = os.path.join(args.visualize_dir, 'skeletons_count.png')
    fig = plt.Figure()
    counts = [len(skeletons[k]) for k in skeletons]
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(len(counts)), sorted(counts, key=lambda x:-x))
    ax.set_xlabel('skeleton')
    ax.set_ylabel('count')
    ax.set_yscale('log')
    fig.savefig(fig_path)
    print(f"visualized count at {fig_path}")


def count_bbs(args, skeletons, vis=True):
    fig_path = os.path.join(args.visualize_dir, 'bb_count.png')
    fig = plt.Figure()
    bb_count = defaultdict(int)
    for sk in tqdm(skeletons):
        for st in skeletons[sk]:
            for c in st.chemicals:
                if c.is_leaf:
                    bb_count[c.smiles] += 1
    if vis:        
        counts = list(bb_count.values())
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(range(len(counts)), sorted(counts, key=lambda x:-x))
        ax.set_xlabel('bb')
        ax.set_ylabel('count')
        ax.set_yscale('log')
        fig.savefig(fig_path)
        print(f"visualized count at {fig_path}") 
    return bb_count   


def count_rxns(args, skeletons, vis=True):
    fig_path = os.path.join(args.visualize_dir, 'rxn_count.png')
    fig = plt.Figure()
    rxn_count = defaultdict(int)
    for sk in skeletons:
        for st in skeletons[sk]:
            for r in st.reactions:
                rxn_count[r.rxn_id] += 1
    if vis:
        counts = list(rxn_count.values())
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(range(len(counts)), sorted(counts, key=lambda x:-x))
        ax.set_xlabel('rxn')
        ax.set_ylabel('count')
        ax.set_yscale('log')
        fig.savefig(fig_path)
        print(f"visualized count at {fig_path}")   
    return rxn_count