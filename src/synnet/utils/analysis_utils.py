from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet, Skeleton
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
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


def reorder(syntree):    
    total_swapped = 0
    correct = 0
    incorrect = 0
    for i in range(len(syntree.chemicals)):
        swapped = False
        edges = [e for e in syntree.edges if e[0] == i]
        assert len(edges) in [0, 1, 2]
        if len(edges) == 2:
            # decide if swap needed
            c1, c2 = edges[0][1], edges[1][1]
            assert syntree.chemicals[c1].parent == syntree.chemicals[c2].parent
            rxn_id = syntree.chemicals[c1].parent
            if not rxns[rxn_id].is_reactant_first(syntree.chemicals[c1].smiles) \
                or not rxns[rxn_id].is_reactant_second(syntree.chemicals[c2].smiles):
                ind1 = syntree.edges.index(edges[0])
                ind2 = syntree.edges.index(edges[1])
                syntree.edges[ind1], syntree.edges[ind2] = syntree.edges[ind2], syntree.edges[ind1]
                swapped = True
                print("swapped")
        total_swapped += swapped
    for reaction in syntree.reactions:
        if len(reaction.child) == 2:
            rxn_id = reaction.rxn_id
            if rxns[rxn_id].is_reactant_first(reaction.child[0]):
                if not rxns[rxn_id].is_reactant_second(reaction.child[0]):
                    assert rxns[rxn_id].is_reactant_second(reaction.child[1])                    
                else:
                    assert rxns[rxn_id].is_reactant_first(reaction.child[1]) \
                        or rxns[rxn_id].is_reactant_second(reaction.child[1])
                correct += 1
            else:
                assert rxns[rxn_id].is_reactant_second(reaction.child[0])
                assert rxns[rxn_id].is_reactant_first(reaction.child[1])
                incorrect += 1     
    return syntree, total_swapped, correct, incorrect


def reorder_syntrees(syntrees, rxns):
    correct = 0
    incorrect = 0
    total_swapped = 0    
    globals()["rxns"] = rxns
    with mp.Pool(50) as p:
        res = p.map(reorder, tqdm(syntrees, desc="reordering"))
    syntrees = [r[0] for r in res]
    total_swapped = sum([r[1] for r in res])
    correct = sum([r[2] for r in res])
    incorrect = sum([r[3] for r in res])
    print()
    print(f"correct: {correct}, incorrect: {incorrect}, total swapped: {total_swapped}")   
    return syntrees



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


