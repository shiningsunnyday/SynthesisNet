from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/pre-process/syntrees/synthetic-trees.json.gz"
    )
    parser.add_argument(
        "--stgen-cache",
        type=str,
        default="data/assets/stgen.pkl"
    )    
    parser.add_argument(
        "--rxn_file",
        type=str
    )        
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/pre-process/syntrees/synthetic-trees.csv"
    )       

    # Processing
    # parser.add_argument("--ncpu", type=int, help="Number of cpus")
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    syntree_collection = SyntheticTreeSet()
    syntrees = syntree_collection.load(args.input_file)
    stgen = pickle.load(open(args.stgen_cache,'rb'))
    df_data = defaultdict(list)
    for syntree in syntrees:
        for rxn in syntree.reactions:
            product = rxn.parent
            reactants = '.'.join(rxn.child)
            rxn_smirks = stgen.rxns[rxn.rxn_id].smirks
            if '>>' in rxn_smirks:        
                a, b = rxn_smirks.split('>>')
                retro_template = '>>'.join([b, a])
            else:
                a, b, c = rxn_smirks.split('>')
                retro_template = '>'.join([c, b, a])
            df_data['product'].append(product)
            df_data['reactants'].append(reactants)
            df_data['retro_template'].append(retro_template)
    df = pd.DataFrame(df_data)
    df.to_csv(args.output_file)
