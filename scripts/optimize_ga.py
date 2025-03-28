"""
Generates synthetic trees where the root molecule optimizes for a specific objective
based on Therapeutics Data Commons (TDC) oracle functions.
Uses a genetic algorithm to optimize embeddings before decoding.
"""  # TODO: Refactor/Consolidate with generic inference script
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tdc import Oracle

from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.encoding.distances import cosine_distance
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.MolEmbedder import MolEmbedder
from synnet.utils.data_utils import ReactionSet
from synnet.utils.ga_utils import crossover, mutation
from synnet.utils.predict_utils import mol_fp, synthetic_tree_decoder, tanimoto_similarity


def _fetch_gin_molembedder():
    from dgllife.model import load_pretrained

    # define model to use for molecular embedding
    model_type = "gin_supervised_contextpred"
    device = "cpu"
    mol_embedder = load_pretrained(model_type).to(device)
    return mol_embedder.eval()


def _fetch_molembedder(featurize: str):
    """Fetch molembedder."""
    if featurize == "fp":
        return None  # not in use
    else:
        raise NotImplementedError
        return _fetch_gin_molembedder()


def func(emb):
    """
    Generates the synthetic tree for the input molecular embedding.

    Args:
        emb (np.ndarray): Molecular embedding to decode.

    Returns:
        str: SMILES for the final chemical node in the tree.
        SyntheticTree: The generated synthetic tree.
    """
    emb = emb.reshape((1, -1))
    try:
        tree, action = synthetic_tree_decoder(
            z_target=emb,
            sk_coords=None,
            building_blocks=bblocks,
            bb_dict=bb_dict,
            reaction_templates=rxns,
            mol_embedder=bblocks_molembedder.kdtree,  # TODO: fix this, currently misused,
            action_net=act_net,
            reactant1_net=rt1_net,
            rxn_net=rxn_net,
            reactant2_net=rt2_net,
            bb_emb=bb_emb,
            rxn_template=rxn_template,
            n_bits=nbits,
            max_step=15,
        )
    except Exception as e:
        print(e)
        action = -1
    if action != 3:
        return None, None
    else:
        scores = np.array(tanimoto_similarity(emb, [node.smiles for node in tree.chemicals]))
        max_score_idx = np.where(scores == np.max(scores))[0][0]
        return tree.chemicals[max_score_idx].smiles, tree


def dock_drd3(smi):
    """
    Returns the docking score for the DRD3 target.

    Args:
        smi (str): SMILES for the molecule to predict the docking score of.

    Returns:
        float: Predicted docking score against the DRD3 target.
    """
    # define the oracle function from the TDC
    _drd3 = Oracle(name="drd3_docking")

    if smi is None:
        return 0.0
    else:
        try:
            return -_drd3(smi)
        except:
            return 0.0


def dock_7l11(smi):
    """
    Returns the docking score for the 7L11 target.

    Args:
        smi (str): SMILES for the molecule to predict the docking score of.

    Returns:
        float: Predicted docking score against the 7L11 target.
    """
    # define the oracle function from the TDC
    _7l11 = Oracle(name="7l11_docking")
    if smi is None:
        return 0.0
    else:
        try:
            return -_7l11(smi)
        except:
            return 0.0
        


def dock_liep(smi):
    """
    Returns the docking score for the liep target.

    Args:
        smi (str): SMILES for the molecule to predict the docking score of.

    Returns:
        float: Predicted docking score against the liep target.
    """
    # define the oracle function from the TDC
    _liep = Oracle(name="liep_docking")
    if smi is None:
        return 0.0
    else:
        try:
            return -_liep(smi)
        except:
            return 0.0
        


def dock_2rgp(smi):
    """
    Returns the docking score for the 2rgp target.

    Args:
        smi (str): SMILES for the molecule to predict the docking score of.

    Returns:
        float: Predicted docking score against the 2rgp target.
    """
    # define the oracle function from the TDC
    _2rgp = Oracle(name="2rgp_docking")
    if smi is None:
        return 0.0
    else:
        try:
            return -_2rgp(smi)
        except:
            return 0.0
        


def dock_3pbl(smi):
    """
    Returns the docking score for the 3pbl target.

    Args:
        smi (str): SMILES for the molecule to predict the docking score of.

    Returns:
        float: Predicted docking score against the 3pbl target.
    """
    # define the oracle function from the TDC
    _3pbl = Oracle(name="3pbl_docking")
    if smi is None:
        return 0.0
    else:
        try:
            return -_3pbl(smi)
        except:
            return 0.0



def fitness(embs, _pool, obj):
    """
    Returns the scores for the root molecules in synthetic trees generated by the
    input molecular embeddings.

    Args:
        embs (list): Contains molecular embeddings (vectors).
        _pool (mp.Pool): A pool object, which represents a pool of workers (used
            for multiprocessing).
        obj (str): The objective function to use to compute the fitness.

    Raises:
        ValueError: Raised if the specified objective function is not implemented.

    Returns:
        scores (list): Contains the scores for the root molecules in the
            generated trees.
        smiles (list): Contains the root molecules encoded as SMILES strings.
        trees (list): Contains the synthetic trees generated from the input
            embeddings.
    """
    results = _pool.map(func, embs)
    smiles = [r[0] for r in results]
    trees = [r[1] for r in results]

    if obj == "qed":
        # define the oracle function from the TDC
        qed = Oracle(name="QED")
        scores = [qed(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == "logp":
        # define the oracle function from the TDC
        logp = Oracle(name="LogP")
        scores = [logp(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == "jnk":
        # define the oracle function from the TDC
        jnk = Oracle(name="JNK3")
        scores = [jnk(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == "gsk":
        # define the oracle function from the TDC
        gsk = Oracle(name="GSK3B")
        scores = [gsk(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == "drd2":
        # define the oracle function from the TDC
        drd2 = Oracle(name="DRD2")
        scores = [drd2(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == "ASKCOS":
        askcos = Oracle(name="ASKCOS")
        host_ip = 'http://xx.xx.xxx.xxx'
        scores = [askcos(smi, host_ip, output='plausibility') for smi in smiles]
    elif obj == "IBM_RXN":
        oracle = Oracle(name='IBM_RXN')
        key = 'apk-c9db......' # You can obtain a key from https://rxn.res.ibm.com
        scores = [oracle(smi, key) for smi in smiles]
    elif obj == "Celecoxib_Rediscovery":
        oracle = Oracle(name='Celecoxib_Rediscovery')
        scores = [oracle(smi) for smi in smiles]
    elif obj == "Troglitazone_Rediscovery":
        oracle = Oracle(name='Troglitazone_Rediscovery')
        scores = [oracle(smi) for smi in smiles]        
    elif obj == "Thiothixene_Rediscovery":
        oracle = Oracle(name='Thiothixene_Rediscovery')
        scores = [oracle(smi) for smi in smiles]                
    elif obj == "Aripiprazole_Similarity":
        oracle = Oracle(name='Aripiprazole_Similarity')
        scores = [oracle(smi) for smi in smiles]
    elif obj == "Albuterol_Similarity":
        oracle = Oracle(name='Albuterol_Similarity')
        scores = [oracle(smi) for smi in smiles]
    elif obj == "Mestranol_Similarity":
        oracle = Oracle(name='Mestranol_Similarity')
        scores = [oracle(smi) for smi in smiles]                  
    elif obj == "Median_1":
        oracle = Oracle(name="Median 1")
        scores = [oracle(smi) for smi in smiles]
    elif obj == "Median_2":
        oracle = Oracle(name="Median 2")
        scores = [oracle(smi) for smi in smiles]
    elif obj == "Isomers":
        oracle = Oracle(name="Isomers")
        scores = [oracle(smi) for smi in smiles]        
    elif obj == "Osimertinib_MPO":
        oracle = Oracle(name="Osimertinib_MPO")
        scores = [oracle(smi) for smi in smiles]
    elif obj == "Fexofenadine_MPO":
        oracle = Oracle(name="Fexofenadine_MPO")
        scores = [oracle(smi) for smi in smiles]        
    elif obj == "Ranolazine_MPO":
        oracle = Oracle(name="Ranolazine_MPO")
        scores = [oracle(smi) for smi in smiles]        
    elif obj == "Perindopril_MPO":
        oracle = Oracle(name="Perindopril_MPO")
        scores = [oracle(smi) for smi in smiles]        
    elif obj == "Amlodipine_MPO":
        oracle = Oracle(name="Amlodipine_MPO")
        scores = [oracle(smi) for smi in smiles]        
    elif obj == "Sitagliptin_MPO":
        oracle = Oracle(name="Sitagliptin_MPO")
        scores = [oracle(smi) for smi in smiles]        
    elif obj == "Zaleplon_MPO":
        oracle = Oracle(name="Zaleplon_MPO")
        scores = [oracle(smi) for smi in smiles]                
    # docking
    elif obj == "liep":
        scores = [dock_liep(smi) for smi in smiles]
    elif obj == "2rgp":
        scores = [dock_2rgp(smi) for smi in smiles]        
    elif obj == "7l11":
        scores = [dock_7l11(smi) for smi in smiles]
    elif obj == "drd3":
        scores = [dock_drd3(smi) for smi in smiles]
    elif obj == "3pbl":
        scores = [dock_3pbl(smi) for smi in smiles]
    else:
        raise ValueError("Objective function not implemneted")
    return scores, smiles, trees


def distribution_schedule(n, total):
    """
    Determines the type of probability to use in the `crossover` function, based
    on the number of generations which have occured.

    Args:
        n (int): Number of elapsed generations.
        total (int): Total number of expected generations.

    Returns:
        str: Describes a type of probability distribution.
    """
    if n < 4 * total / 5:
        return "linear"
    else:
        return "softmax_linear"


def num_mut_per_ele_scheduler(n, total):
    """
    Determines the number of bits to mutate in each vector, based on the number
    of elapsed generations.

    Args:
        n (int): Number of elapsed generations.
        total (int): Total number of expected generations.

    Returns:
        int: Number of bits to mutate.
    """
    # if n < total/2:
    #     return 256
    # else:
    #     return 512
    return 24


def mut_probability_scheduler(n, total):
    """
    Determines the probability of mutating a vector, based on the number of elapsed
    generations.

    Args:
        n (int): Number of elapsed generations.
        total (int): Total number of expected generations.

    Returns:
        float: The probability of mutation.
    """
    if n < total / 2:
        return 0.5
    else:
        return 0.5


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--building-blocks-file",
        type=str,
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--rxns-collection-file",
        type=str,
        help="Input file for the collection of reactions matched with building-blocks.",
    )
    parser.add_argument(
        "--embeddings-knn-file",
        type=str,
        help="Input file for the pre-computed embeddings (*.npy).",
    )
    parser.add_argument(
        "--top-bbs-file",
        type=str,
        help="If given, limit to only bbs from this"
    )    
    parser.add_argument("--ckpt-versions", type=int, help="If given, use ckpt versions in ckpt-dir", nargs='+')
    parser.add_argument(
        "--ckpt-dir", type=str, help="Directory with checkpoints for {act,rt1,rxn,rt2}-model."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="A file contains the starting mating pool.",
    )
    parser.add_argument(
        "--objective", type=str, default="qed", help="Objective function to optimize"
    )
    parser.add_argument("--radius", type=int, default=2, help="Radius for Morgan fingerprint.")
    parser.add_argument(
        "--nbits", type=int, default=4096, help="Number of Bits for Morgan fingerprint."
    )
    parser.add_argument(
        "--num_population", type=int, default=100, help="Number of parents sets to keep."
    )
    parser.add_argument(
        "--num_offspring",
        type=int,
        default=300,
        help="Number of offsprings to generate each iteration.",
    )
    parser.add_argument("--num_gen", type=int, default=30, help="Number of generations to proceed.")
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument(
        "--mut_probability",
        type=float,
        default=0.5,
        help="Probability to mutate for one offspring.",
    )
    parser.add_argument(
        "--num_mut_per_ele",
        type=int,
        default=1,
        help="Number of bits to mutate in one fingerprint.",
    )
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    return parser.parse_args()


def fetch_population(args) -> np.ndarray:
    if args.restart:
        population = np.load(args.input_file)
        print(f"Starting with {len(population)} fps from {args.input_file}")
    else:
        if args.input_file is None:
            population = np.ceil(np.random.random(size=(args.num_population, args.nbits)) * 2 - 1)
            print(f"Starting with {args.num_population} fps with {args.nbits} bits")
        else:
            if '.csv' in args.input_file:
                starting_smiles = pd.read_csv(args.input_file).sample(args.num_population)
                starting_smiles = starting_smiles["smiles"].tolist()
            elif '.json' in args.input_file:
                dics = json.load(open(args.input_file))
                starting_smiles = [dic['smi'] for dic in dics]
            else:
                raise NotImplementedError
            population = np.array([mol_fp(smi, args.radius, args.nbits) for smi in starting_smiles])
            print(f"Starting with {len(starting_smiles)} fps from {args.input_file}")
    return population


if __name__ == "__main__":

    args = get_args()
    np.random.seed(args.seed)

    # define some constants (here, for the Hartenfeller-Button test set)
    nbits = 4096
    out_dim = 256
    rxn_template = "hb"
    featurize = "fp"
    param_dir = "hb_fp_2_4096_256"

    # Load data
    mol_embedder = _fetch_molembedder(featurize)

    # load the purchasable building block SMILES to a dictionary
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    if args.top_bbs_file:
        bblock_inds = [bblocks.index(l.rstrip('\n')) for l in open(args.top_bbs_file).readlines()]
        bblock_inds = sorted(bblock_inds)
        bblocks = [bblocks[ind] for ind in bblock_inds]
        bb_dict = {block: i for i, block in enumerate(bblocks)}
        emb_path = args.top_bbs_file.replace('.txt', '.npy')
        # if not os.path.exists(emb_path):
        data = np.load(args.embeddings_knn_file)
        top_emb = data[bblock_inds]
        np.save(emb_path, top_emb)  
        # ... reaction templates
        bblock_set = set(bblocks)
        rxns = ReactionSet().load(args.rxns_collection_file).rxns
        for rxn in rxns:
            for i in range(len(rxn.available_reactants)):
                rxn.available_reactants[i] = [reactant for reactant in rxn.available_reactants[i] if reactant in bblock_set]              
        bblocks_molembedder = (
            MolEmbedder().load_precomputed(emb_path).init_balltree(cosine_distance)
        )     
        bb_emb = bblocks_molembedder.get_embeddings()         
    else:
        bblock_inds = None        
        # A dict is used as lookup table for 2nd reactant during inference:
        bb_dict = {block: i for i, block in enumerate(bblocks)}
        # Reactions
        rxns = ReactionSet().load(args.rxns_collection_file).rxns
        # ... building block embedding
        bblocks_molembedder = (
            MolEmbedder().load_precomputed(args.embeddings_knn_file).init_balltree(cosine_distance)
        )
        bb_emb = bblocks_molembedder.get_embeddings()


    # load the pre-trained modules
    path = Path(args.ckpt_dir)
    if args.ckpt_versions:
        versions = args.ckpt_versions
    else:
        versions = [None, None, None, None]    
    ckpt_files = []
    for model, version in zip("act rt1 rxn rt2".split(), versions):
        ckpt_file = find_best_model_ckpt(path / model, version)
        ckpt_files.append(ckpt_file)    
    act_net, rt1_net, rxn_net, rt2_net = [load_mlp_from_ckpt(file) for file in ckpt_files]

    # Get initial population
    population = fetch_population(args)

    # Evaluation initial population
    with mp.Pool(processes=args.ncpu) as pool:
        scores, mols, trees = fitness(embs=population, _pool=pool, obj=args.objective)

    scores = np.array(scores)
    score_x = np.argsort(scores)
    population = population[score_x[::-1]]
    mols = [mols[i] for i in score_x[::-1]]
    scores = scores[score_x[::-1]]
    print(f"Initial: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"Scores: {scores}")
    print(f"Top-3 Smiles: {mols[:3]}")

    # Genetic Algorithm: loop over generations
    recent_scores = []
    for n in range(args.num_gen):
        t = time.time()

        dist_ = distribution_schedule(n, args.num_gen)
        num_mut_per_ele_ = num_mut_per_ele_scheduler(n, args.num_gen)
        mut_probability_ = mut_probability_scheduler(n, args.num_gen)

        offspring = crossover(
            parents=population, offspring_size=args.num_offspring, distribution=dist_
        )
        offspring = mutation(
            offspring_crossover=offspring,
            num_mut_per_ele=num_mut_per_ele_,
            mut_probability=mut_probability_,
        )
        new_population = np.unique(np.concatenate([population, offspring], axis=0), axis=0)
        with mp.Pool(processes=args.ncpu) as pool:
            new_scores, new_mols, trees = fitness(new_population, pool, args.objective)
        new_scores = np.array(new_scores)
        scores = []
        mols = []

        parent_idx = 0
        indices_to_print = []
        while parent_idx < args.num_population:
            max_score_idx = np.where(new_scores == np.max(new_scores))[0][0]
            if new_mols[max_score_idx] not in mols:
                indices_to_print.append(max_score_idx)
                scores.append(new_scores[max_score_idx])
                mols.append(new_mols[max_score_idx])
                population[parent_idx, :] = new_population[max_score_idx, :]
                new_scores[max_score_idx] = -999999
                parent_idx += 1
            else:
                new_scores[max_score_idx] = -999999

        scores = np.array(scores)
        print(f"Generation {n+1}: {scores.mean():.3f} +/- {scores.std():.3f}")
        print(f"Scores: {scores}")
        print(f"Top-3 Smiles: {mols[:3]}")
        print(f"Consumed time: {(time.time() - t):.3f} s")
        print()
        for i in range(3):
            trees[indices_to_print[i]]._print()
        print()

        recent_scores.append(scores.mean())
        if len(recent_scores) > 10:
            del recent_scores[0]

        np.save("population_" + args.objective + "_" + str(n + 1) + ".npy", population)

        data = {
            "objective": args.objective,
            "top1": np.mean(scores[:1]),
            "top10": np.mean(scores[:10]),
            "top100": np.mean(scores[:100]),
            "smiles": mols,
            "scores": scores.tolist(),
        }
        with open("opt_" + args.objective + f"_{str(n + 1)}.json", "w") as f:
            json.dump(data, f)

        if n > 30 and recent_scores[-1] - recent_scores[0] < 0.01:
            print("Early Stop!")
            break

    # Save results
    data = {
        "objective": args.objective,
        "top1": np.mean(scores[:1]),
        "top10": np.mean(scores[:10]),
        "top100": np.mean(scores[:100]),
        "smiles": mols,
        "scores": scores.tolist(),
    }
    with open("opt_" + args.objective + ".json", "w") as f:
        json.dump(data, f)

    np.save("population_" + args.objective + ".npy", population)
