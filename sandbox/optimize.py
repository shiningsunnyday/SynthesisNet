import functools
import logging
import pathlib
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Literal, Optional

import numpy as np
import pydantic_cli
import torch
import tqdm
from tdc import Oracle

from ga.config import GeneticSearchConfig
from ga.search import GeneticSearch
from synnet.MolEmbedder import MolEmbedder
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.encoding.distances import cosine_distance
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.utils.data_utils import ReactionSet, Skeleton, SkeletonSet, binary_tree_to_skeleton
from synnet.utils.predict_utils import synthetic_tree_decoder, tanimoto_similarity
from synnet.utils.reconstruct_utils import (
    decode,
    load_data,
    lookup_skeleton_key,
    reconstruct,
    serialize_string,
    set_models,
    test_skeletons,
)

logger = logging.getLogger(__name__)

args = None  # Hack so reconstruct_utils.py works


class OptimizeGAConfig(GeneticSearchConfig):
    """Config for running the GA."""

    method: Literal["synnet", "ours"] = "synnet"

    log_file: Optional[str] = None

    # Input file with SMILES strings (First row `SMILES`, then one per line)
    building_blocks_file: str = "data/assets/building-blocks/enamine_us_matched.csv"

    # Input file with reaction templates as SMARTS(No header, one per line)
    rxn_templates_file: str = "data/assets/reaction-templates/hb.txt"
    rxns_collection_file: str = "data/assets/reaction-templates/reactions_hb.json.gz"

    # Input file for the pre-computed embeddings (*.npy)
    embeddings_knn_file: str = "data/assets/building-blocks/enamine_us_emb_fp_256.npy"

    # If given, consider only these bbs
    top_bbs_file: Optional[str] = None

    # Model checkpoint to use
    ckpt_bb: Optional[str] = None

    # Model checkpoint to use
    ckpt_rxn: Optional[str] = None

    # Recognizer checkpoint to use
    ckpt_recognizer: Optional[str] = None

    # Model checkpoint dir, if given assume one ckpt per class
    ckpt_dir: str = None
    ckpt_versions: Optional[List[int]] = None

    # Input file for the ground-truth skeletons to lookup target smiles in
    skeleton_set_file: str

    # Input file for the skeletons of syntree-file
    skeleton_file: str = "results/viz/top_1000/skeletons-top-1000.pkl"

    forcing_eval: bool = False
    mermaid: bool = False
    one_per_class: bool = False

    out_dir: str

    # Beam width for first bb
    top_k: int = 3

    # Beam width for first rxn
    top_k_rxn: int = 3

    # Restrict syntree test set to max number of reactions (-1 to do syntrees, 0 to
    # syntrees whose skeleton class was trained on by ckpt_dir)
    max_rxns: int = -1

    # Restrict skeleton prediction to max number of reactions
    max_num_rxns: int = -1

    filter_only: List[Literal["rxn", "bb"]] = []

    # Objective function to optimize
    objective: Literal["qed", "logp", "jnk", "gsk", "drd2", "7l11", "drd3"] = "qed"

    num_workers: int = 1
    chunksize: int = 32

    # Conf: Decode all reactions before bbs. Choose highest-confidence reaction. Choose closest neighbor bb.
    # Topological: Decode every topological order of the rxn+bb nodes.
    strategy: Literal["conf", "topological"] = "topological"
    test_correct_method: Literal["preorder", "postorder", "reconstruct"] = "reconstruct"


def dock_drd3(smi):
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
    # define the oracle function from the TDC
    _7l11 = Oracle(name="7l11_docking")

    if smi is None:
        return 0.0
    else:
        try:
            return -_7l11(smi)
        except:
            return 0.0


def fetch_oracle(objective):
    if objective == "qed":
        # define the oracle function from the TDC
        return Oracle(name="QED")
    elif objective == "logp":
        # define the oracle function from the TDC
        return Oracle(name="LogP")
    elif objective == "jnk":
        # return oracle function from the TDC
        return Oracle(name="JNK3")
    elif objective == "gsk":
        # return oracle function from the TDC
        return Oracle(name="GSK3B")
    elif objective == "drd2":
        # return oracle function from the TDC
        return Oracle(name="DRD2")
    elif objective == "7l11":
        return dock_7l11
    elif objective == "drd3":
        return dock_drd3
    else:
        raise ValueError("Objective function not implemented")


def get_smiles_ours(ind):
    sk = binary_tree_to_skeleton(ind.bt)
    tree_key = serialize_string(sk.tree, sk.tree_root)
    index = lookup_skeleton_key(sk.zss_tree, tree_key)

    st0 = globals()["skeleton_list"][index]
    sk0 = Skeleton(st0, index)

    ans = 0.0
    best_smi = ""
    for sk in decode(sk0, ind.fp):
        score, smi = reconstruct(sk, ind.fp)
        if score > ans:
            ans = score
            best_smi = smi
    return best_smi


def get_smiles_synnet(
    ind,
    bblocks, bb_dict,
    rxns,
    bblocks_molembedder,
    act_net, rt1_net, rxn_net, rt2_net,
    bb_emb,
    rxn_template,
    nbits
):
    emb = ind.fp.reshape((1, -1))
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


# Needed to avoid deadlock
# Reference:
#   https://github.com/pytorch/pytorch/issues/17199
def thread_quarantine(ind, converter):
    with ThreadPoolExecutor(max_workers=1) as exe:
        return exe.submit(converter, ind).result()


def test_surrogate(batch, converter, config: OptimizeGAConfig):
    oracle = fetch_oracle(config.objective)

    # Debug option
    if config.num_workers <= 0:
        smiles = map(converter, batch)
        pbar = tqdm.tqdm(zip(smiles, batch), total=len(batch), desc="Evaluating", leave=False)
        for smi, ind in pbar:
            ind.smiles = smi
            ind.fitness = oracle(smi)
        return

    with ProcessPoolExecutor(max_workers=config.num_workers) as exe:
        smiles = exe.map(converter, batch, chunksize=config.chunksize)
        pbar = tqdm.tqdm(zip(smiles, batch), total=len(batch), desc="Evaluating", leave=False)
        for smi, ind in pbar:
            ind.smiles = smi
            ind.fitness = oracle(smi)


def main(config: OptimizeGAConfig):
    global args
    args = config  # Hack so reconstruct_utils.py works

    if config.log_file:
        handler = logging.FileHandler(config.log_file)
        logger.addHandler(handler)

    if config.method == "ours":
        set_models(config, logger)
        load_data(config, logger)
        with open(config.skeleton_set_file, "rb") as f:
            skeletons = pickle.load(f)
            globals()["skeleton_list"] = list(skeletons)  # FIXME: (AL) why does this work?
        skeleton_set = SkeletonSet().load_skeletons(skeletons)
        SKELETON_INDEX = test_skeletons(config, skeleton_set)
        logger.info(f"SKELETON INDEX: {SKELETON_INDEX}")

        converter = get_smiles_ours

    elif config.method == "synnet":
        # define some constants (here, for the Hartenfeller-Button test set)
        nbits = 4096
        rxn_template = "hb"

        # load the purchasable building block SMILES to a dictionary
        bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
        bblock_inds = None
        # A dict is used as lookup table for 2nd reactant during inference:
        bb_dict = {block: i for i, block in enumerate(bblocks)}
        # ... building block embedding
        bblocks_molembedder = (
            MolEmbedder().load_precomputed(args.embeddings_knn_file).init_balltree(cosine_distance)
        )
        bb_emb = bblocks_molembedder.get_embeddings()

        # load the reaction templates as a ReactionSet object
        rxns = ReactionSet().load(args.rxns_collection_file).rxns
        for rxn in rxns:
            for i in range(len(rxn.available_reactants)):
                rxn.available_reactants[i] = [reactant for reactant in rxn.available_reactants[i] if
                                              reactant in bblocks]

        # load the pre-trained modules
        path = pathlib.Path(args.ckpt_dir)
        if args.ckpt_versions:
            versions = args.ckpt_versions
        else:
            versions = [None, None, None, None]
        ckpt_files = []
        for model, version in zip("act rt1 rxn rt2".split(), versions):
            ckpt_file = find_best_model_ckpt(path / model, version)
            ckpt_files.append(ckpt_file)
        act_net, rt1_net, rxn_net, rt2_net = [load_mlp_from_ckpt(file) for file in ckpt_files]

        converter = functools.partial(
            get_smiles_synnet,
            bblocks=bblocks, bb_dict=bb_dict,
            rxns=rxns,
            bblocks_molembedder=bblocks_molembedder,
            act_net=act_net, rt1_net=rt1_net, rxn_net=rxn_net, rt2_net=rt2_net,
            bb_emb=bb_emb,
            rxn_template=rxn_template,
            nbits=nbits,
        )

    else:
        raise NotImplementedError()
    
    converter = functools.partial(thread_quarantine, converter=converter)
    fn = functools.partial(test_surrogate, converter=converter, config=config)
    search = GeneticSearch(config)
    search.optimize(fn)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(OptimizeGAConfig, main)
