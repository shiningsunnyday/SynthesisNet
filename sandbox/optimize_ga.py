import functools
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import List, Literal, Optional

import pydantic_cli
import tqdm
from tdc import Oracle

from ga.config import GeneticSearchConfig
from ga.search import GeneticSearch
from synnet.utils.data_utils import Skeleton, SkeletonSet, binary_tree_to_skeleton
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

    log_file: Optional[str] = None

    # Input file with SMILES strings (First row `SMILES`, then one per line)
    building_blocks_file: str = "data/assets/building-blocks/enamine_us_matched.csv"

    # Input file with reaction templates as SMARTS(No header, one per line)
    rxn_templates_file: str = "data/assets/reaction-templates/hb.txt"
    rxns_collection_file: str = "data/assets/reaction-templates/reactions_hb.json.gz"

    # Input file for the pre-computed embeddings (*.npy)
    embeddings_knn_file: str = "data/assets/building-blocks/enamine_us_emb_fp_256.npy"

    # If given, consider only these bbs
    top_bbs_file: Optional[str] = "/ssd/msun415/bblocks-top-1000.txt"

    # Model checkpoint to use
    ckpt_bb: Optional[str] = None

    # Model checkpoint to use
    ckpt_rxn: Optional[str] = None

    # Recognizer checkpoint to use
    ckpt_recognizer: Optional[str] = None

    # Model checkpoint dir, if given assume one ckpt per class
    ckpt_dir: Optional[str] = None

    # Input file for the ground-truth skeletons to lookup target smiles in
    skeleton_set_file: str

    hash_dir: str

    # Input file for the skeletons of syntree-file
    skeleton_file: str = "results/viz/top_1000/skeletons-top-1000.pkl"

    forcing_eval: bool = False
    mermaid: bool = False
    out_dir: str
    top_k: int = 3
    filter_only: List[Literal["rxn", "bb"]] = []

    # Objective function to optimize
    objective: Literal["qed", "logp", "jnk", "gsk", "drd2", "7l11", "drd3"] = "qed"

    num_workers: int = 1
    chunksize: int = 32

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


def reconstruct(ind):
    sk = binary_tree_to_skeleton(ind.bt)
    tree_key = serialize_string(sk.tree, sk.tree_root)
    index = lookup_skeleton_key(sk.zss_tree, tree_key)

    st0 = globals()["skeleton_list"][index]
    sk0 = Skeleton(st0, index)

    ans = 0.0
    best_smi = ""
    for sk in decode(sk0, ind.fp):
        score, smi = reconstruct(sk, ind.fp, return_smi=True)
        if score > ans:
            ans = score
            best_smi = smi
    return best_smi


def test_surrogate(batch, config: OptimizeGAConfig):
    oracle = fetch_oracle(config.objective)

    with ProcessPoolExecutor(max_workers=config.num_workers) as exe:
        recons = exe.map(reconstruct, batch, chunksize=config.chunksize)
        pbar = tqdm.tqdm(zip(recons, batch), total=len(batch), desc="Evaluating", leave=False)
        for smi, ind in pbar:
            ind.smi = smi
            ind.fitness = oracle(smi)


def main(config: OptimizeGAConfig):
    global args
    args = config  # Hack so reconstruct_utils.py works

    if config.log_file:
        handler = logging.FileHandler(config.log_file)
        logger.addHandler(handler)

    set_models(config, logger)
    load_data(config, logger)
    with open(config.skeleton_set_file, "rb") as f:
        skeletons = pickle.load(f)
        globals()["skeleton_list"] = list(skeletons)  # FIXME: (AL) why does this work?
    skeleton_set = SkeletonSet().load_skeletons(skeletons)
    SKELETON_INDEX = test_skeletons(config, skeleton_set)

    print(f"SKELETON INDEX: {SKELETON_INDEX}")
    logger.info(f"SKELETON INDEX: {SKELETON_INDEX}")

    fn = functools.partial(test_surrogate, config=config)
    search = GeneticSearch(config)
    search.optimize(fn)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(OptimizeGAConfig, main)
