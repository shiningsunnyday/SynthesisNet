"""Generate synthetic trees.
"""  # TODO: clean up this mess
import json
import logging
from collections import Counter
from pathlib import Path

from rdkit import RDLogger
from tqdm import tqdm
import pickle
import os

from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    ReactionTemplateFileHandler,
)
from synnet.data_generation.syntrees import SynTreeGenerator, wraps_syntreegenerator_generate
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet
import numpy as np

logger = logging.getLogger(__name__)
from typing import Tuple, Union

RDLogger.DisableLog("rdApp.*")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--building-blocks-file",
        type=str,
        default="data/pre-process/building-blocks/enamine-us-smiles.csv.gz",  # TODO: change
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument("--top-bbs-file", help='if given, consider only these bbs')
    parser.add_argument(
        "--rxn-templates-file",
        type=str,
        default="data/assets/reaction-templates/hb.txt",  # TODO: change
        help="Input file with reaction templates as SMARTS(No header, one per line).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/pre-precess/synthetic-trees.json.gz",
        help="Output file for the generated synthetic trees (*.json.gz)",
    )
    parser.add_argument(
        "--stgen_cache",
        type=str,
        default="",
        help="location of cached stgen, or location to cache stgen",
    )
    # Parameters
    parser.add_argument(
        "--number-syntrees", type=int, default=600000, help="Number of SynTrees to generate."
    )

    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--log_file", help="Where to log file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def generate_mp() -> Tuple[dict[int, str], list[Union[SyntheticTree, None]]]:
    from functools import partial

    import numpy as np
    from pathos import multiprocessing as mp

    def wrapper(stgen, _):
        stgen.rng = np.random.default_rng()  # TODO: Think about this...
        return wraps_syntreegenerator_generate(stgen)

    func = partial(wrapper, stgen)

    with mp.Pool(processes=args.ncpu) as pool:
        results = list(tqdm(pool.imap(func, range(args.number_syntrees)), total=args.number_syntrees//args.ncpu))

    outcomes = {
        i: e.__class__.__name__ if e is not None else "success" for i, (_, e) in enumerate(results)
    }
    syntrees = [st for (st, e) in results if e is None]
    return outcomes, syntrees


def generate() -> Tuple[dict[int, str], list[Union[SyntheticTree, None]]]:
    outcomes: dict[int, str] = dict()
    syntrees: list[Union[SyntheticTree, None]] = []
    myrange = tqdm(range(args.number_syntrees)) if args.verbose else range(args.number_syntrees)
    for i in myrange:
        st, e = wraps_syntreegenerator_generate(stgen)        
        if e is None:
            print(st.chemicals[-1].smiles)
            if st.chemicals[-1].smiles == 'COc1cc(C)c(N)c(C(=O)OC(CCl)(c2ccc([N+](=O)[O-])c(Cl)c2S(N)(=O)=O)C2CCCCO2)c1':
                breakpoint()
        outcomes[i] = e.__class__.__name__ if e is not None else "success"
        syntrees.append(st)    

    return outcomes, syntrees


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Load assets
    # building blocks
    if args.top_bbs_file:
        bblocks = [l.rstrip('\n') for l in open(args.top_bbs_file).readlines()]
    else:
        bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)            
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)
    logger.info("Loaded building block & rxn-template assets.")

    # Init SynTree Generator
    logger.info("Start initializing SynTreeGenerator...")
    if args.stgen_cache and os.path.exists(args.stgen_cache):
        stgen = pickle.load(open(args.stgen_cache, 'rb'))
    else:
        stgen = SynTreeGenerator(
            building_blocks=bblocks, rxn_templates=rxn_templates, verbose=args.verbose, rng=np.random.default_rng(seed=args.seed)
        )
        if args.stgen_cache:
            pickle.dump(stgen, open(args.stgen_cache, 'wb+'))
    logger.info("Successfully initialized SynTreeGenerator.")

    # Generate synthetic trees
    logger.info(f"Start generation of {args.number_syntrees} SynTrees...")
    if args.ncpu > 1:
        outcomes, syntrees = generate_mp()
    else:
        outcomes, syntrees = generate()
    result_summary = Counter(outcomes.values())
    logger.info(f"SynTree generation completed. Results: {result_summary}")

    summary_file = Path(args.output_file).parent / "results-summary.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(result_summary, indent=2))

    # Save synthetic trees on disk
    syntree_collection = SyntheticTreeSet(syntrees)
    syntree_collection.save(args.output_file)

    logger.info(f"Completed.")
