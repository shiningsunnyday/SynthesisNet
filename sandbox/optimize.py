import functools
import logging
import pathlib
import pickle
from typing import List, Literal, Optional

import jsonargparse
import numpy as np
import tqdm
from rdkit import Chem
from torch.multiprocessing import Pool

from ga.config import GeneticSearchConfig
from ga.search import GeneticSearch
from synnet.MolEmbedder import MolEmbedder
from synnet.encoding.fingerprints import mol_fp
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.encoding.distances import cosine_distance
from synnet.models.legacy import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.utils.data_utils import ReactionSet, SkeletonSet, binary_tree_to_skeleton
from synnet.utils.predict_utils import synthetic_tree_decoder, tanimoto_similarity
from synnet.utils.reconstruct_utils import (
    decode,
    load_data,
    reconstruct,
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
    embeddings_knn_file_large: str = "data/assets/building-blocks/enamine_us_emb_fp_2048.npy"

    # If given, consider only these bbs
    top_bbs_file: Optional[str] = None

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

    forcing_eval: bool = False
    mermaid: bool = False
    one_per_class: bool = False

    out_dir: Optional[str] = None

    # Beam width for first bb
    top_k: int = 3

    # Beam width for first rxn
    top_k_rxn: int = 3

    filter_only: List[Literal["rxn", "bb"]] = []

    num_workers: int = 0
    chunksize: int = 1

    # Conf: Decode all reactions before bbs. Choose highest-confidence reaction. Choose closest neighbor bb.
    # Topological: Decode every topological order of the rxn+bb nodes.
    strategy: Literal["conf", "topological"] = "conf"
    test_correct_method: Literal["preorder", "postorder", "reconstruct"] = "reconstruct"
    max_topological_orders: int = 5

    reassign_fps: bool = True
    reassign_bts: bool = True


def get_smiles_ours(idx_and_ind):
    idx, ind = idx_and_ind
    sk0 = binary_tree_to_skeleton(ind.bt)

    ans = 0.0
    best_smi = ""
    best_bt = None
    for sk in decode(sk0, ind.fp):
        score, smi, bt = reconstruct(sk, ind.fp, return_bt=True) # do stuff with bt
        if score > ans:
            ans = score
            best_smi = smi
            best_bt = bt
    return idx, best_smi, best_bt


def get_smiles_synnet(
    idx_and_ind,
    building_blocks, bb_dict,
    rxns,
    bblocks_molembedder,
    act_net, rt1_net, rxn_net, rt2_net,
    bb_emb,
    rxn_template,
    nbits
):
    idx, ind = idx_and_ind

    emb = ind.fp.reshape((1, -1))
    try:
        tree, action = synthetic_tree_decoder(
            z_target=emb,
            sk_coords=None,
            building_blocks=building_blocks,
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
        return idx, None, None
    else:
        scores = np.array(tanimoto_similarity(emb, [node.smiles for node in tree.chemicals]))
        max_score_idx = np.where(scores == np.max(scores))[0][0]
        return idx, tree.chemicals[max_score_idx].smiles, None


def test_surrogate(batch, desc, converter, pool, config: OptimizeGAConfig):
    indexed_batch = list(enumerate(batch))
    if config.num_workers <= 0:
        indexed_smiles = map(converter, indexed_batch)
    else:
        indexed_smiles = pool.imap_unordered(converter, indexed_batch, chunksize=config.chunksize)

    pbar = tqdm.tqdm(indexed_smiles, total=len(batch), desc=desc)
    for idx, smi, bt in pbar:
        ind = batch[idx]
        if smi is None:
            ind.smiles = None
        else:
            ind.smiles = Chem.CanonSmiles(smi)
            if config.reassign_fps:
                ind.fp = mol_fp(ind.smiles, _nBits=config.fp_bits).astype(np.float32)
        if config.reassign_bts:
            ind.bt = bt


def main():
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(OptimizeGAConfig, as_positional=False)
    config = OptimizeGAConfig(**parser.parse_args().as_dict())
    global args
    args = config  # Hack so reconstruct_utils.py works

    if config.log_file:
        handler = logging.FileHandler(config.log_file)
        logger.addHandler(handler)

    if config.method == "ours":
        assert config.fp_bits == 2048

        set_models(config, logger)
        load_data(config, logger)
        with open(config.skeleton_set_file, "rb") as f:
            skeletons = pickle.load(f)
        skeleton_set = SkeletonSet().load_skeletons(skeletons)
        test_skeletons(config, skeleton_set, max_rxns=config.max_num_rxns)

        converter = get_smiles_ours

    elif config.method == "synnet":
        assert config.fp_bits == 4096

        # Load the purchasable building block embeddings
        bblocks_molembedder = (
            MolEmbedder().load_precomputed(args.embeddings_knn_file).init_balltree(cosine_distance)
        )
        bb_emb = bblocks_molembedder.get_embeddings()

        # Load the purchasable building block SMILES to a dictionary
        building_blocks = BuildingBlockFileHandler().load(args.building_blocks_file)

        # A dict is used as lookup table for 2nd reactant during inference:
        bb_dict = {block: i for i, block in enumerate(building_blocks)}

        # Load the reaction templates as a ReactionSet object
        rxns = ReactionSet().load(args.rxns_collection_file).rxns

        # Load the pre-trained modules
        path = pathlib.Path(__file__).parents[1] / "data" / "checkpoints"
        # ckpt_files = [find_best_model_ckpt(path / model) for model in "act rt1 rxn rt2".split()]
        ckpt_files = [path / model / f'{model}.ckpt' for model in "act rt1 rxn rt2".split()]
        print(ckpt_files)
        act_net, rt1_net, rxn_net, rt2_net = [load_mlp_from_ckpt(file).cpu() for file in ckpt_files]

        converter = functools.partial(
            get_smiles_synnet,
            building_blocks=building_blocks,
            bb_dict=bb_dict,
            rxns=rxns,
            bblocks_molembedder=bblocks_molembedder,
            act_net=act_net, rt1_net=rt1_net, rxn_net=rxn_net, rt2_net=rt2_net,
            bb_emb=bb_emb,
            rxn_template="hb",
            nbits=config.fp_bits,
        )

    else:
        raise NotImplementedError()

    search = GeneticSearch(config)

    if config.num_workers > 0:
        pool = Pool(processes=config.num_workers)
    else:
        pool = None

    surrogate = functools.partial(test_surrogate, converter=converter, pool=pool, config=config)
    search.optimize(surrogate=surrogate)

    if pool is not None:
        pool.close()
        pool.join()

    return 0


if __name__ == "__main__":
    main()