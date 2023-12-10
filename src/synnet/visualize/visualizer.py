from pathlib import Path
from typing import Union
from collections import defaultdict

import sys
from synnet.utils.data_utils import NodeChemical, NodeRxn, SyntheticTree, SyntheticTreeSet, Skeleton
from synnet.visualize.drawers import MolDrawer, RxnDrawer
from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    ReactionTemplateFileHandler,
)
from synnet.visualize.writers import subgraph

from synnet.visualize.drawers import MolDrawer
from synnet.visualize.writers import SynTreeWriter, SkeletonPrefixWriter

import pickle
import argparse
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
import os


class SynTreeVisualizer:
    actions_taken: dict[int, str]
    CHEMICALS: dict[str, NodeChemical]
    outfolder: Union[str, Path]
    version: int

    ACTIONS = {
        0: "Add",
        1: "Expand",
        2: "Merge",
        3: "End",
    }

    def __init__(self, syntree: SyntheticTree, outfolder: str = "./syntree-viz/st"):
        self.syntree = syntree
        self.actions_taken = {
            depth: self.ACTIONS[action] for depth, action in enumerate(syntree.actions)
        }
        self.CHEMICALS = {node.smiles: node for node in syntree.chemicals}

        # Placeholder for images for molecues.
        self.drawer: Union[MolDrawer, None]
        self.molecule_filesnames: Union[None, dict[str, str]] = None

        # Folders
        outfolder = Path(outfolder)
        self.version = self._get_next_version(outfolder)
        self.path = outfolder.with_name(outfolder.name + f"_{self.version}")
        return None

    def _get_next_version(self, dir: str) -> int:
        root_dir = Path(dir).parent
        name = Path(dir).name

        existing_versions = []
        for d in Path(root_dir).glob(f"{name}_*"):
            d = str(d.resolve())
            existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    def with_drawings(self, drawer: MolDrawer):
        """Init `MolDrawer` to plot molecules in the nodes."""
        self.path.mkdir(parents=True)
        self.drawer = drawer(self.path)
        return self

    def plot(self):
        """Plots molecules via `self.drawer.plot()`."""
        if self.drawer is None:
            raise ValueError("Must initialize drawer beforehand.")
        self.drawer.plot(self.CHEMICALS)
        self.molecule_filesnames = self.drawer.get_molecule_filesnames()
        return self

    def _define_chemicals(
        self,
        chemicals: dict[str, NodeChemical] = None,
    ) -> list[str]:
        chemicals = self.CHEMICALS if chemicals is None else chemicals

        if self.drawer.outfolder is None or self.molecule_filesnames is None:
            raise NotImplementedError("Must provide drawer via `_with_drawings()` before plotting.")

        out: list[str] = []

        for node in chemicals.values():
            name = f'"node.smiles"'
            name = f'<img src=""{self.drawer.outfolder.name}/{self.molecule_filesnames[node.smiles]}.svg"" height=75px/>'
            classdef = self._map_node_type_to_classdef(node)
            info = f"n{node.index}[{name}]:::{classdef}"
            out += [info]
        return out

    def _map_node_type_to_classdef(self, node: NodeChemical) -> str:
        """Map a node to pre-defined mermaid class for styling."""
        if node.is_leaf:
            classdef = "buildingblock"
        elif node.is_root:
            classdef = "final"
        else:
            classdef = "intermediate"
        return classdef

    def _write_reaction_connectivity(
        self, reactants: list[NodeChemical], product: NodeChemical
    ) -> list[str]:
        """Write the connectivity of the graph.
        Unimolecular reactions have one edge, bimolecular two.

        Examples:
            n1 --> n3
            n2 --> n3
        """
        NODE_PREFIX = "n"
        r1, r2 = reactants
        out = [f"{NODE_PREFIX}{r1.index} --> {NODE_PREFIX}{product.index}"]
        if r2 is not None:
            out += [f"{NODE_PREFIX}{r2.index} --> {NODE_PREFIX}{product.index}"]
        return out

    def write(self) -> list[str]:
        """Write markdown with mermaid block."""
        # 1. Plot images
        self.plot()
        # 2. Write markdown (with reference to image files.)
        rxns: list[NodeRxn] = self.syntree.reactions
        text = []

        # Add node definitions
        text.extend(self._define_chemicals(self.CHEMICALS))

        # Add paragraphs (<=> actions taken)
        for i, action in self.actions_taken.items():
            if action == "End":
                continue
            rxn = rxns[i]
            product: str = rxn.parent
            reactant1: str = rxn.child[0]
            reactant2: str = rxn.child[1] if rxn.rtype == 2 else None

            @subgraph(f'"{i:>2d} : {action}"')
            def __printer():
                return self._write_reaction_connectivity(
                    [self.CHEMICALS.get(reactant1), self.CHEMICALS.get(reactant2)],
                    self.CHEMICALS.get(product),
                )

            out = __printer()
            text.extend(out)
        return text

class SkeletonVisualizer:
    outfolder: Union[str, Path]

    def __init__(self, skeleton: Skeleton, outfolder: str = "./syntree-viz/st"):
        self.skeleton = skeleton
        # Placeholder for images for molecues.
        self.mol_drawer: Union[MolDrawer, None]
        self.rxn_drawer: Union[RxnDrawer, None]
        self.molecule_filesnames: Union[None, dict[str, str]] = None
        self.reaction_filenames: Union[None, dict[str, str]] = None        

        # Folders
        outfolder = Path(outfolder)
        self.version = self._get_next_version(outfolder)
        self.path = outfolder.with_name(outfolder.name + f"_{self.version}")
        return None

    def _get_next_version(self, dir: str) -> int:
        root_dir = Path(dir).parent
        name = Path(dir).name

        existing_versions = []
        for d in Path(root_dir).glob(f"{name}_*"):
            d = str(d.resolve())
            existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    def with_drawings(self, mol_drawer: MolDrawer, rxn_drawer: RxnDrawer):
        """Init `MolDrawer` or `RxnDrawer` to plot molecules or rxns in the nodes."""
        self.path.mkdir(parents=True)
        self.mol_drawer = mol_drawer(self.path)
        self.rxn_drawer = rxn_drawer(self.path)
        return self

    def plot(self):
        """Plots molecules and reactions via `self.drawer.plot()`."""
        if self.mol_drawer is None:
            raise ValueError("Must initialize drawer beforehand.")
        if self.rxn_drawer is None:
            raise ValueError("Must initialize drawer beforehand.")        
 
        smiles = []
        smirks = []
        tree = self.skeleton.tree
        for n in tree.nodes():
            if 'smiles' in tree.nodes[n]:
                smiles.append(tree.nodes[n]['smiles'])
            else:
                assert 'rxn_id' in tree.nodes[n]
                assert 'smirks' in tree.nodes[n]
                smirks.append(tree.nodes[n]['smirks'])
        self.mol_drawer.plot(smiles)
        self.rxn_drawer.plot(smirks)
        self.molecule_filesnames = self.mol_drawer.get_molecule_filesnames()
        self.reaction_filesnames = self.rxn_drawer.get_reaction_filesnames()
        return self

    def _define_chemicals_and_reactions(
        self, node_mask=None
    ) -> list[str]:

        if self.mol_drawer.outfolder is None or self.molecule_filesnames is None:
            raise NotImplementedError("Must provide drawer via `_with_drawings()` before plotting.")
        if self.rxn_drawer.outfolder is None or self.reaction_filesnames is None:
            raise NotImplementedError("Must provide drawer via `_with_drawings()` before plotting.")

        out: list[str] = []
        tree = self.skeleton.tree
        for node in tree:
            if 'smiles' in tree.nodes[node]:
                name = f'"node.smiles"'
                fname = self.molecule_filesnames[tree.nodes[node]['smiles']]
                fname += ".svg"
            else: 
                name = f'"node.rxn"'
                fname = self.reaction_filesnames[tree.nodes[node]['smirks']]
                fname += ".png"
            assert self.mol_drawer.outfolder.name == self.rxn_drawer.outfolder.name
            if node_mask[node]:
                name = f'<img src=""{self.mol_drawer.outfolder.name}/{fname}"" height=75px/>'
            else:
                name = f'<div height=75px/>'
            classdef = self._map_node_type_to_classdef(tree, node)
            info = f"n{node}[{name}]:::{classdef}"
            out += [info]
        return out

    def _map_node_type_to_classdef(self, tree, node) -> str:
        """Map a node to pre-defined mermaid class for styling."""        
        if 'rxn_id' in tree.nodes[node]:
            classdef = "reaction"
        elif list(tree.successors(node)) == []:
            classdef = "buildingblock"
        elif list(tree.predecessors(node)) == []:
            classdef = "final"
        else:
            classdef = "intermediate"
        return classdef

    def _write_reaction_connectivity(
        self, reactants, product
    ) -> list[str]:
        """Write the connectivity of the graph.
        Unimolecular reactions have one edge, bimolecular two.

        Examples:
            n1 --> n3
            n2 --> n3
        """
        NODE_PREFIX = "n"
        r1, r2 = reactants
        out = [f"{NODE_PREFIX}{r1} --> {NODE_PREFIX}{product}"]
        if r2:
            out += [f"{NODE_PREFIX}{r2} --> {NODE_PREFIX}{product}"]
        return out

    def write(self, node_mask=None) -> list[str]:
        """Write markdown with mermaid block."""
        # 1. Plot images
        self.plot()
        # 2. Write markdown (with reference to image files.)
        text = []

        # Add node definitions        
        text.extend(self._define_chemicals_and_reactions(node_mask))

        tree = self.skeleton.tree
        assert list(tree.predecessors(self.skeleton.tree_root)) == []
        order = nx.dfs_postorder_nodes(tree, self.skeleton.tree_root)
        # Add paragraphs (<=> actions taken)
        for (i, node) in enumerate(order):
            if not list(tree.successors(node)):
                continue
            succ = list(tree.successors(node))
            if len(succ) == 2:
                reactant1, reactant2 = succ
            elif len(succ) == 1:
                reactant1, reactant2 = succ[0], None
            else:
                raise
            @subgraph(f'"{i:>2d} : {node}"')
            def __printer():
                return self._write_reaction_connectivity(
                    [reactant1, reactant2], node
                )

            out = __printer()
            text.extend(out)
        return text


def demo():
    """Demo syntree visualisation"""
    # 1. Load syntree
    import json

    infile = "tests/assets/syntree-small.json"
    with open(infile, "rt") as f:
        data = json.load(f)

    st = SyntheticTree()
    st.read(data)

    outpath = Path("./figures/syntrees/generation/st")
    outpath.mkdir(parents=True, exist_ok=True)

    # 2. Plot & Write mermaid markup diagram
    stviz = SynTreeVisualizer(syntree=st, outfolder=outpath).with_drawings(drawer=MolDrawer)
    mermaid_txt = stviz.write()
    # 3. Write everything to a markdown doc
    outfile = stviz.path / "syntree.md"
    SynTreeWriter().write(mermaid_txt).to_file(outfile)
    print(f"Generated markdown file.")
    print(f"  Input file:", infile)
    print(f"  Output file:", outfile)
    return None


if __name__ == "__main__":
    # demo()
    parser = argparse.ArgumentParser()
    parser.add_argument('--syntree_json')
    parser.add_argument('--rxn-templates-file', help="to visualize reactions")
    parser.add_argument(
        "--skeleton-file",
        type=str,
        default="results/viz/skeletons.pkl",
        help="Input file for the skeletons of syntree-file",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="/ssd/msun415/gnn_featurized_target/",
        help="Directory of featurized targets",
    )   
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Directory of experiment metrics, compatible with skeleton classes",
    )       
    parser.add_argument('--out_folder')
    args = parser.parse_args()

    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)

    # syntree_collection = SyntheticTreeSet().load(args.syntree_json)
    # for st in syntree_collection:
    #     if st == None:
    #         breakpoint()
    #         continue
    #     breakpoint()
    #     stviz = SynTreeVisualizer(syntree=st, outfolder=args.out_folder).with_drawings(drawer=MolDrawer)
    #     mermaid_txt = stviz.write()        
    #     outfile = stviz.path / f"syntree.md"
    #     SynTreeWriter().write(mermaid_txt).to_file(outfile)
    #     print(f"Generated markdown file.", outfile)
        
    # for st in syntree_collection:
    #     print(st.root.smiles)


    skeletons = pickle.load(open(args.skeleton_file, 'rb'))
    for index in range(len(list(skeletons))):
        syntree = list(skeletons)[index]        
        sk = Skeleton(syntree, index=index)
        fpaths = defaultdict(dict)
        for f in os.listdir(args.features):
            if not f.endswith('node_masks.npz'):
                continue
            ind, batch_num, *pargs = f.split('_')
            fpath = os.path.join(args.features, f)
            fpaths[int(ind)][int(batch_num)] = fpath

        node_masks = sparse.load_npz(fpaths[index][0]).toarray()
        rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)
        for n in sk.tree: # add reactions
            if 'rxn_id' in sk.tree.nodes[n]:
                sk.tree.nodes[n]['smirks'] = rxn_templates[sk.tree.nodes[n]['rxn_id']]
        skviz = SkeletonVisualizer(skeleton=sk, outfolder=args.out_folder).with_drawings(mol_drawer=MolDrawer, rxn_drawer=RxnDrawer)
        if args.metrics:
            df = pd.read_csv(args.metrics)

        for i in range(node_masks.shape[0]//10):
            node_mask = node_masks[i]
            key = f'{index}'+''.join(list(map(str, node_mask)))                
            suffix = key
            if f'val_nn_accuracy_{key}' in df:
                bb_acc = df[f'val_nn_accuracy_{key}']
                bb_acc_best = bb_acc[bb_acc==bb_acc].max()
                suffix += f"_bb_acc={bb_acc_best}"
            if f'val_accuracy_{key}' in df:
                rxn_acc = df[f'val_accuracy_{key}']
                rxn_acc_best = rxn_acc[rxn_acc==rxn_acc].max()
                suffix += f"_rxn_acc={rxn_acc_best}"
            if f'val_cross_entropy_loss_{key}' in df:
                ce_loss = df[f'val_cross_entropy_loss_{key}']
                ce_loss_best = ce_loss[ce_loss==ce_loss].min()        
                suffix += f"_ce_loss={ce_loss_best}"
            mermaid_txt = skviz.write(node_mask=node_mask)
            outfile = skviz.path / f"skeleton_{suffix}.md"
            SynTreeWriter(prefixer=SkeletonPrefixWriter()).write(mermaid_txt).to_file(outfile)
            print(f"Generated markdown file.", outfile)

        
