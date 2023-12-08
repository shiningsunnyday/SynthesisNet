"""
Here we define the following classes for working with synthetic tree data:
* `Reaction`
* `ReactionSet`
* `NodeChemical`
* `NodeRxn`
* `SyntheticTree`
* `SyntheticTreeSet`
"""
import functools
import gzip
import itertools
import json
import numpy as np
import random
from typing import Any, Optional, Set, Tuple, Union
import multiprocessing as mp
mp.set_start_method('fork')
from synnet.config import MP_MIN_COMBINATIONS, MAX_PROCESSES
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm
import pickle
from itertools import permutations, product
from collections import defaultdict
from networkx.algorithms.isomorphism import rooted_tree_isomorphism
from networkx.algorithms import weisfeiler_lehman_graph_hash
import networkx as nx
from sklearn.manifold import MDS
from zss import Node as ZSSNode, simple_distance
from copy import deepcopy

from synnet.encoding.fingerprints import fp_2048, fp_256


# the definition of reaction classes below
class Reaction:
    """
    This class models a chemical reaction based on a SMARTS transformation.

    Args:
        template (str): SMARTS string representing a chemical reaction.
        rxnname (str): The name of the reaction for downstream analysis.
        smiles: (str): A reaction SMILES string that macthes the SMARTS pattern.
        reference (str): Reference information for the reaction.
    """

    smirks: str  # SMARTS pattern
    rxn: Chem.rdChemReactions.ChemicalReaction
    num_reactant: int
    num_agent: int
    num_product: int
    reactant_template: Tuple[str, str]
    product_template: str
    agent_template: str
    available_reactants: Tuple[list[str], Optional[list[str]]]
    rxnname: str
    smiles: Any
    reference: Any

    def __init__(self, template=None, rxnname=None, smiles=None, reference=None):

        if template is not None:
            # define a few attributes based on the input
            self.smirks = template.strip()
            self.rxnname = rxnname
            self.smiles = smiles
            self.reference = reference

            # compute a few additional attributes
            self.rxn = self.__init_reaction(self.smirks)

            # Extract number of ...
            self.num_reactant = self.rxn.GetNumReactantTemplates()
            if self.num_reactant not in (1, 2):
                raise ValueError("Reaction is neither uni- nor bi-molecular.")
            self.num_agent = self.rxn.GetNumAgentTemplates()
            self.num_product = self.rxn.GetNumProductTemplates()

            # Extract reactants, agents, products
            reactants, agents, products = self.smirks.split(">")

            if self.num_reactant == 1:
                self.reactant_template = list((reactants,))
            else:
                self.reactant_template = list(reactants.split("."))
            self.product_template = products
            self.agent_template = agents
        else:
            self.smirks = None

    def __init_reaction(self, smirks: str) -> Chem.rdChemReactions.ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = AllChem.ReactionFromSmarts(smirks)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn

    def load(
        self,
        smirks,
        num_reactant,
        num_agent,
        num_product,
        reactant_template,
        product_template,
        agent_template,
        available_reactants,
        rxnname,
        smiles,
        reference,
    ):
        """
        This function loads a set of elements and reconstructs a `Reaction` object.
        """
        self.smirks = smirks
        self.num_reactant = num_reactant
        self.num_agent = num_agent
        self.num_product = num_product
        self.reactant_template = list(reactant_template)
        self.product_template = product_template
        self.agent_template = agent_template
        self.available_reactants = list(available_reactants)  # TODO: use Tuple[list,list] here
        self.rxnname = rxnname
        self.smiles = smiles
        self.reference = reference
        self.rxn = self.__init_reaction(self.smirks)
        return self

    @functools.lru_cache(maxsize=20)
    def get_mol(self, smi: Union[str, Chem.Mol]) -> Chem.Mol:
        """
        A internal function that returns an `RDKit.Chem.Mol` object.

        Args:
            smi (str or RDKit.Chem.Mol): The query molecule, as either a SMILES
                string or an `RDKit.Chem.Mol` object.

        Returns:
            RDKit.Chem.Mol
        """
        if isinstance(smi, str):
            return Chem.MolFromSmiles(smi)
        elif isinstance(smi, Chem.Mol):
            return smi
        else:
            raise TypeError(f"{type(smi)} not supported, only `str` or `rdkit.Chem.Mol`")

    def visualize(self, name="./reaction1_highlight.o.png"):
        """
        A function that plots the chemical translation into a PNG figure.
        One can use "from IPython.display import Image ; Image(name)" to see it
        in a Python notebook.

        Args:
            name (str): The path to the figure.

        Returns:
            name (str): The path to the figure.
        """
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        d2d = Draw.MolDraw2DCairo(800, 300)
        d2d.DrawReaction(rxn, highlightByReactant=True)
        png = d2d.GetDrawingText()
        open(name, "wb+").write(png)
        del rxn
        return name

    def is_reactant(self, smi: Union[str, Chem.Mol]) -> bool:
        """Checks if `smi` is a reactant of this reaction."""
        smi = self.get_mol(smi)
        return self.rxn.IsMoleculeReactant(smi)

    def is_agent(self, smi: Union[str, Chem.Mol]) -> bool:
        """Checks if `smi` is an agent of this reaction."""
        smi = self.get_mol(smi)
        return self.rxn.IsMoleculeAgent(smi)

    def is_product(self, smi):
        """Checks if `smi` is a product of this reaction."""
        smi = self.get_mol(smi)
        return self.rxn.IsMoleculeProduct(smi)

    def is_reactant_first(self, smi: Union[str, Chem.Mol]) -> bool:
        """Check if `smi` is the first reactant in this reaction"""
        mol = self.get_mol(smi)
        if mol is None:
            return False
        pattern = Chem.MolFromSmarts(self.reactant_template[0])
        return mol.HasSubstructMatch(pattern)

    def is_reactant_second(self, smi: Union[str, Chem.Mol]) -> bool:
        """Check if `smi` the second reactant in this reaction"""
        mol = self.get_mol(smi)
        if mol is None:
            return False
        pattern = Chem.MolFromSmarts(self.reactant_template[1])
        return mol.HasSubstructMatch(pattern)

    def run_reaction(
        self, reactants: Tuple[Union[str, Chem.Mol, None]], keep_main: bool = True
    ) -> Union[str, None]:
        """Run this reactions with reactants and return corresponding product.

        Args:
            reactants (tuple): Contains SMILES strings for the reactants.
            keep_main (bool): Return main product only or all possibel products. Defaults to True.

        Returns:
            uniqps: SMILES string representing the product or `None` if not reaction possible
        """
        # Input validation.
        if not isinstance(reactants, tuple):
            raise TypeError(f"Unsupported type '{type(reactants)}' for `reactants`.")
        if not len(reactants) in (1, 2):
            raise ValueError(f"Can only run reactions with 1 or 2 reactants, not {len(reactants)}.")

        rxn = self.rxn  # TODO: investigate if this is necessary (if not, delete "delete rxn below")

        # Convert all reactants to `Chem.Mol`
        r: Tuple = tuple(self.get_mol(smiles) for smiles in reactants if smiles is not None)

        if self.num_reactant == 1:
            if len(r) == 2:  # Provided two reactants for unimolecular reaction -> no rxn possible
                return None
            if not self.is_reactant(r[0]):
                return None
        elif self.num_reactant == 2:
            # Match reactant order with reaction template
            if self.is_reactant_first(r[0]) and self.is_reactant_second(r[1]):
                pass
            elif self.is_reactant_first(r[1]) and self.is_reactant_second(r[0]):
                r = tuple(reversed(r))
            else:  # No reaction possible
                return None
        else:
            raise ValueError("This reaction is neither uni- nor bi-molecular.")

        # Run reaction with rdkit magic
        ps = rxn.RunReactants(r)

        # Filter for unique products (less magic)
        # Note: Use chain() to flatten the tuple of tuples
        uniqps = list({Chem.MolToSmiles(p) for p in itertools.chain(*ps)})

        # Sanity check
        if not len(uniqps) >= 1:
            # TODO: Raise (custom) exception?
            raise ValueError("Reaction did not yield any products.")

        del rxn

        if keep_main:
            uniqps = uniqps[:1]
        # >>> TODO: Always return list[str] (currently depends on "keep_main")
        uniqps = uniqps[0]
        # <<< ^ delete this line if resolved.
        return uniqps

    def  _filter_reactants(
        self, smiles: list[str], verbose: bool = False
    ) -> Tuple[list[str], list[str]]:
        """
        Filters reactants which do not match the reaction.

        Args:
            smiles: Possible reactants for this reaction.

        Returns:
            :lists of SMILES which match either the first
                reactant, or, if applicable, the second reactant.

        Raises:
            ValueError: If `self` is not a uni- or bi-molecular reaction.
        """
        smiles = tqdm(smiles) if verbose else smiles

        if self.num_reactant == 1:  # uni-molecular reaction
            reactants_1 = [smi for smi in smiles if self.is_reactant_first(smi)]
            return (reactants_1,)

        elif self.num_reactant == 2:  # bi-molecular reaction
            reactants_1 = [smi for smi in smiles if self.is_reactant_first(smi)]
            reactants_2 = [smi for smi in smiles if self.is_reactant_second(smi)]

            return (reactants_1, reactants_2)
        else:
            raise ValueError("This reaction is neither uni- nor bi-molecular.")

    def set_available_reactants(self, building_blocks: list[str], verbose: bool = False):
        """
        Finds applicable reactants from a list of building blocks.
        Sets `self.available_reactants`.

        Args:
            building_blocks: Building blocks as SMILES strings.
        """
        self.available_reactants = self._filter_reactants(building_blocks, verbose=verbose)
        return self

    @property
    def get_available_reactants(self) -> Set[str]:
        return {x for reactants in self.available_reactants for x in reactants}

    def asdict(self) -> dict():
        """Returns serializable fields as new dictionary mapping.
        *Excludes* Not-easily-serializable `self.rxn: rdkit.Chem.ChemicalReaction`."""
        import copy

        out = copy.deepcopy(self.__dict__)  # TODO:
        _ = out.pop("rxn")
        return out



class Program:
    def __init__(self, rxn_tree=None, keep_prods=0):
        self.rxn_tree = rxn_tree if rxn_tree is not None else nx.DiGraph()        
        self._entries = [n for n in self.rxn_tree if list(self.rxn_tree.successors(n)) == []] 
        self.rxn_map = {}
        self.keep_prods = keep_prods
        if keep_prods:
            """
            map from node -> {interm: entry: {[index1, index2]}}
            for each node n, store its intermediates
            for each intermediate, store the entry nodes
            for each entry, store the indices in .available_reactants
            """
            self.product_map = defaultdict(self.make_default_dict)
            # assert 'depth' in self.rxn_tree.graph


    @property
    def entries(self):
        return self._entries

    
    @staticmethod
    def make_default_dict():
        return defaultdict(dict)  
    
    
    @staticmethod
    def input_length(p):
        react = []
        for e in p.entries:
            num = np.prod([len(ar) for ar in p.rxn_map[e].available_reactants])
            react.append(num)
        return np.prod(react)
    

    @staticmethod
    def avg_input_length(progs):
        num_poss = []
        for p in progs:
            num_poss.append(Program.input_length(p))
        return np.mean(num_poss) if len(num_poss) else 0


    def hash(self, mask, return_json=False):
        # used to hash the partial state defined by mask
        # can also return the tree data defined by mask
        data = {}
        for n in self.rxn_tree.nodes():
            if not mask[n]:
                data[n] = self.rxn_tree.nodes[n]['rxn_id']
                self.rxn_tree.nodes[n].pop('rxn_id')
        json_data = nx.tree_data(self.rxn_tree, len(self.rxn_tree)-1)
        ans = hash(json.dumps(json_data))
        for n, r in data.items():
            self.rxn_tree.nodes[n]['rxn_id'] = r
        if return_json:
            return json_data        
        return str(ans)


    def output_dict(self):
        breakpoint()
    
    

    def add_rxn(self, id, left, right=None):
        assert left in self.rxn_tree.nodes()
        assert right is None or right in self.rxn_tree.nodes()
        key = len(self.rxn_tree)
        self.rxn_tree.add_node(key, rxn_id=id)                
        self.rxn_tree.add_edge(key, left)
        self.rxn_tree.nodes[left]['child'] = 'left'
        depth = self.rxn_tree.nodes[left]['depth'] +1
        if right:
            self.rxn_tree.nodes[right]['child'] = 'right'
            depth = max(depth, self.rxn_tree.nodes[right]['depth'] +1)
            self.rxn_tree.add_edge(key, right)
        self.rxn_tree.nodes[key]['depth'] = depth
    

    def combine(self, other):    
        offset = len(self.rxn_tree)
        self.rxn_tree = nx.disjoint_union(self.rxn_tree, other.rxn_tree)
        self._entries = [n for n in self.rxn_tree if list(self.rxn_tree.successors(n)) == []]                 
        for n in other.rxn_map:
            self.rxn_map[n+offset] = other.rxn_map[n]
        # combine rxn_map's
        if self.keep_prods:
            for n in other.product_map:
                for r in other.product_map[n]:
                    for e, v in other.product_map[n][r].items():
                        self.product_map[n+offset][r][e+offset] = v
            # combine self.product_map's and correct for entries
        return self
    

    def init_rxns(self, rxns):
        for n in self.rxn_tree.nodes():
            if n not in self.rxn_map:                
                rxn = deepcopy(rxns[self.rxn_tree.nodes[n]['rxn_id']])
                assert rxn.available_reactants is not None
                self.rxn_map[n] = rxn
                
                if self.keep_prods <= self.rxn_tree.nodes[n]['depth']:
                    """
                    n is a new reaction node, but we can filter the entry reactants using the interms
                    """
                    prod_before = [[len(avail_reactants) for avail_reactants in self.rxn_map[n].available_reactants] for n in self.entries]
                    # print([[avail_reactants for avail_reactants in self.rxn_map[n].available_reactants] for n in self.entries])

                    for succ in self.rxn_tree.successors(n):
                        """
                        for each child of n, we can define a filter function depending on if it's the left or right child
                        left also includes uni-molecular reaction
                        """
                        if self.rxn_tree.nodes[succ]['child'] == 'left':
                            filter_func = rxn.is_reactant_first 
                        else:
                            filter_func = rxn.is_reactant_second

                        res = []    
                        num_interms = len(self.product_map[succ])
                        bad_interms = []                
                        """
                        product_map stores the interms at n's successor
                        """
                        for interm in self.product_map[succ]:
                            if filter_func(interm):
                                if list(self.product_map[succ][interm].keys()) != self.entries: # same order
                                    # make sure appear in same order
                                    appear_entries = [self.entries.index(p) for p in list(self.product_map[succ][interm].keys())]
                                    assert sorted(appear_entries) == appear_entries                        
                                entry_reactants = [[self.rxn_map[e].available_reactants[i][ind] for (i, ind) in enumerate(self.product_map[succ][interm][e])] for e in self.product_map[succ][interm]]
                                res.append((interm, entry_reactants))
                            else:
                                bad_interms.append(interm)            
                        
                        entries = list(self.product_map[succ][interm].keys())                    
                        for n in entries:
                            cur = self.rxn_map[n].available_reactants
                            self.rxn_map[n].available_reactants = tuple([] for _ in cur)                    

                        """
                        for each entry n, store the available_reactants that are in entry_reactants
                        """  
                        avail_index = {n: [{}, {}] for n in entries}
                        for _, entry_reactants in res:            
                            for entry_reactant, n in zip(entry_reactants, entries):
                                for i, reactant in enumerate(entry_reactant):
                                    if reactant not in avail_index[n][i]:                        
                                        avail_index[n][i][reactant] = len(self.rxn_map[n].available_reactants[i]) # store the index
                                        self.rxn_map[n].available_reactants[i].append(reactant)
                        
                        # remove the bad interms
                        for interm in bad_interms:
                            self.product_map[succ].pop(interm)                    
                        # print(f"{num_interms}->{num_interms-len(bad_interms)} node {n} interm prods")

                    
                    prod_after = [[len(avail_reactants) for avail_reactants in self.rxn_map[n].available_reactants] for n in self.entries]
                    # print(f"pruned avail reactants from {prod_before}->{prod_after} using interms")
                    # print([[avail_reactants for avail_reactants in self.rxn_map[n].available_reactants] for n in self.entries])
                    # change available_reactants to be valid products of previous            
                    # propagate down the tree    

    @staticmethod
    def run_rxns(i):
        entry_reactants = all_entry_reactants[i]
        good = True
        product_map = {}
        for node in nx.dfs_postorder_nodes(rxn_tree, len(rxn_tree)-1):
            if node in entries:
                entry = entries.index(node)
                prod = rxn_map[node].run_reaction(entry_reactants[entry])
                product_map[node] = prod
            else:
                reactants = []
                succ = list(rxn_tree.successors(node))
                assert len(succ) in [1, 2]
                if len(succ) == 2 and rxn_tree.nodes[succ[-1]]['child'] == 'left':
                    succ = succ[::-1]
                for i, n in enumerate(succ):     
                    if i == 0:
                        if rxn_map[node].is_reactant_first(product_map[n]):
                            reactants.append(product_map[n])
                        else: 
                            good = False
                    if i == 1:
                        if rxn_map[node].is_reactant_second(product_map[n]):
                            reactants.append(product_map[n])
                        else: 
                            good = False                            
                    if not good:
                        break                        
                if good:
                    product_map[node] = rxn_map[node].run_reaction(tuple(reactants))
                else:
                    breakpoint()
                    break
        if good:
            return product_map[len(rxn_tree)-1]
        else:          
            return None             

    

    def run_rxn_tree(self): 
        # Assume entry inputs satisfy everything except the "root"        
        reactant_map = {}
        # if self.rxn_tree.graph['super']:
        #     breakpoint()
        for n in self.entries:
            reactant_map[n] = product(*[reactants for reactants in self.rxn_map[n].available_reactants])
        all_entry_reactants = list(product(*[reactant_map[n] for n in self.entries]))
        res = []
        count = len(list(all_entry_reactants))
        if count == 0:            
            return 0, None
        
        rxn_tree = self.rxn_tree
        rxn_map = self.rxn_map
        entries = self.entries

        globals()["all_entry_reactants"] = all_entry_reactants
        globals()["rxn_tree"] = rxn_tree
        globals()["rxn_map"] = rxn_map
        globals()["entries"] = entries

        if count >= MP_MIN_COMBINATIONS:
            with mp.Pool(MAX_PROCESSES) as p:
                res = p.map(self.run_rxns, range(len(all_entry_reactants)))           
                assert len(res) == len(all_entry_reactants)
        else:
            res = [self.run_rxns(i) for i in range(len(all_entry_reactants))]
        res = [(r, entry_reactants) for (r, entry_reactants) in zip(res, all_entry_reactants) if r is not None]

        # Update the reactants to only valid inputs
        for n in self.entries:
            cur = self.rxn_map[n].available_reactants
            self.rxn_map[n].available_reactants = tuple([] for _ in cur)

        avail_index = {n: [{}, {}] for n in self.entries}  # sets of possible first and second reactants per entry
        for r, entry_reactants in res:            
            for entry_reactant, n in zip(entry_reactants, self.entries):
                entry_point = []
                for i, reactant in enumerate(entry_reactant):
                    if reactant not in avail_index[n][i]:                        
                        avail_index[n][i][reactant] = len(self.rxn_map[n].available_reactants[i])
                        self.rxn_map[n].available_reactants[i].append(reactant)
                    if self.keep_prods:
                        entry_point.append(avail_index[n][i][reactant])
                if self.keep_prods:
                    self.product_map[len(self.rxn_tree)-1][r][n] = entry_point
        return count, res
                        



class ReactionSet:
    """Represents a collection of reactions, for saving and loading purposes."""

    def __init__(self, rxns: Optional[list[Reaction]] = None):
        self.rxns = rxns if rxns is not None else []

    def load(self, file: str):
        """Load a collection of reactions from a `*.json.gz` file."""
        assert str(file).endswith(".json.gz"), f"Incompatible file extension for file {file}"
        with gzip.open(file, "r") as f:
            data = json.loads(f.read().decode("utf-8"))

        for r in data["reactions"]:
            rxn = Reaction().load(
                **r
            )  # TODO: `load()` relies on postional args, hence we cannot load a reaction that has no `available_reactants` for extample (or no template)
            self.rxns.append(rxn)
        return self

    def save(self, file: str) -> None:
        """Save a collection of reactions to a `*.json.gz` file."""

        assert str(file).endswith(".json.gz"), f"Incompatible file extension for file {file}"

        r_list = {"reactions": [r.asdict() for r in self.rxns]}
        with gzip.open(file, "w") as f:
            f.write(json.dumps(r_list).encode("utf-8"))

    def __len__(self):
        return len(self.rxns)

    def _print(self, x=3):
        # For debugging
        for i, r in enumerate(self.rxns):
            if i >= x:
                break
            print(json.dumps(r.asdict(), indent=2))


# the definition of classes for defining synthetic trees below
class NodeChemical:
    """Represents a chemical node in a synthetic tree.

    Args:
        smiles: Molecule represented as SMILES string.
        parent: Parent molecule represented as SMILES string (i.e. the result of a reaction)
        child: Index of the reaction this object participates in.
        is_leaf: Is this a leaf node in a synthetic tree?
        is_root: Is this a root node in a synthetic tree?
        depth: Depth this node is in tree (+1 for an action, +.5 for a reaction)
        index: Incremental index for all chemical nodes in the tree.
    """

    def __init__(
        self,
        smiles: Union[str, None] = None,
        parent: Union[int, None] = None,
        child: Union[int, None] = None,
        is_leaf: bool = False,
        is_root: bool = False,
        depth: float = 0,
        index: int = 0,
    ):
        self.smiles = smiles
        self.parent = parent
        self.child = child
        self.is_leaf = is_leaf
        self.is_root = is_root
        self.depth = depth
        self.index = index


class Node:
    """Represents a chemical node in a synthetic tree
    """

    def __init__(
        self,
        smiles: Union[str, None] = None,
        parent = None,
        rxn_id: Union[int, None] = None,
        rtype: Union[int, None] = None,        
        child : Union[list, None] = [],
        is_leaf: bool = False,
        is_root: bool = False     
    ):
        self.smiles = smiles
        self.parent = parent
        self.rxn_id = rxn_id,
        self.rtype = rtype
        self.child = child
        self.is_leaf = is_leaf
        self.is_root = is_root   


class NodeRxn:
    """Represents a chemical reaction in a synthetic tree.


    Args:
        rxn_id (None or int): Index corresponding to reaction in a one-hot vector
            of reaction templates.
        rtype (None or int): Indicates if uni- (1) or bi-molecular (2) reaction.
        parent (None or list):
        child (None or list): Contains SMILES strings of reactants which lead to
            the specified reaction.
        depth (float):
        index (int): Indicates the order of this reaction node in the tree.
    """

    def __init__(
        self,
        rxn_id: Union[int, None] = None,
        rtype: Union[int, None] = None,
        parent: Union[list, None] = [],
        child: Union[list, None] = None,
        depth: float = 0,
        index: int = 0,
    ):
        self.rxn_id = rxn_id
        self.rtype = rtype
        self.parent = parent
        self.child = child
        self.depth = depth
        self.index = index


class SyntheticTree:
    """
    A class representing a synthetic tree.

    Args:
        chemicals (list): A list of chemical nodes, in order of addition.
        reactions (list): A list of reaction nodes, in order of addition.
        actions (list): A list of actions, in order of addition.
        root (NodeChemical): The root node.
        depth (int): The depth of the tree.
        rxn_id2type (dict): A dictionary that maps reaction indices to reaction
            type (uni- or bi-molecular).
    """

    def __init__(self, tree=None):
        self.chemicals: list[NodeChemical] = []
        self.reactions: list[NodeRxn] = []
        self.edges = []
        self.root = None
        self.depth: float = 0
        self.actions = []
        self.rxn_id2type = None

        if tree is not None:
            self.read(tree)

    def read(self, data):
        """
        A function that loads a dictionary from synthetic tree data.

        Args:
            data (dict): A dictionary representing a synthetic tree.
        """
        self.root = NodeChemical(**data["root"])
        self.depth = data["depth"]
        self.actions = data["actions"]
        self.rxn_id2type = data["rxn_id2type"]
        if "edges" in data:
            self.edges = data["edges"]

        for r_dict in data["reactions"]:
            r = NodeRxn(**r_dict)
            self.reactions.append(r)

        for m_dict in data["chemicals"]:
            r = NodeChemical(**m_dict)
            self.chemicals.append(r)

    def output_dict(self):
        """
        A function that exports dictionary-formatted synthetic tree data.

        Returns:
            data (dict): A dictionary representing a synthetic tree.
        """
        return {
            "reactions": [r.__dict__ for r in self.reactions],
            "chemicals": [m.__dict__ for m in self.chemicals],
            "root": self.root.__dict__,
            "depth": self.depth,
            "actions": self.actions,
            "rxn_id2type": self.rxn_id2type,
            "edges": self.edges
        }


    def is_isomorphic(self, other):
        def get_parent(node):
            if node.parent: return node.parent[0]
            return None
        
        def try_perm(perm, self_nodes, other_nodes):
            other_nodes = [other_nodes[perm[i]] for i in range(len(perm))]
            for i in range(len(self_nodes)):
                for j in range(len(other_nodes)):
                    if (get_parent(self_nodes[i]) == self_nodes[j]) ^ (get_parent(other_nodes[i]) == other_nodes[j]):
                        return False
                    if (get_parent(self_nodes[j]) == self_nodes[i]) ^ (get_parent(other_nodes[j]) == other_nodes[i]):
                        return False 
            return True       
        

        if len(self.nodes) != len(other.nodes): return False


        self_nodes = self.nodes
        other_nodes = other.nodes    
        self_tree = nx.DiGraph(self.edges)
        other_tree = nx.DiGraph(other.edges)
        return rooted_tree_isomorphism(self_tree, len(self_tree)-1, other_tree, len(other_tree)-1)
        # for perm in permutations(range(len(other.nodes))):

        #     if try_perm(perm, self_nodes, other_nodes): 
        #         return True
                    



    def _print(self):
        """
        A function that prints the contents of the synthetic tree.
        """
        print("===============Stored Molecules===============")
        for node in self.chemicals:
            print(node.smiles, node.is_root)
        print("===============Stored Reactions===============")
        for node in self.reactions:
            print(node.rxn_id, node.rtype)
        print("===============Followed Actions===============")
        print(self.actions)

    def get_node_index(self, smi):
        """
        Returns the index of the node matching the input SMILES.

        Args:
            smi (str): A SMILES string that represents the query molecule.

        Returns:
            index (int): Index of chemical node corresponding to the query
                molecule. If the query moleucle is not in the tree, return None.
        """
        for node in self.chemicals:
            if smi == node.smiles:
                return node.index
        return None

    def get_state(self) -> list[str]:
        """Get the state of this synthetic tree.
        The most recent root node has 0 as its index.

        Returns:
            state (list): A list contains all root node molecules.
        """
        state = [node.smiles for node in self.chemicals if node.is_root]
        return state[::-1]

    def update(self, action: int, rxn_id: int, mol1: str, mol2: str, mol_product: str):
        """Update this synthetic tree by adding a reaction step.

        Args:
            action (int): Action index, where the indices (0, 1, 2, 3) represent
                (Add, Expand, Merge, and End), respectively.
            rxn_id (int): Index of the reaction occured, where the index can be
               anything in the range [0, len(template_list)-1].
            mol1 (str): SMILES string representing the first reactant.
            mol2 (str): SMILES string representing the second reactant.
            mol_product (str): SMILES string representing the product.
        """
        self.actions.append(int(action))

        if action == 3:  # End
            self.root = self.chemicals[-1]
            self.depth = self.root.depth

        elif action == 2:  # Merge (with bi-mol rxn)
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_mol2 = self.chemicals[self.get_node_index(mol2)]
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=2,
                parent=None,
                child=[node_mol1.smiles, node_mol2.smiles],
                depth=max(node_mol1.depth, node_mol2.depth) + 0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=node_rxn.depth + 0.5,
                index=len(self.chemicals),
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id
            node_mol1.is_root = False
            node_mol2.is_root = False

            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)
            self.edges.append((self.chemicals.index(node_product), self.chemicals.index(node_mol1)))
            self.edges.append((self.chemicals.index(node_product), self.chemicals.index(node_mol2)))            

        elif action == 1 and mol2 is None:  # Expand with uni-mol rxn
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=1,
                parent=None,
                child=[node_mol1.smiles],
                depth=node_mol1.depth + 0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=node_rxn.depth + 0.5,
                index=len(self.chemicals),
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol1.is_root = False

            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)
            self.edges.append((self.chemicals.index(node_product), self.chemicals.index(node_mol1)))

        elif action == 1 and mol2 is not None:  # Expand with bi-mol rxn
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_mol2 = NodeChemical(
                smiles=mol2,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals),
            )
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=2,
                parent=None,
                child=[node_mol1.smiles, node_mol2.smiles],
                depth=max(node_mol1.depth, node_mol2.depth) + 0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=node_rxn.depth + 0.5,
                index=len(self.chemicals) + 1,
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id
            node_mol1.is_root = False

            self.chemicals.append(node_mol2)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)
            self.edges.append((self.chemicals.index(node_product), self.chemicals.index(node_mol1)))
            self.edges.append((self.chemicals.index(node_product), self.chemicals.index(node_mol2)))            
            

        elif action == 0 and mol2 is None:  # Add with uni-mol rxn
            node_mol1 = NodeChemical(
                smiles=mol1,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals),
            )
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=1,
                parent=None,
                child=[node_mol1.smiles],
                depth=0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=1,
                index=len(self.chemicals) + 1,
            )



            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id

            self.chemicals.append(node_mol1)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)
            self.edges.append((self.chemicals.index(node_product), self.chemicals.index(node_mol1)))

        elif action == 0 and mol2 is not None:  # Add with bi-mol rxn
            node_mol1 = NodeChemical(
                smiles=mol1,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals),
            )
            node_mol2 = NodeChemical(
                smiles=mol2,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals) + 1,
            )
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=2,
                parent=None,
                child=[node_mol1.smiles, node_mol2.smiles],
                depth=0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=1,
                index=len(self.chemicals) + 2,
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id

            self.chemicals.append(node_mol1)
            self.chemicals.append(node_mol2)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

            self.edges.append((self.chemicals.index(node_product), self.chemicals.index(node_mol1)))
            self.edges.append((self.chemicals.index(node_product), self.chemicals.index(node_mol2)))

        else:
            raise ValueError("Check input")

        return None
    

    def build_tree(self):
        nodes = []
        for chem in self.chemicals:
            node = Node(smiles=chem.smiles,
                 is_leaf=chem.is_leaf, is_root=chem.is_root)
            nodes.append(node)
        for edge in self.edges:
            a, b = edge
            nodes[a].child.append(nodes[b])
            nodes[b].parent = (nodes[a], self.chemicals[b].parent)
        self.nodes = nodes
        assert nodes[-1].is_root
        return nodes[-1]
        

class Skeleton:
    def __init__(self, st, index):
        """
        st: example of SyntheticTree with the skeleton
        This is a dual use class. It also remembers st for later use.
        """   
        tree = nx.DiGraph(st.edges)
        n = len(st.chemicals)
        smile_set = dict() 
        for c, ind in zip([c.smiles for c in st.chemicals], range(n)):
            smile_set[c] = smile_set.get(c, []) + [ind]                          
        
        nodes = [ZSSNode(node) for node in tree.nodes()]
        for a, b in tree.edges:
            nodes[a].addkid(nodes[b])        
        self.zss_tree = nodes[-1]

        whole_tree = nx.DiGraph()
        for i in tree.nodes():
            whole_tree.add_node(i, smiles=st.chemicals[i].smiles)
        for j, r in zip(range(n, n+len(st.reactions)), st.reactions):
            p = smile_set[r.parent][0]         
            whole_tree.add_node(j, rxn_id=r.rxn_id)
            whole_tree.add_edge(p, j)
            inds = []
            for c in r.child:
                inds.append(smile_set[c][0])
                smile_set[c].pop(0)
            for i in inds:
                assert tree.has_edge(p, i)
                whole_tree.add_edge(j, i)

        self.tree = whole_tree
        self.tree_edges = np.array(self.tree.edges).T        
        self.tree_root = len(st.chemicals)-1
        self.non_root_tree_edges = self.tree_edges[:, (self.tree_edges != self.tree_root).all(axis=0)] # useful later
        self.leaves = np.array([((t not in self.tree_edges[0]) and t != self.tree_root) for t in self.tree.nodes()])                
        self.rxns = np.array(['rxn_id' in self.tree.nodes[n] for n in range(len(self.tree.nodes()))])
        self.bidir_edges = np.concatenate((self.tree_edges, self.tree_edges[::-1]), axis=-1)
        self.index = index
        self.reset()


    def reset(self, mask=None):
        self._mask = np.zeros(len(self.tree), dtype=np.int8)
        self.leaves_up = True
        self.all_leaves = False        
        self.frontier = True # because we have target?
        if mask is not None:
            self.mask = mask


    @property
    def mask(self):
        return self._mask


    @mask.setter
    def mask(self, mask):
        self._mask[mask] = 1        
        src = self.mask[self.non_root_tree_edges[0]]
        dest = self.mask[self.non_root_tree_edges[1]]
        self.leaves_up = not (src > dest).any()
        self.all_leaves = self.mask[self.leaves].all()
        non_mask_rxns = ~(self.mask == 1) & self.rxns
        src_in_mask = self.mask[self.bidir_edges.T[:, 0]]
        self.frontier_nodes = self.bidir_edges.T[src_in_mask == 1][:, 1]        
        self.rxn_frontier = non_mask_rxns[self.frontier_nodes].any()
        self.bb_frontier = self.mask.sum() < len(self.mask) # bad only when no frontier
        src = self.mask[self.tree_edges[0]]
        dest = self.mask[self.tree_edges[1]]
        self.target_down = (src >= dest).all()

        
    
    @staticmethod
    def one_hot(n, ind):
        zeros = np.zeros(n)
        zeros[ind] = 1.
        return zeros
    

    def fill_node(self, n, y):
        if 'smiles' in self.tree.nodes[n]:
            y[n][:256] = fp_256(self.tree.nodes[n]['smiles'])    
            # print(self.tree.nodes[n])
        elif 'rxn_id' in self.tree.nodes[n]:
            y[n][256:] = self.one_hot(91,self.tree.nodes[n]['rxn_id'])
        else:
            print("bad node")

           
    def get_state(self, leaves_up=False, rxn_frontier=False, bb_frontier=False, target_down=False):
        """
        Return the partial graph with self.mask determining which nodes are available
        If leaves_up is true, further zero out y at nodes where there is an un-filled child
        Specifically, return (node_mask, edge_index, X)
            node_mask: self.mask (len(self.tree),)
            X: (len(self.tree), in_dim) matrix of node features, with rows at ~node_mask zero'ed out
            y: (len(self.tree), out_dim) 256-dim mol_fp, with rows at node_mask zero'ed out
        leaves_up: whether to set targets to be nodes where all its children are targets
        rxn_frontier: whether to set targets to be rxn nodes on bfs frontier
        bb_frontier: whether to set targets to be rxn nodes on bfs frontier if present, else bb nodes
        target_down: same as leaves_up but from target down
        """        
        X = np.zeros((len(self.tree), 2*2048+91))
        y = np.zeros((len(self.tree), 256+91))
        try:
            for n in self.tree.nodes():
                leaves_filled = self.mask[list(self.tree.neighbors(n))].all()
                is_frontier = n in self.frontier_nodes
                is_frontier_rxn = is_frontier and self.rxns[n]
                is_frontier_bb = is_frontier and not self.rxns[n] and not self.rxn_frontier
                if is_frontier_rxn:
                    assert self.mask[self.bidir_edges.T[self.bidir_edges[1] == n][:, 0]].any()
                    assert 'rxn_id' in self.tree.nodes[n]                
                if self.mask[n]:
                    # if leaves_filled and list(self.tree.neighbors(n)):
                    #     if 'smiles' in self.tree.nodes[n]: # impossible
                    #         if n != self.tree_root:
                    #             breakpoint()
                    #     else:
                    #         if not self.leaves[list(self.tree.neighbors(n))].all(): # rxn on intermediate
                    #             breakpoint()
                    if 'smiles' in self.tree.nodes[n]:
                        X[n][:2048] = fp_2048(self.tree.nodes[n]['smiles'])
                        X[n][2048:2*2048] = fp_2048(self.tree.nodes[self.tree_root]['smiles'])
                        # print(self.tree.nodes[n])
                    elif 'rxn_id' in self.tree.nodes[n]:
                        assert len(list(self.tree.predecessors(n))) == 1
                        X[n][:2048] = fp_2048(self.tree.nodes[list(self.tree.predecessors(n))[0]]['smiles'])
                        X[n][2048:2*2048] = fp_2048(self.tree.nodes[self.tree_root]['smiles'])
                        X[n][2*2048:] = self.one_hot(91,self.tree.nodes[n]['rxn_id'])
                    else:
                        print("bad node")
                else:
                    if rxn_frontier and is_frontier_rxn:
                        self.fill_node(n, y)
                    if bb_frontier and is_frontier_bb:
                        self.fill_node(n, y)
                    if not rxn_frontier and not bb_frontier:
                        if target_down:
                            if is_frontier:
                                self.fill_node(n, y)
                        else:
                            if not leaves_up or leaves_filled:
                                self.fill_node(n, y)
        except Exception as e:
            print(e)
            print(f"{self.tree.nodes[self.tree_root]['smiles']} bad")
            pass

        return np.atleast_2d(self.mask), X, y
    

    def get_partial_state(self, poss, node):
        """
        Return the partial graph with self.mask determining which nodes are available
        Similar to get_state, but poss is all possible nodes for where node can be
            node_mask: self.mask + poss
            X: (len(self.tree), in_dim) matrix of node features, with rows at ~self.mask zero'ed out
            y: (len(self.tree), out_dim) 256-dim, with rows at poss having the 256-dim mol_fp of node
        """
        X = np.zeros((len(self.tree), 2*2048+91))
        y = np.zeros((len(self.tree), 256+91))
        node_mask = self.mask.copy()
        node_mask[poss] = 1
        for n in self.tree.nodes():
            if self.mask[n]:
                if 'smiles' in self.tree.nodes[n]:
                    try:
                        X[n][:2048] = fp_2048(self.tree.nodes[n]['smiles'])
                        X[n][2048:2*2048] = fp_2048(self.tree.nodes[self.tree_root]['smiles'])
                    except:
                        pass
                elif 'rxn_id' in self.tree.nodes[n]:
                    X[n][2048:] = self.one_hot(91,self.tree.nodes[n]['rxn_id'])
                else:
                    print("bad node")
            elif node_mask[n]:
                if n != node:
                    print(f"{n} is symmetric with {node}")
                assert not (('smiles' in self.tree.nodes[n]) ^ ('smiles' in self.tree.nodes[node]))
                assert not (('rxn_id' in self.tree.nodes[n]) ^ ('rxn_id' in self.tree.nodes[node]))
                if 'smiles' in self.tree.nodes[n]:
                    try: 
                        y[n][:256] = fp_256(self.tree.nodes[node]['smiles'])
                    except: 
                        pass
                elif 'rxn_id' in self.tree.nodes[node]:
                    y[n][256:] = self.one_hot(91,self.tree.nodes[node]['rxn_id'])
                else:
                    print("bad node")    
        return np.atleast_2d(self.mask), X, y                


    @staticmethod
    def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

        '''
        From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
        Licensed under Creative Commons Attribution-Share Alike 
        
        If the graph is a tree this will return the positions to plot this in a 
        hierarchical layout.
        
        G: the graph (must be a tree)
        
        root: the root node of current branch 
        - if the tree is directed and this is not given, 
        the root will be found and used
        - if the tree is directed and this is given, then 
        the positions will be just for the descendants of this node.
        - if the tree is undirected and not given, 
        then a random choice will be used.
        
        width: horizontal space allocated for this branch - avoids overlap with other branches
        
        vert_gap: gap between levels of hierarchy
        
        vert_loc: vertical location of root
        
        xcenter: horizontal location of root
        '''
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
            '''
            see hierarchy_pos docstring for most arguments

            pos: a dict saying where all nodes go if they have been assigned
            parent: parent of this branch. - only affects it if non-directed

            '''
        
            if pos is None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)  
            if len(children)!=0:
                dx = width/len(children) 
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                        pos=pos, parent = root)
            return pos


# helper functions
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
    return graph


def compute_md(tree, root_ind):
    """
    https://en.wikipedia.org/wiki/Metric_dimension_(graph_theory)
    The metric dimension for a path is 1
    The metric dimension for a tree is |leaves|-|joints|
    Leaf is node of degree 1
    Joint is node of degree 3 that has a straight path to at least one leaf
    This function obtains |leaves|-|joints| leaves which is resolving
    Then prune ones which are redundant because the root is also in the set
    """
    leaves = [n for n in tree.nodes() if tree.degree(n) == 1]
    if [n for n in tree.nodes() if tree.degree(n) > 2]:
        joints = dict()
        for leaf in leaves:
            if root_ind == leaf:
                continue
            cur = leaf
            while tree.degree(cur) < 3:
                preds = list(tree.predecessors(cur))
                if len(preds) != 1:
                    breakpoint()
                cur = preds[0]
            joints[cur] = joints.get(cur, []) + [leaf]
        r_set = [root_ind]
        for j, j_leaves in joints.items():
            if len(j_leaves) > 2:
                breakpoint()
            r_set.append(j_leaves[0])
        ntable = {}
        ancs = dict(nx.tree_all_pairs_lowest_common_ancestor(tree))
        dists = dict(nx.all_pairs_shortest_path_length(tree))
        for k in range(len(tree)):
            for i in range(len(tree)):
                for j in range(len(tree)):  
                    if i == j: continue
                    ik = ancs[(i,k)] if (i,k) in ancs else ancs[(k,i)]
                    jk = ancs[(j,k)] if (j,k) in ancs else ancs[(k,j)]
                    d1 = dists[ik][i]+dists[ik][k]
                    d2 = dists[jk][j]+dists[jk][k]    
                    if d1 == d2:
                        ntable[k] = ntable.get(k, []) + [(i, j)]
        for k in ntable:
            ntable[k] = set(ntable[k])
        # greedily remove redundant landmarks
        # later try to prove this will be optimal
        def eval_r_set(r_set):
            cur_set = None
            for r in r_set:
                if cur_set is None:
                    cur_set = deepcopy(ntable[r])
                else:
                    cur_set &= ntable[r]
            return cur_set
        assert eval_r_set(r_set) == set()

        while True:
            for i in range(len(r_set)-1,0,-1): # don't remove root
                new_r_set = r_set[:i]+r_set[i+1:]
                if eval_r_set(new_r_set) == set():
                    r_set = new_r_set
                    break
            if eval_r_set(r_set):
                break
            if i == 1:
                break
            
    else:
        r_set = [root_ind]
    return r_set
            

def get_bool_mask(i, size=-1):    
    mask = list(map(bool, map(int, format(i,'b'))))
    if size > -1:
        mask = [0 for _ in range(size-len(mask))] + mask
    return mask



def get_wl_kernel(tree: nx.digraph, fill_in=[]):
    for n in tree.nodes():
        tree.nodes[n]['id'] = 0
    for i, n in enumerate(fill_in):
        tree.nodes[n]['id'] = i+1
    return weisfeiler_lehman_graph_hash(tree, iterations=len(tree), node_attr='id')



def process_syntree_mask(i, sk, args, min_r_set, anchors=None):
    kwargs = {}
    if args.determine_criteria in ['leaves_up', 'all_leaves']:
        kwargs['leaves_up'] = True
    elif args.determine_criteria == 'rxn_frontier':        
        kwargs['rxn_frontier'] = True        
    elif args.determine_criteria == 'bb_frontier':        
        kwargs['rxn_frontier'] = True
        kwargs['bb_frontier'] = True        
    elif args.determine_criteria == 'target_down':
        kwargs['target_down'] = True
    if anchors is not None:
        poss_vals = []
        val = get_wl_kernel(sk.tree, min_r_set[:2+len(anchors)])
        sk.reset(min_r_set[:1+len(anchors)])
        for poss in min_r_set[len(anchors)+1:]:
            poss_val = get_wl_kernel(sk.tree, min_r_set[:1+len(anchors)] + [poss])
            if poss_val == val:
                poss_vals.append(poss)
        if len(poss_vals) > 1:
            breakpoint()
        # featurize prediction problem of next anchor, which can be any of poss_vals
        node_mask, X, y = sk.get_partial_state(poss_vals, min_r_set[len(anchors)+1])
        return (node_mask, X, y, sk.tree.nodes[sk.tree_root]['smiles'])
    else:
        sk.reset(min_r_set)
        zero_mask_inds = np.where(sk.mask == 0)[0]    
        bool_mask = get_bool_mask(i)
        sk.mask = zero_mask_inds[-len(bool_mask):][bool_mask]   
        node_mask, X, y = sk.get_state(**kwargs)        
        if args.determine_criteria == 'all_leaves':
            assert sk.all_leaves
        return (node_mask, X, y, sk.tree.nodes[sk.tree_root]['smiles'])        


def test_is_leaves_up(i, sk, min_r_set):
    sk.reset(min_r_set)
    zero_mask_inds = np.where(sk.mask == 0)[0]    
    bool_mask = get_bool_mask(i)
    sk.mask = zero_mask_inds[-len(bool_mask):][bool_mask]   
    return sk.leaves_up




class SyntheticTreeSet:
    """Represents a collection of synthetic trees, for saving and loading purposes."""

    def __init__(self, sts: Optional[list[SyntheticTree]] = None):
        self.sts = sts if sts is not None else []

    def __len__(self):
        return len(self.sts)

    def __getitem__(self, index):
        if self.sts is None:
            raise IndexError("No Synthetic Trees.")
        return self.sts[index]


    def load(self, file: str):
        """Load a collection of synthetic trees from a `*.json.gz` file."""
        assert str(file).endswith(".json.gz"), f"Incompatible file extension for file {file}"

        with gzip.open(file, "rt") as f:
            data = json.loads(f.read())

        for st in data["trees"]:
            st = SyntheticTree(st) if st is not None else None
            self.sts.append(st)

        return self

    def save(self, file: str) -> None:
        """Save a collection of synthetic trees to a `*.json.gz` file."""
        assert str(file).endswith(".json.gz"), f"Incompatible file extension for file {file}"
          

        st_list = {"trees": [st.output_dict() for st in self.sts if st is not None]}
        with gzip.open(file, "wt") as f:
            f.write(json.dumps(st_list))

        
        pkl_file = str(file).replace(".json.gz", ".pkl")      
        sts = []
        for st in self.sts:
            if st: 
                try:
                    st.build_tree()
                except:
                    breakpoint()
                sts.append(st)

        done = False
        skeletons = {}
        
        for st in sts:
            for sk in skeletons:
                if st.is_isomorphic(sk):
                    done = True
                    skeletons[sk].append(st)
            if done: continue
            skeletons[st] = [st]

        
        for k, v in skeletons.items():
            print(f"count: {len(v)}") 

        pickle.dump(skeletons, open(pkl_file, 'wb+'))        

    def _print(self, x=3):
        """Helper function for debugging."""
        for i, r in enumerate(self.sts):
            if i >= x:
                break
            print(r.output_dict())


class SkeletonSet:
    def __init__(self, skeletons=None):
        """
        skeletons is a dictionary from syntree: [syntrees]
        Each key is a representative for a different skeleton
        """
        self.skeletons = skeletons
        self.lookup = None
        self.sks = None
        self.coords = None
        self.sim = None
        

    def load_skeletons(self, skeletons):
        """
        This converts skeletons into a lookup from smiles to 
        the skeletons of its synthetic tree(s)
        """
        lookup = {} # should be multi-set?
        sks = []
        for st, sts in skeletons.items():
            sk = Skeleton(st, len(sks)) # uses representative syntree
            for st in sts:
                lookup[st.root.smiles] = lookup.get(st.root.smiles, []) + [sk]
            sks.append(sk)
        self.lookup = lookup
        self.sks = sks   
        self.skeletons = skeletons
        return self
    
    @staticmethod
    def compute_dists(i, j, sk1, sk2):
        # dist = nx.graph_edit_distance(sk1, sk2, roots=(len(sk1)-1, len(sk2)-1), upper_bound=6)
        return simple_distance(sk1, sk2)
        return dist



    def embed_skeletons(self, sks=None, ncpu=0):
        if not sks:
            sks = self.sks
        sim = np.zeros((len(sks), len(sks)))
        print("begin computing similarity matrix")
        # args = [(i, j, sks[i].tree,sks[j].tree) for i in range(len(sks)) for j in range(len(sks)) if j > i]
        args = [(i, j, sks[i].zss_tree,sks[j].zss_tree) for i in range(len(sks)) for j in range(len(sks)) if j > i]
        with Pool(120) as p:
            res = p.starmap(self.compute_dists, tqdm(args, total=len(args)))
        assert len(args) == len(res)
        for (i, j, _, _), d in zip(args, res):
            sim[i][j] = d or 6            
        sim += sim.T
        self.sim = sim
        ms = MDS(n_components=256, dissimilarity='precomputed', verbose=1)
        print("begin mds")
        coords = ms.fit_transform(-sim)
        self.coords = coords
    


if __name__ == "__main__":
    pass
