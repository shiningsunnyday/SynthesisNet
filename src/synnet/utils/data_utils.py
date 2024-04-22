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
import shutil
import uuid
import json
import numpy as np
import random
from typing import Any, Optional, Set, Tuple, Union
import multiprocessing as mp
from multiprocessing import Array, Manager
mp.set_start_method('fork')
from synnet.config import MP_MIN_COMBINATIONS, MAX_PROCESSES, PRODUCT_DIR, PRODUCT_JSON, NUM_THREADS, DELIM
import threading
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm
import pickle
from time import time
from itertools import permutations, product
from collections import defaultdict, deque
from networkx.algorithms.isomorphism import rooted_tree_isomorphism
from networkx.algorithms import weisfeiler_lehman_graph_hash
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from zss import Node as ZSSNode, simple_distance
from copy import deepcopy
from filelock import FileLock
from synnet.encoding.fingerprints import fp_2048, fp_256
import os
import math
import logging
import hashlib


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



class ProductMap:
    def __init__(self, fpath, loaded=True):         
        self.fpath = fpath
        if loaded:
            self._product_map = {}
            self._loaded = True
        else:
            assert os.path.exists(fpath)
            self._product_map = None
            self._loaded = False

    
    def save(self):
        logger = logging.getLogger('global_logger')
        logger.info(f"begin saving product map")
        assert self._loaded, "need to call load() first"      
        if PRODUCT_JSON:
            ProductMap.json_dump(self._product_map, open(self.fpath, 'w+'))
        else:
            pickle.dump(self._product_map, open(self.fpath, 'wb+'))
        logger.info(f"done saving product map")
        self._product_map = None
        self._loaded = False


    def unload(self):
        if self._loaded:
            self._product_map = None
            self._loaded = False        

    

    @staticmethod
    def str_key_to_int(dic):
        """
        recursively convert str int keys to int
        """
        if isinstance(dic, dict):
            str_keys = [int(k) if isinstance(k, str) and k.isdigit() else k for k in dic]
            dic = dict(zip(str_keys, dic.values()))        
            for k in dic:
                dic[k] = ProductMap.str_key_to_int(dic[k])
        return dic

    
    @staticmethod
    def json_dump(dic=None, f=None):
        # simple wrapper that converts tuple keys into - delimited keys
        if isinstance(dic, dict):
            str_keys = [DELIM.join(map(str, k)) if isinstance(k, tuple) else k for k in dic]
            dic = dict(zip(str_keys, dic.values()))
            for k in dic:
                dic[k] = ProductMap.json_dump(dic=dic[k])
        if f is not None:
            assert dic is not None
            json.dump(dic, f)
        return dic

    
    @staticmethod
    def json_load(f=None, dic=None):
        # simple wrapper that restores - delimited keys into tuple keys
        if f is not None: # base case
            try:
                dic = json.load(f)
            except:
                print(f)
                raise
        if isinstance(dic, dict):
            tup_keys = [tuple(map(int, (k.split(DELIM)))) if DELIM in k else k for k in dic]
            dic = dict(zip(tup_keys, dic.values()))
            str_keys = [int(k) if isinstance(k, str) and k.isdigit() else k for k in dic]
            dic = dict(zip(str_keys, dic.values()))                              
            for k in dic:
                dic[k] = ProductMap.json_load(dic=dic[k])
        return dic
        

    
    def load(self):
        if not self._loaded:      
            if PRODUCT_JSON:
                self._product_map = ProductMap.json_load(open(self.fpath, 'r'))       
            else:
                self._product_map = pickle.load(open(self.fpath, 'rb'))
                self._product_map = self.str_key_to_int(self._product_map)
            self._loaded = True
    

    def get_num_interms(self, key):
        if key not in self._product_map:
            print(self.fpath, f"{key} has no interms!")
            print(self._product_map.keys())
            breakpoint()
        return len(self._product_map[key])

    
    def __setitem__(self, key, val):   
        assert self._loaded
        assert isinstance(key, tuple)   
        assert len(key) == 3
        n, interm, e = key
        if n not in self._product_map:
            self._product_map[n] = {}
        if interm not in self._product_map[n]:
            self._product_map[n][interm] = {}        
        self._product_map[n][interm][e] = val


    def __getitem__(self, key):   
        assert self._loaded
        assert isinstance(key, tuple)   
        if len(key) == 3:
            n, interm, e = key
            return self._product_map[n][interm][e]  
        elif len(key) == 2:
            n, interm = key
            return self._product_map[n][interm]
        elif len(key) == 1:
            n = key[0]
            return self._product_map[n]
    

    def copy(self):
        ext = "json" if PRODUCT_JSON else "pkl"
        new_fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")
        while os.path.exists(new_fpath):
            new_fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")
        if not self._loaded:
            self.load()
        if PRODUCT_JSON:
            ProductMap.json_dump(self._product_map, open(new_fpath, 'w+'))
        else:
            pickle.dump(self._product_map, open(new_fpath, 'wb+'))
        new_pmap = ProductMap(new_fpath, loaded=False)        
        self.unload()
        return new_pmap



    def combine(self, other, offset):
        if not self._loaded:
            self.load()
        if not other._loaded:
            other.load()
        for n in other._product_map:
            for r in other._product_map[n]:
                for e, v in other._product_map[n][r].items():
                    self[(n+offset, r, e+offset)] = v
        # combine self.product_map's and correct for entries            
                



class ProductMapLink:
    """
    Making a product map memory-efficient by storing a dict of files
    The following functions are available:
        load(): load all base files into a in-memory dict
        save(): write _product_map back into the base files
        copy(): create a new ProductMapLink with new (copied) list of files
        combine(): combine two ProductMapLinks, by combining base files with disjoint keys
        __getitem__
        __setitem__        
    """
    def __init__(self, fpaths):
        self.fpaths = fpaths
        assert isinstance(fpaths, dict)
        self._product_map = None
        self._loaded = False
        for fpath in self.fpaths.values():
            assert os.path.exists(fpath)   


    def load(self):
        logger = logging.getLogger('global_logger')
        logger.info(f"begin loading product map link")        
        if not self._loaded:    
            self._product_map = {}
            self._loaded = True
            for entry_key, fpath in self.fpaths.items():
                logger.info(f"begin loading {fpath} for entry key {entry_key}")
                # with FileLock(f"{fpath}.lock"):
                offset = 0
                if isinstance(fpath, tuple):
                    offset, fpath = fpath
                    self.fpaths[entry_key] = fpath  
                try:                       
                    f = open(fpath, 'r' if PRODUCT_JSON else 'rb')
                    self._product_map[entry_key] = (ProductMap.json_load if PRODUCT_JSON else pickle.load)(f)
                except:
                    print(fpath)
                if offset:
                    n = entry_key
                    for r in self._product_map[n]:
                        entries = list(self._product_map[n][r].keys())
                        for e in entries:
                            if isinstance(e, tuple):
                                offset_key = (int(e[0])+offset, int(e[1])) 
                            else:
                                offset_key = int(e)+offset
                            assert offset_key not in self[(n,r)]
                            self[(n, r, offset_key)] = self._product_map[n][r][e]                                
                        for e in entries:
                            self._product_map[n][r].pop(e)
        logger.info(f"done loading product map link")

    
    def save(self):
        logger = logging.getLogger('global_logger')
        logger.info(f"begin saving product map")
        assert self._loaded, "need to call load() first" 
        for entry_key in self._product_map:
            if entry_key in self.fpaths:
                fpath = self.fpaths[entry_key]
            else:
                ext = "json" if PRODUCT_JSON else "pkl"
                fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")
                while os.path.exists(fpath):
                    fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")                
                self.fpaths[entry_key] = fpath
            with FileLock(f"{fpath}.lock"):
                with open(fpath, 'w+' if PRODUCT_JSON else 'wb+') as f:
                    (ProductMap.json_dump if PRODUCT_JSON else pickle.dump)(self._product_map[entry_key], f)
        logger.info(f"done saving product map")
        self._product_map = None
        self._loaded = False


    def unload(self):
        if self._loaded:
            self._product_map = None
            self._loaded = False        


    def copy(self):
        """
        copy _product_map into new files
        filenames are random uuid's to avoid collision
        """
        ext = "json" if PRODUCT_JSON else "pkl"
        if not self._loaded:
            # since product map not loaded, we can copy without loading
            new_fpaths = {}
            for entry_key, fpath in self.fpaths.items():                
                new_fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")
                while os.path.exists(new_fpath):
                    new_fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")
                shutil.copyfile(fpath, new_fpath)
                new_fpaths[entry_key] = new_fpath
            new_pmap = ProductMapLink(new_fpaths)        
        else:                        
            new_fpaths = {}
            for entry_key in self._product_map:
                new_fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")
                while os.path.exists(new_fpath):
                    new_fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")
                with open(new_fpath, 'w+' if PRODUCT_JSON else 'wb+') as f:
                    (ProductMap.json_dump if PRODUCT_JSON else pickle.dump)(self._product_map[entry_key], f)
                new_fpaths[entry_key] = new_fpath
            new_pmap = ProductMapLink(new_fpaths)        
            self.unload()
        return new_pmap    


    def combine(self, other, offset):
        """
        combine two maplinks by combining the fpaths
        unsaved work on _product_map will not be carried over
        NOTE: when loading fpath, the innermost keys will also need to be offset                
        """
        assert not self._loaded
        assert not other._loaded
        for entry_key, fpath in other.fpaths.items():
            self.fpaths[entry_key+offset] = (offset, fpath)



    def __setitem__(self, key, val):   
        assert self._loaded
        assert isinstance(key, tuple)   
        assert len(key) == 3
        n, interm, e = key
        if n not in self._product_map:
            self._product_map[n] = {}
        if interm not in self._product_map[n]:
            self._product_map[n][interm] = {}        
        self._product_map[n][interm][e] = val


    def __getitem__(self, key):   
        assert self._loaded
        assert isinstance(key, tuple)   
        if len(key) == 3:
            n, interm, e = key
            return self._product_map[n][interm][e]  
        elif len(key) == 2:
            n, interm = key
            return self._product_map[n][interm]
        elif len(key) == 1:
            n = key[0]
            return self._product_map[n]  
    

    def get_num_interms(self, key):
        if key not in self._product_map:
            print(self.fpaths, f"{key} has no interms!")
            print(self._product_map.keys())
            breakpoint()
        return len(self._product_map[key])                 



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
            ext = "json" if PRODUCT_JSON else "pkl"              
            fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")
            while os.path.exists(fpath):
                fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")
            self.product_map = ProductMap(fpath)
            self.product_map.save()
            # assert 'depth' in self.rxn_tree.graph


    def copy(self):        
        other = Program(deepcopy(self.rxn_tree))
        other.rxn_map = deepcopy(self.rxn_map)        
        other.keep_prods = self.keep_prods
        other._entries = self._entries
        if self.keep_prods:
            other.product_map = self.product_map.copy()        
        return other



    @property
    def entries(self):
        return self._entries

    
    @staticmethod
    def make_default_dict():
        return defaultdict(dict)  
    
    
    @staticmethod
    def input_length(p, rxns=None):
        react = []
        if p.rxn_map:
            for e in p.entries:
                if isinstance(e, tuple):
                    r, idx = e
                    if r in p.rxn_map:
                        num = len(p.rxn_map[r].available_reactants[idx])
                    else:
                        assert rxns is not None
                        rxn_id = p.rxn_tree.nodes[r]['rxn_id']
                        num = len(rxns[rxn_id].available_reactants[idx])
                else:
                    num = np.prod([len(ar) for ar in p.rxn_map[e].available_reactants])
                react.append(num)
            return np.prod(react)
        else:
            return 0
    

    @staticmethod
    def avg_input_length(progs):
        num_poss = []
        for p in progs:
            num_poss.append(Program.input_length(p))
        return np.mean(num_poss) if len(num_poss) else 0

    
    @staticmethod
    def hash_str(s):
        # s is bytes string
        return hashlib.md5(s).hexdigest() # deterministic hashing 


    @staticmethod
    def hash_json(json_data):
        return Program.hash_str(json.dumps(json_data, sort_keys=True).encode())    
    

    def get_path(self):
        mask = []
        for n in self.rxn_tree:
            if 'rxn_id_forcing' in self.rxn_tree.nodes[n]:
                b = self.rxn_tree.nodes[n]['rxn_id_forcing'] != -1
            elif 'rxn_id' in self.rxn_tree.nodes[n]:
                b = self.rxn_tree.nodes[n]['rxn_id'] != -1
            else:
                breakpoint()
            mask.append(b)
        hash_val = self.hash(mask)
        return f"{hash_val}.json"
    

    def hash(self, mask, return_json=False, attrs=['rxn_id', 'depth', 'child']):
        # used to hash the partial state defined by mask
        # can also return the tree data defined by mask
        # if attrs, keep only the attrs
        rxn_tree_copy = deepcopy(self.rxn_tree)
        for n in self.rxn_tree:
            node_names = list(rxn_tree_copy.nodes[n])
            for k in node_names:
                if k not in attrs:
                    rxn_tree_copy.nodes[n].pop(k)
        data = {}
        for n in rxn_tree_copy.nodes():
            if not mask[n]:
                data[n] = rxn_tree_copy.nodes[n]['rxn_id']
                rxn_tree_copy.nodes[n].pop('rxn_id')
        json_data = nx.tree_data(rxn_tree_copy, len(rxn_tree_copy)-1)        
        ans = self.hash_json(json_data)
        for n, r in data.items():
            rxn_tree_copy.nodes[n]['rxn_id'] = r
        if return_json:
            return json_data        
        return str(ans)

    
    def hash_program(self):
        return self.hash_json(self.output_dict)


    def output_dict(self):
        dic = {
            'rxn_tree': nx.tree_data(self.rxn_tree, len(self.rxn_tree)-1),
            'entries': self._entries,
            'rxn_map': {r: rxn.__dict__ for r, rxn in self.rxn_map.items()},
            'keep_prods': self.keep_prods            
        }
        if self.keep_prods:            
            dic['product_map'] = self.product_map.fpath
        return dic
    

    def combine_bi_mol(self, rxn_id, child='left'):
        """
        1. Add a reaction node rxn_id n
        2. Connect it to len(rxn_tree)-1
        3. Update child, depth
        4. Update _entries at reactant_idx level to include n
        """
        assert child in ['left', 'right']
        key = len(self.rxn_tree)
        self.rxn_tree.add_node(key, rxn_id=rxn_id)
        self.rxn_tree.add_edge(key, key-1)
        self.rxn_tree.nodes[key-1]['child'] = child
        depth = self.rxn_tree.nodes[key-1]['depth'] + 1
        self.rxn_tree.nodes[key]['depth'] = depth
        # make this list of tuples (node, reactant idx)
        idx = 1 if child == 'left' else 0
        self._entries = self._entries + [(key, idx)]

    

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
        apply_offset = lambda e, offset: (e[0]+offset, e[1]) if isinstance(e, tuple) else e+offset
        self._entries = self.entries + [apply_offset(e, offset) for e in other.entries]               
        for n in other.rxn_map:
            self.rxn_map[n+offset] = other.rxn_map[n]
        # combine rxn_map's
        assert self.keep_prods == other.keep_prods
        if self.keep_prods:
            self.product_map.combine(other.product_map, offset=offset)
        return self
    

    @staticmethod
    def fill_reactant_indices(res, all_reactant_idxes):
        for _, idxes in res:
            assert len(all_reactant_idxes) == len(idxes)
            for i in range(len(all_reactant_idxes)):
                idx = idxes[i]
                if len(idx) != len(all_reactant_idxes[i]):
                    breakpoint()
                assert len(idx) == len(all_reactant_idxes[i])
                for j in range(len(idx)):
                    all_reactant_idxes[i][j][idx[j]] += 1

    @staticmethod
    def fill_product_reactant_indices(res, all_reactant_idxes, product_map=None):
        for r, index in tqdm(res):
            idxes = Program.infer_product_index(index, interm_counts, entries, rxn_map, return_idx=True)            
            assert len(idxes) == len(all_reactant_idxes)
            prod = {}
            for i in range(len(all_reactant_idxes)):
                idx = idxes[i]
                assert len(idx) == len(all_reactant_idxes[i])
                for j in range(len(idx)):
                    all_reactant_idxes[i][j][idx[j]] += 1            
                prod[entries[i]] = idxes[i]
            if product_map is not None:           
                if r not in product_map:
                    product_map[r] = prod
    

    def init_rxns(self, rxns):
        """
        Init rxns will look at the top node, n, in rxn_tree which should not be in rxn_map        
        Then use rxns to add the reaction to self.rxn_map
        Also, it will retrieve the products of each of n's successors
        Then use n's reaction to filter them
        Then re-index all the entry nodes for each successor
        """
        logger = logging.getLogger('global_logger')  
        new_nodes = [n for n in self.rxn_tree if n not in self.rxn_map]
        assert len(new_nodes) == 1
        new_node = new_nodes[0]          
        n = new_node               
        rxn = deepcopy(rxns[self.rxn_tree.nodes[n]['rxn_id']])
        assert rxn.available_reactants is not None
        self.rxn_map[n] = rxn
        count = Program.input_length(self)  
        # if (self.rxn_tree.nodes[0]=={'rxn_id': 41, 'depth': 1, 'child': 'left'}) and (self.rxn_tree.nodes[1]=={'rxn_id': 1, 'depth': 2}):
        #     breakpoint()
        if self.keep_prods:
            # do we really need to load?            
            # self.keep_prods specifies the length of the longest root-leave path
            # we need to load if one of new node's successors has depth <= self.keep_prods
            need_load = False
            for succ in list(self.rxn_tree.successors(new_node)):
                if self.keep_prods >= self.rxn_tree.nodes[succ]['depth']:
                    need_load = True
            if need_load:
                self.product_map.load()          
        """
        For each of new node's successor, if its depth is <= self.keep_prods, then we can use the product map to filter the reactants of the 
        successor's subtree's entry nodes
        """        
        for succ in self.rxn_tree.successors(n):           
            """
            for each child of n, we can define a filter function depending on if it's the left or right child
            left also includes uni-molecular reaction
            """
            if self.keep_prods < self.rxn_tree.nodes[succ]['depth']: 
                continue
            """
            n is a new reaction node, but we can filter the entry reactants using the interms
            """
            if self.rxn_tree.nodes[succ]['child'] == 'left':
                filter_func = rxn.is_reactant_first 
            else:
                filter_func = rxn.is_reactant_second
            res = []    
            num_interms = self.product_map.get_num_interms(succ)        
            bad_interms = []                
            """
            product_map stores the interms at n's successor
            """
            if count >= MP_MIN_COMBINATIONS:
                with mp.Pool(100) as p:
                    pass_filter = p.map(filter_func, tqdm(self.product_map[(succ,)], desc=f"filtering {num_interms} interms"))
            else:
                pass_filter = [filter_func(interm) for interm in tqdm(self.product_map[(succ,)], desc=f"filtering {num_interms} interms")]               
            last_interm = None
            for interm_pass, interm in tqdm(zip(pass_filter, self.product_map[(succ,)]), desc=f"sorting good vs bad interms"):
                if interm_pass:
                    if list(self.product_map[(succ, interm)].keys()) != self.entries: # same order
                        # make sure appear in same order
                        appear_entries = [self.entries.index(p) for p in list(self.product_map[(succ, interm)].keys())]
                        if sorted(appear_entries) != appear_entries:
                            breakpoint()
                            
                    entry_reactants = []
                    for e in self.product_map[(succ, interm)]:
                        e_reactants = []
                        for (i, ind) in enumerate(self.product_map[(succ, interm, e)]):
                            # e_reactants.append(self.rxn_map[e].available_reactants[i][ind])
                            e_reactants.append(ind)                            
                        entry_reactants.append(e_reactants)
                    res.append((interm, entry_reactants))
                else:
                    bad_interms.append(interm)            
                last_interm = interm
            if last_interm is None:
                breakpoint()
            entries = list(self.product_map[(succ, last_interm)].keys())                        
            appear_entries = [self.entries.index(p) for p in list(self.product_map[(succ, interm)].keys())]
            if sorted(appear_entries) != appear_entries:
                breakpoint()            
                 
            if count >= MP_MIN_COMBINATIONS and NUM_THREADS > 1:
                threads = []
                elems_per_thread = (len(res)+NUM_THREADS-1) // NUM_THREADS
                all_reactant_indices = []
                for e in entries:
                    zero_count = []               
                    if isinstance(e, tuple):
                        zero_array = [0 for _ in range(len(self.rxn_map[e[0]].available_reactants[e[1]]))]
                        arr = Array('i', zero_array)            
                        zero_count.append(arr)
                    else:
                        for i in range(len(self.rxn_map[e].available_reactants)):
                            zero_array = [0 for _ in range(len(self.rxn_map[e].available_reactants[i]))]            
                            arr = Array('i', zero_array)            
                            zero_count.append(arr)                        
                    all_reactant_indices.append(zero_count)                
                for i in range(NUM_THREADS):
                    start = i*elems_per_thread
                    end = (i+1)*elems_per_thread                
                    thread = threading.Thread(target=self.fill_reactant_indices, args=(res[start:end], all_reactant_indices))
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()
            else:         
                all_reactant_indices = []
                for e in entries:
                    zero_count = []
                    if isinstance(e, tuple):
                        zero_array = [0 for _ in range(len(self.rxn_map[e[0]].available_reactants[e[1]]))]
                        if NUM_THREADS > 1:
                            arr = Array('i', zero_array)
                        else:
                            arr = zero_array                
                        zero_count.append(arr)
                    else:
                        for i in range(len(self.rxn_map[e].available_reactants)):
                            zero_array = [0 for _ in range(len(self.rxn_map[e].available_reactants[i]))]
                            if NUM_THREADS > 1:
                                arr = Array('i', zero_array)
                            else:
                                arr = zero_array
                            zero_count.append(arr)
                    all_reactant_indices.append(zero_count)                                   
                self.fill_reactant_indices(res, all_reactant_indices)
            rxn_map_copy = deepcopy(self.rxn_map)
            for n in entries:
                idx = -1
                if isinstance(n, tuple):
                    n, idx = n                
                cur = self.rxn_map[n].available_reactants
                available_reactants = []
                for i, reactants in enumerate(cur):
                    if idx > -1 and i != idx:
                        available_reactants.append(reactants)
                    else:
                        available_reactants.append([])                
                self.rxn_map[n].available_reactants = tuple(available_reactants)                 

            """
            A simple algorithm to re-index all_reactant_indices            
            """
            assert len(entries) == len(all_reactant_indices)
            for i, e in enumerate(entries): # per entry
                if isinstance(e, tuple):
                    assert len(all_reactant_indices[i]) == 1
                    e, idx = e
                    reactant_indices = [idx]
                else:
                    reactant_indices = range(len(all_reactant_indices[i]))                
                for idx, j in enumerate(reactant_indices): # per reactant
                    c = 0
                    idxes = all_reactant_indices[i][idx]
                    for k in range(len(idxes)): # per available reactant
                        # [0, 0, 1, 0, 1, 0, 1] -> [-1, -1, 0, -1, 1, -1, 2]                            
                        if idxes[k]:
                            # add this reactant to self.rxn_map
                            reactant = rxn_map_copy[e].available_reactants[j][k]
                            self.rxn_map[e].available_reactants[j].append(reactant)
                            idxes[k] = c                                
                            c += 1
                        else:
                            idxes[k] = -1
            """
            Fix product map
            """
            logging.info(f"begin re-indexing product map")
            # remove the bad interms
            for interm in bad_interms:                            
                self.product_map[tuple([succ])].pop(interm)                    
          
            self.reindex_product_map(succ, entries, all_reactant_indices)
            # re-index the entry_reactant indices of self.product_map
            logging.info(f"done re-indexing product map")

            """
            Sanity check: there exists successor with zero products iff program length is 0
            """            
            zero_prods = len(self.product_map[tuple([succ])]) == 0
            if zero_prods:
                assert Program.input_length(self) == 0
            
        if self.keep_prods and need_load:
            new_count = Program.input_length(self)
            logger.info(f"{count}->{new_count} products")
            self.product_map.save()
            

    @staticmethod
    def infer_product_index(i, interm_counts, entries, rxn_map, return_idx=False):
        # For example:
        # [[8,4],[8]] has 2 entry points
        # entries[0] has 8 choices for reactant1 and 4 for reactant2
        # index can be 0 to 32*8-1
        # the trick is to use the intermediate counts [32,8]
        # then repeat for (index//8, [8,4]) and (index%8, [8])        
        # I comment the code with index 42 as an example
        assert len(interm_counts) == len(entries)
        entry_indices = [[] for _ in entries]
        for j in range(len(interm_counts)-1,-1,-1):                     
            if isinstance(entries[j], tuple):
                num_reactants = 1
                r, idx = entries[j]
                entry = r
                reverse_reactant_idxes = [idx]
            else:
                num_reactants = len(rxn_map[entries[j]].available_reactants)
                entry = entries[j]
                reverse_reactant_idxes = range(num_reactants-1,-1,-1) # [0], [1, 0]
            reactant_indices = [None, None]
            reactant_i = i%interm_counts[j] # 2, 5
            for k in reverse_reactant_idxes: # [0], [1, 0]
                num_reactants_i = len(rxn_map[entry].available_reactants[k]) # 8, 4 8
                reactant_indices[k] = reactant_i % num_reactants_i # 2, 1 0
                reactant_i //= num_reactants_i # 0, 1 0
            assert reactant_i == 0
            entry_indices[j] = reactant_indices # [2, -1] [1, 0]
            i //= interm_counts[j] # 5 0
        assert i == 0        
        entry_reactants = []
        for entry, entry_reactant_indices in zip(entries, entry_indices):
            entry_reactants.append([])
            for j, reactant_index in enumerate(entry_reactant_indices):
                if reactant_index is not None:
                    if isinstance(entry, tuple):
                        r, idx = entry
                        assert j == idx
                        reactant = rxn_map[r].available_reactants[j][reactant_index]
                    else:
                        reactant = rxn_map[entry].available_reactants[j][reactant_index]
                    entry_reactants[-1].append(reactant_index if return_idx else reactant)
        return entry_reactants


    @staticmethod
    def run_rxns(i):
        """
        Use the index and rxn_map to infer the reactant combination
        index refers to position in product(product(reactants))
        """        
        entry_reactants = Program.infer_product_index(i, interm_counts, entries, rxn_map)       
        good = True
        product_map = {}
        for node in nx.dfs_postorder_nodes(rxn_tree, len(rxn_tree)-1):
            if node in entries:
                entry = entries.index(node)
                prod = rxn_map[node].run_reaction(tuple(entry_reactants[entry]))
                product_map[node] = prod
            elif node in [e[0] for e in entries if isinstance(e, tuple)]: # tuple
                entry = [e[0] if isinstance(e, tuple) else -1 for e in entries].index(node)
                _, reactant_index = entries[entry]
                reactant = entry_reactants[entry]
                succ = list(rxn_tree.successors(node))
                assert len(reactant) == 1                
                assert len(succ) == 1
                reactant = reactant[0]
                succ = succ[0]
                prod = product_map[succ]
                reactants = [prod, reactant] if reactant_index == 1 else [reactant, prod]
                good = True
                if not rxn_map[node].is_reactant_first(reactants[0]):
                    good = False
                if not rxn_map[node].is_reactant_second(reactants[1]):
                    good = False                    
                if good:
                    product_map[node] = rxn_map[node].run_reaction(tuple(reactants))
                else:
                    break
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
                    break
        if good:
            return product_map[len(rxn_tree)-1]
        else:          
            return None             


    @staticmethod
    def retrieve_entry(r, i):
        return (r, Program.infer_product_index(i, interm_counts, entries, rxn_map))
    


    def reindex_product_map(self, n, entries, all_reactant_indices):
        """
        We want to re-index self.product_map using all intermediates of node n
        We are given all_reactant_indices, which re-indexes the reactant indices in product_map
        """
        for interm in tqdm(self.product_map[tuple([n])], "re-indexing product map interms"):
            assert list(self.product_map[tuple([n])][interm]) == entries
            for entry in self.product_map[tuple([n])][interm]:
                e = entries.index(entry)
                for i in range(len(self.product_map[tuple([n])][interm][entry])):                    
                    idx = self.product_map[tuple([n])][interm][entry][i]                        
                    new_idx = all_reactant_indices[e][i][idx]               
                    assert new_idx != -1
                    self.product_map[tuple([n])][interm][entry][i] = new_idx       
    

    def run_rxn_tree(self): 
        # Assume entry inputs satisfy everything except the "root"        
        # reactant_map = {}
        # if self.rxn_tree.graph['super']:
        #     breakpoint()
        # for n in self.entries:
        #     reactant_map[n] = product(*[reactants for reactants in self.rxn_map[n].available_reactants])
        # all_entry_reactants = list(product(*[reactant_map[n] for n in self.entries]))
        logger = logging.getLogger('global_logger')  
        prods = []
        for n in self.entries:            
            if isinstance(n, tuple):
                r, idx = n
                poss_reactants = len(self.rxn_map[r].available_reactants[idx])
            else:                
                poss_reactants = np.prod([len(reactants) for reactants in self.rxn_map[n].available_reactants])
            prods.append(poss_reactants)            
        
        res = []
        if len(prods):
            interm_counts = prods
            count = np.prod(prods)        
        else:
            count = 0          
            return 0, None
        
        rxn_tree = self.rxn_tree
        rxn_map = self.rxn_map
        entries = self.entries

        globals()["rxn_tree"] = rxn_tree
        globals()["rxn_map"] = rxn_map
        globals()["interm_counts"] = interm_counts
        globals()["entries"] = entries
        # globals()["all_entry_reactants"] = all_entry_reactants # debug
        if count >= MP_MIN_COMBINATIONS:            
            logger.info(f"running {count} entry_reactants")
            with mp.Pool(MAX_PROCESSES) as p:
                res = p.map(self.run_rxns, tqdm(range(count), desc="executing reactions"))           
                assert len(res) == count
        else:
            res = [self.run_rxns(i) for i in range(count)]
        
        res = [(r, i) for (r, i) in zip(res, range(count)) if r is not None]
        # Update the reactants to only valid inputs
        rxn_map_copy = deepcopy(self.rxn_map)
        for n in self.entries:                    
            idx = -1
            if isinstance(n, tuple):
                n, idx = n        
            cur = rxn_map_copy[n].available_reactants
            available_reactants = []
            for i, reactants in enumerate(cur):
                if idx > -1 and i != idx:
                    available_reactants.append(reactants)
                else:
                    available_reactants.append([])
            rxn_map_copy[n].available_reactants = tuple(available_reactants)

        """
        The following re-labels the available reactants of each entry
        For each entry, for each reactant index (0 or 1),
        we'd like a dict mapping reactant to index
        """
        keep_prods = self.rxn_tree.nodes[len(self.rxn_tree)-1]['depth'] <= self.keep_prods
        if keep_prods:
            self.product_map.load()
            for e in self.product_map._product_map:
                for interm in self.product_map._product_map[e]:
                    if list(self.product_map._product_map[e][interm]) == [(1, 1), 0]:
                        breakpoint()              
        logger.info(f"begin post-processing {len(res)} products")

        
        # rxn_map_debug = deepcopy(rxn_map_copy)
        # product_map_debug = self.product_map.copy()
        # product_map_debug.load()
        # for r, index in tqdm(res, desc="post-processing products"):               
        #     entry_reactants = Program.infer_product_index(index, interm_counts, entries, rxn_map)            
        #     for entry_reactant, n in zip(entry_reactants, self.entries):
        #         entry_point = []
        #         for i, reactant in enumerate(entry_reactant):
        #             if reactant not in avail_index[n][i]:                        
        #                 avail_index[n][i][reactant] = len(rxn_map_debug[n].available_reactants[i])
        #                 rxn_map_debug[n].available_reactants[i].append(reactant)
        #             if keep_prods:
        #                 entry_point.append(avail_index[n][i][reactant])
        #         if keep_prods:
        #             product_map_debug[(len(self.rxn_tree)-1, r, n)] = entry_point

        """
        Smarter way using multi-threading
        """
        threads = []
        elems_per_thread = (len(res)+NUM_THREADS-1) // NUM_THREADS

        all_reactant_indices = []
        for e in self.entries:
            zero_count = []
            if isinstance(e, tuple):
                zero_array = [0 for _ in range(len(self.rxn_map[e[0]].available_reactants[e[1]]))]
                if NUM_THREADS > 1:
                    arr = Array('i', zero_array)
                else:
                    arr = zero_array                
                zero_count.append(arr)
            else:
                for i in range(len(self.rxn_map[e].available_reactants)):
                    zero_array = [0 for _ in range(len(self.rxn_map[e].available_reactants[i]))]
                    if NUM_THREADS > 1:
                        arr = Array('i', zero_array)
                    else:
                        arr = zero_array
                    zero_count.append(arr)
            all_reactant_indices.append(zero_count)   

        if keep_prods:
            product_map = self.product_map._product_map
            if count >= MP_MIN_COMBINATIONS and NUM_THREADS > 1:
                product_map[len(rxn_tree)-1] = Manager().dict()
            else:
                product_map[len(rxn_tree)-1] = {}
        else:
            product_map = None

        if count >= MP_MIN_COMBINATIONS and NUM_THREADS > 1:
            for i in range(NUM_THREADS):
                start = i*elems_per_thread
                end = (i+1)*elems_per_thread                
                thread = threading.Thread(target=self.fill_product_reactant_indices, 
                                          args=(res[start:end], 
                                          all_reactant_indices, 
                                          product_map[len(rxn_tree)-1] if product_map is not None else None))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()    
        else:  
            self.fill_product_reactant_indices(res, all_reactant_indices, product_map[len(rxn_tree)-1] if product_map is not None else None)

        if keep_prods: # sanity check
            for e in product_map:
                for interm in product_map[e]:
                    if list(product_map[e][interm]) == [(1, 1), 0]:
                        breakpoint()

        """
        A simple algorithm to re-index all_reactant_indices            
        """
        for i in range(len(all_reactant_indices)): # per entry
            entry = self.entries[i]
            if isinstance(entry, tuple):
                assert len(all_reactant_indices[i]) == 1
                entry, idx = entry
                reactant_indices = [idx]
            else:
                reactant_indices = range(len(all_reactant_indices[i]))
            for idx, j in enumerate(reactant_indices): # per reactant
                c = 0
                idxes = all_reactant_indices[i][idx]
                for k in range(len(idxes)): # per available reactant
                    # [0, 0, 1, 0, 1, 0, 1] -> [-1, -1, 0, -1, 1, -1, 2]                                                
                    if idxes[k]:
                        # add this reactant to self.rxn_map
                        reactant = self.rxn_map[entry].available_reactants[j][k]
                        rxn_map_copy[entry].available_reactants[j].append(reactant)
                        idxes[k] = c                                
                        c += 1
                    else:
                        idxes[k] = -1 


        # for e1, e2 in zip(rxn_map_copy, rxn_map_debug):
        #     assert e1 == e2
        #     if rxn_map_copy[e1].available_reactants != rxn_map_debug[e2].available_reactants:
        #         breakpoint()        
   
        if keep_prods:
            product_map[len(rxn_tree)-1] = dict(product_map[len(rxn_tree)-1])
            self.product_map._product_map = product_map
            logging.info(f"begin re-indexing product map")               
            self.reindex_product_map(len(rxn_tree)-1, self.entries, all_reactant_indices)
            # if self.product_map._product_map != product_map_debug._product_map:
            #     breakpoint()            
            
        # re-index the entry_reactant indices of self.product_map
        logging.info(f"done re-indexing product map")         
                                           

        
        logger.info(f"done post-processing {len(res)} products")
        if keep_prods:
            for e in self.product_map._product_map:
                for interm in self.product_map._product_map[e]:
                    if list(self.product_map._product_map[e][interm]) == [(1, 1), 0]:
                        breakpoint()            
            self.product_map.save()
            
        self.rxn_map = rxn_map_copy
        return count, res
                        

    def logging_info(self):
        return nx.tree_data(self.rxn_tree, len(self.rxn_tree)-1)  


    @staticmethod
    def migrate(p):
        ext = "json" if PRODUCT_JSON else "pkl"
        try:
            p.product_map.load()
        except:
            print(p.product_map.fpath)
            print(nx.tree_data(p.rxn_tree, len(p.rxn_tree)-1))
            return p
        new_fpaths = {}
        for entry_key in p.product_map._product_map:
            new_fpath = os.path.join(PRODUCT_DIR, f"{str(uuid.uuid4())}.{ext}")
            new_fpaths[entry_key] = new_fpath                        
            with open(new_fpath, 'w+' if PRODUCT_JSON else 'wb+') as f:
                (ProductMap.json_dump if PRODUCT_JSON else pickle.dump)(p.product_map[(entry_key,)], f)
        pmap_link = ProductMapLink(new_fpaths)
        p.product_map = pmap_link
        return p




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
        breakpoint()
        sk1 = Skeleton(self, -1)
        sk2 = Skeleton(other, -1)
        
        
        def serialize(tree, root, ans):
            bfs = deque([root])
            while len(bfs):
                cur = bfs.popleft()
                childs = list(tree[cur])                
                if len(childs):
                    if tree.nodes[childs[0]]['child'] == 'right':
                        childs = childs[::-1]
                    for c in childs:
                        bfs.append(c)
                else:
                    bfs += [0, 0]

        

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

    def __init__(self, st, index, whole_tree=None, zss_tree=None):
        """
        st: example of SyntheticTree with the skeleton
        This is a dual use class. It also remembers st for later use.
        """   
        i = 0 # chemical index
        j = 0 # reaction index
        interms = []
        whole_tree = nx.DiGraph()        
        for action in st.actions:
            n = len(whole_tree)-1
            if action == 0:                
                if st.reactions[j].rtype == 1:                    
                    whole_tree.add_node(n+1, smiles=st.chemicals[i].smiles, child='left')
                    whole_tree.add_node(n+2, rxn_id=st.reactions[j].rxn_id)
                    whole_tree.add_edge(n+2, n+1)
                    whole_tree.add_node(n+3, smiles=st.chemicals[i+1].smiles)
                    whole_tree.add_edge(n+3, n+2)
                    interms.append(n+3)
                    i += 2                    
                else:
                    child = st.reactions[j].child.index(st.chemicals[i].smiles)                    
                    whole_tree.add_node(n+1, smiles=st.chemicals[i].smiles, child=['left', 'right'][child])
                    whole_tree.add_node(n+2, smiles=st.chemicals[i+1].smiles, child=['right', 'left'][child])
                    whole_tree.add_node(n+3, rxn_id=st.reactions[j].rxn_id)
                    assert set(st.reactions[j].child) == set([st.chemicals[i].smiles, st.chemicals[i+1].smiles])
                    whole_tree.add_edge(n+3, n+1)
                    whole_tree.add_edge(n+3, n+2)
                    whole_tree.add_node(n+4, smiles=st.chemicals[i+2].smiles)
                    whole_tree.add_edge(n+4, n+3)
                    interms.append(n+4)
                    i += 3
                j += 1
            elif action == 1:
                a = interms[-1]
                if st.reactions[j].rtype == 1:
                    whole_tree.add_node(n+1, rxn_id=st.reactions[j].rxn_id)
                    whole_tree.add_edge(n+1, n)
                    whole_tree.nodes[n]['child'] = 'left'
                    whole_tree.add_node(n+2, smiles=st.chemicals[i].smiles)
                    whole_tree.add_edge(n+2, n+1)
                    interms[-1] = n+2
                    i += 1
                else:
                    child = st.reactions[j].child.index(whole_tree.nodes[n]['smiles'])
                    whole_tree.nodes[n]['child'] = ['left', 'right'][child]
                    whole_tree.add_node(n+1, smiles=st.chemicals[i].smiles, child=['right', 'left'][child])
                    whole_tree.add_node(n+2, rxn_id=st.reactions[j].rxn_id)                
                    whole_tree.add_edge(n+2, n)
                    whole_tree.add_edge(n+2, n+1)
                    whole_tree.add_node(n+3, smiles=st.chemicals[i+1].smiles)
                    whole_tree.add_edge(n+3, n+2)
                    interms[-1] = n+3
                    i += 2
                j += 1
            elif action == 2:
                a = interms[-1]
                b = interms[-2] # second
                assert st.reactions[j].rtype == 2
                child = st.reactions[j].child.index(whole_tree.nodes[a]['smiles'])
                whole_tree.nodes[a]['child'] = ['left', 'right'][child]
                whole_tree.nodes[b]['child'] = ['right', 'left'][child]
                whole_tree.add_node(n+1, rxn_id=st.reactions[j].rxn_id)
                whole_tree.add_node(n+2, smiles=st.chemicals[i].smiles)
                whole_tree.add_edge(n+1, a)
                whole_tree.add_edge(n+1, b)
                whole_tree.add_edge(n+2, n+1)
                interms[-1] = n+2
                i += 1
                j += 1
            elif action == 3:
                break

        self.tree = whole_tree            
        self.tree_edges = np.array(self.tree.edges).T        
        # self.tree_root = len(st.chemicals)-1
        self.tree_root = len(self.tree)-1
        self.non_root_tree_edges = self.tree_edges[:, (self.tree_edges != self.tree_root).all(axis=0)] # useful later
        self.leaves = np.array([((t not in self.tree_edges[0]) and t != self.tree_root) for t in range(len(self.tree))])                
        self.rxns = np.array(['rxn_id' in self.tree.nodes[n] for n in range(len(self.tree))])
        self.bidir_edges = np.concatenate((self.tree_edges, self.tree_edges[::-1]), axis=-1)
        self.index = index
        self.reset()


    def visualize(self, path=None, Xy=None, ax=None):
        pos = Skeleton.hierarchy_pos(self.tree, self.tree_root)
        if ax is None:
            pos_np = np.array([v for v in pos.values()])
            w, l = pos_np.max(axis=0)-pos_np.min(axis=0)
            fig = plt.Figure(figsize=(20*w, 20*l))
            ax = fig.add_subplot(1, 1, 1)        
        else:
            fig = None
        if Xy:
            X, y = Xy
            node_colors = []
            for i in self.tree:
                assert ((X[i].sum() > 0) + (y[i].sum() > 0)) <= 1
                if X[i].sum():
                    node_colors.append('green')
                elif y[i].sum():
                    node_colors.append('yellow')
                else:
                    node_colors.append('gray')
        else:
            node_colors = [['gray', 'red'][self.mask[n]] for n in self.tree]

        # nx.draw_networkx(self.tree, pos=pos, ax=ax, node_color=node_colors)
        # fig.savefig(path)
        # print(path)    

        node_labels = {}
        node_sizes = []
        for n in self.tree:
            if 'smiles' in self.tree.nodes[n]:
                smiles = self.tree.nodes[n]['smiles']
                m = int(math.sqrt(len(smiles)))
                l = (len(smiles)+m-1)//m
                smiles = '\n'.join([smiles[m*i:m*i+m] for i in range(l)])
                node_labels[n] = f"{n}: {smiles}"
                node_sizes.append(5000)
            else:
                rxn_id = self.tree.nodes[n]['rxn_id']
                node_labels[n] = f"{n}: {rxn_id}"
                node_sizes.append(1000)
        nx.draw_networkx(self.tree, pos=pos, ax=ax, 
                         node_color=node_colors, 
                         labels=node_labels,
                         node_size=node_sizes)
        if fig is not None:
            if path is not None:
                fig.savefig(path)
                print(os.path.abspath(path))


    def reset(self, mask=None):
        """
        Resets the mask to mask out every node
        """
        self._mask = np.zeros(len(self.tree), dtype=np.int8)
        self.leaves_up = True
        self.all_leaves = False        
        self.frontier = True # because we have target?
        if mask is not None:
            self.mask = mask

    
        
    def modify_tree(self, i, smiles=None, rxn_id=-1, suffix=''):
        """
        Fills node i with smiles or rxn_id
        If suffix is given, add it to the attribute
        """
        if smiles is not None:
            if 'smiles' not in self.tree.nodes[i]:
                breakpoint()
            self.tree.nodes[i]['smiles'+suffix] = smiles
        if rxn_id != -1:
            if 'rxn_id' not in self.tree.nodes[i]:
                breakpoint()
            self.tree.nodes[i]['rxn_id'+suffix] = rxn_id
        self.mask = [i]


    def clear_tree(self, save=[], forcing=False):
        """
        Clears the semantic information in the tree    
        """
        for n in self.tree:
            if n in save:
                continue
            if 'smiles' in self.tree.nodes[n]:
                if forcing:
                    self.tree.nodes[n]['smiles_forcing'] = ''    
                else:
                    self.tree.nodes[n]['smiles'] = ''
            elif 'rxn_id' in self.tree.nodes[n]:
                if forcing:
                    self.tree.nodes[n]['rxn_id_forcing'] = -1
                else:
                    self.tree.nodes[n]['rxn_id'] = -1
                self.tree.nodes[n]['smirks'] = 'C>>' # for debug hashing
                # self.tree.nodes[n]['rxn_id'] = -1
            else:
                raise
        self.reset()

    
    def reconstruct(self, rxns):
        postorder = list(nx.dfs_postorder_nodes(self.tree, source=self.tree_root))
        for i in self.tree:
            if not self.leaves[i]:
                if 'smiles' in self.tree.nodes[i]:
                    self.tree.nodes[i]['smiles'] = ''
        for i in postorder:
            if self.rxns[i]:
                succ = list(self.tree.successors(i))
                if self.tree.nodes[succ[0]]['child'] == 'right':
                    succ = succ[::-1]
                reactants = tuple(self.tree.nodes[j]['smiles'] for j in succ)
                if len(reactants) != rxns[self.tree.nodes[i]['rxn_id']].num_reactant:
                    return False
                interm = rxns[self.tree.nodes[i]['rxn_id']].run_reaction(reactants)              
                pred = list(self.tree.predecessors(i))[0]
                if interm is None:
                    return False
                self.tree.nodes[pred]['smiles'] = interm
        smi1 = Chem.CanonSmiles(self.tree.nodes[self.tree_root]['smiles'])                


    @property
    def mask(self):
        return self._mask

    
    def pred(self, n):
        return list(self.tree.predecessors(n))[0]
    
    @staticmethod
    def lowest_rxns(tree, cur, ans): # returns depth
        # get all nodes whose bottom-up depth is in max_depths
        depth = 0
        if tree[cur]:
            depth = 1+max([Skeleton.lowest_rxns(tree, j, ans) for j in tree[cur]])
        ans[cur] = depth
        return depth


    @staticmethod
    def num_rxns(tree, cur, nums):
        count = 'rxn_id' in tree.nodes[cur]
        for nei in tree[cur]:
            count += Skeleton.num_rxns(tree, nei, nums)
        nums[cur] = count        
        return count


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

        # check that if true, all parent of parent is true
        non_target_interms = (self.mask & ~self.rxns)
        non_target_interms[self.tree_root] = 0
        self.rxn_target_down = not non_target_interms.any() # all non-rxn nodes masked out, and rxns target_down
        non_target_interms[self.leaves] = 0
        self.rxn_target_down_bb = not non_target_interms.any() # all interm nodes masked out, and rxns target_down

        for n in np.argwhere((self.mask & self.rxns)).flatten().tolist(): # check rxns
            if self.pred(n) == self.tree_root: continue
            parent_rxn = self.pred(self.pred(n))
            if not (self.mask & self.rxns)[parent_rxn]:
                self.rxn_target_down = False
                self.rxn_target_down_bb = False
        for n in np.argwhere((self.mask & self.leaves)).flatten().tolist():
            if not self.mask[self.pred(n)]:
                self.rxn_target_down_bb = False

        # bottom-most 2 reactions
        nodes = {}
        self.lowest_rxns(self.tree, self.tree_root, nodes)        
        self.bottom_2_rxns = [n for n in nodes if nodes[n] in [1,3]]
        correct_mask = [i for i in range(len(self.tree)) if i == self.tree_root or i in self.bottom_2_rxns]
        self.leaf_up_2 = np.argwhere(self.mask).flatten().tolist() == correct_mask
        
    
    @staticmethod
    def one_hot(n, ind):
        zeros = np.zeros(n)
        zeros[ind] = 1.
        return zeros
    

    def fill_node(self, n, y):
        if 'smiles' in self.tree.nodes[n]:
            if self.tree.nodes[n]['smiles']:
                y[n][:256] = fp_256(self.tree.nodes[n]['smiles'])    
                # print(self.tree.nodes[n])            
            else:
                pass
                # print("bad smiles")
        elif 'rxn_id' in self.tree.nodes[n]:
            if self.tree.nodes[n]['rxn_id'] != -1:
                y[n][256:] = self.one_hot(91,self.tree.nodes[n]['rxn_id'])
            else:
                pass
                # print("bad rxn_id")
        else:
            print("bad node")

           
    def get_state(self, leaves_up=False, rxn_frontier=False, bb_frontier=False, target_down=False, rxn_target_down=False, rxn_target_down_bb=False):
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
        rxn_target_down: same as leaves_up but from target down, and for reactions only
        """        
        X = np.zeros((len(self.tree), 2*2048+91))
        y = np.zeros((len(self.tree), 256+91))
        try:
            for n in self.tree.nodes():             
                # is target, or parent is target, or parent rxn is fulfilled
                rxn_parent = (n == self.tree_root) or self.pred(n) == self.tree_root or (self.mask & self.rxns)[self.pred(self.pred(n))]
                # if is rxn, then rxn_parent; if is leaf, then parent is fulfilled        
                rxn_parent_bb = (not self.rxns[n] or rxn_parent) and (not self.leaves[n] or self.mask[self.pred(n)]) and (self.rxns[n] or self.leaves[n])
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
                        # X[n][:2048] = fp_2048(self.tree.nodes[self.pred(n)]['smiles'])
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
                        if rxn_target_down:        
                            if rxn_target_down_bb and rxn_parent_bb:
                                # accomodate bb's
                                self.fill_node(n, y)
                            elif not rxn_target_down_bb and rxn_parent:
                                # don't accomodate bb's
                                self.fill_node(n, y)
                        elif target_down:
                            if is_frontier:
                                self.fill_node(n, y)
                        else:
                            if not leaves_up or leaves_filled:
                                self.fill_node(n, y)
        except Exception as e:
            breakpoint()
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
        From Joel's answer at https://stackoverflow.com/a/29597209/2966723
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

                
        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


    @staticmethod
    def ego_graph(graph, cur, level):
        # re-index first, making sure the indexing is consistent with post-order during program synthesis
        # bottom-first, left-first                    
        graph = nx.ego_graph(graph, cur, level)
        relabel = dict(zip(nx.dfs_postorder_nodes(graph), range(len(graph))))
        graph = nx.relabel_nodes(graph, relabel)         
        dists = dict(nx.shortest_path_length(graph, source=relabel[cur]))
        max_depth = max(dists.values())+1
        for tgt in dists:
            graph.nodes[tgt]['depth'] = max_depth-dists[tgt]
        return graph



    @staticmethod
    def label_depth(g, root):
        dq = deque()
        dq.append(root)
        g.nodes[root]['depth'] = 1
        max_depth = 1
        while len(dq):
            cur = dq.popleft()
            for nex in g[cur]:
                d = g.nodes[cur]['depth']+1
                g.nodes[nex]['depth'] = d
                dq.append(nex)
                max_depth = max(max_depth, d)
        for n in g:
            g.nodes[n]['depth'] = max_depth-g.nodes[n]['depth']+1



    def rxn_graph(self):
        g = nx.induced_subgraph(self.tree, np.argwhere(self.rxns).flatten())
        g = g.copy()
        root = None
        for a in g:
            for b in g:
                if self.pred(b) == self.tree_root:
                    root = b
                    continue                
                if self.pred(self.pred(b)) == a:
                    child = self.tree.nodes[self.pred(b)]['child']                        
                    g.add_edge(a, b)
                    g.nodes[b]['child'] = child    
        
        Skeleton.label_depth(g, root)        
        node_map = dict(zip(g.nodes(), range(self.rxns.sum())))
        reverse_node_map = dict(zip(range(self.rxns.sum()), g.nodes()))
        g = nx.relabel_nodes(g, node_map)
        return g, node_map, reverse_node_map
    

    def rxn_prog(self):
        g, _, reverse_node_map = self.rxn_graph()
        p = Program(g)
        dists = nx.shortest_path_length(self.tree, self.tree_root)
        max_dist = 0
        for i in g:
            g.nodes[i]['depth'] = dists[reverse_node_map[i]]//2
            max_dist = max(g.nodes[i]['depth'], max_dist)            
        root = 0
        for i in g:
            g.nodes[i]['depth'] = max_dist+1-g.nodes[i]['depth']
            if g.nodes[i]['depth'] > g.nodes[root]['depth']:
                root = i                       
        return p       


    def hash(self):
        """
        Build rxn tree        
        """
        p = self.rxn_prog()      
        val = p.hash(self.mask[self.rxns])
        return val



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


def inds_to_i(inds, length, min_r_set):
    """
    compute i using indices outside min_r_set
    """
    zeros = np.zeros((length,), dtype=int)
    zeros[inds] = 1
    zeros[min_r_set] = -1    
    return int(''.join(map(str, filter(lambda x: x !=- 1, zeros))), 2)


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
    elif args.determine_criteria == 'rxn_target_down':
        kwargs['rxn_target_down'] = True
    elif args.determine_criteria == 'rxn_target_down_bb':
        kwargs['rxn_target_down'] = True
        kwargs['rxn_target_down_bb'] = True    
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
    else:
        sk.reset(min_r_set)
        zero_mask_inds = np.where(sk.mask == 0)[0]    
        bool_mask = get_bool_mask(i)
        sk.mask = zero_mask_inds[-len(bool_mask):][bool_mask]   
        node_mask, X, y = sk.get_state(**kwargs)        
        if args.determine_criteria == 'all_leaves':
            assert sk.all_leaves
    

    # visualize to help debug
    if not os.path.exists(os.path.join(args.visualize_dir, f"{sk.index}_{i}.png")):
        sk.visualize(os.path.join(args.visualize_dir, f"{sk.index}_{i}.png"))
        sk.visualize(os.path.join(args.visualize_dir, f"{sk.index}_{i}_Xy.png"), Xy=(X, y))

    return (node_mask, X, y, sk.tree.nodes[sk.tree_root]['smiles'])        


def test_is_leaves_up(i, sk, min_r_set):
    sk.reset(min_r_set)
    zero_mask_inds = np.where(sk.mask == 0)[0]    
    bool_mask = get_bool_mask(i)
    sk.mask = zero_mask_inds[-len(bool_mask):][bool_mask]   
    return sk.leaves_up


def load_skeletons(args):
    syntree_collection = SyntheticTreeSet()

    if os.path.exists(args.skeleton_file):
        skeletons = pickle.load(open(args.skeleton_file, 'rb'))
        if args.skeleton_canonical_file:
            canon_skeletons = pickle.load(open(args.skeleton_canonical_file, 'rb'))
            class_nums = {k: len(canon_skeletons[k]) for k in canon_skeletons}        
        else:
            return skeletons
      
        # sanity checks
        for sk, sk_canon in zip(skeletons, canon_skeletons):
            if sk.edges != sk_canon.edges:
                breakpoint()                
    else:
        sts = []
        for st in syntree_collection.sts:
            if st: 
                try:
                    st.build_tree()
                except:
                    breakpoint()
                sts.append(st)
            else:
                breakpoint()
        
        # use the train set to define the skeleton classes
        if args.skeleton_canonical_file:
            skeletons = pickle.load(open(args.skeleton_canonical_file, 'rb'))
            class_nums = {k: len(skeletons[k]) for k in skeletons}
        else:
            skeletons = {}
        for i, st in tqdm(enumerate(sts)):
            done = False
            for sk in skeletons:
                if st.is_isomorphic(sk): 
                    done = True
                    skeletons[sk].append(st)
                    break
                    
            if not done: 
                skeletons[st] = [st]
        if args.skeleton_canonical_file:
            if list(class_nums.keys()) != list(skeletons.keys()):
                breakpoint()
            for k in class_nums:
                skeletons[k] = skeletons[k][class_nums[k]:]
        for k, v in skeletons.items():
            print(f"count: {len(v)}") 

        pickle.dump(skeletons, open(args.skeleton_file, 'wb+'))
    return skeletons



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
