from dataclasses import dataclass
from typing import List, Literal, Optional

import networkx as nx
import numpy as np
import pydantic


ORACLE_REGISTRY = {
    "qed": "QED",
    "logp": "LogP",
    "jnk": "JNK3",
    "gsk": "GSK3B",
    "drd2": "DRD2",
    "7l11": "7l11_docking",
    "drd3": "drd3_docking",
    "median1": "Median 1",
    "median2": "Median 2",
    "osimertinib": "Osimertinib_MPO",
    "fexofenadine": "Fexofenadine_MPO",
    "ranolazine": "Ranolazine_MPO",
    "perindopril": "Perindopril_MPO",
    "amlodipine": "Amlodipine_MPO",
    "sitagliptin": "Sitagliptin_MPO",
    "zaleplon": "Zaleplon_MPO",
    "celecoxib": "Celecoxib_Rediscovery",
}


class GeneticSearchConfig(pydantic.BaseModel):
    """Configuration object for a genetic search.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    seed: int = 10

    # Individuals
    fp_bits: int = 2048
    bt_ignore: bool = False

    # Search parameters
    objective: Literal[tuple(ORACLE_REGISTRY)] = "qed"

    generations: int = 200
    population_size: int = 128
    offspring_size: int = 512

    max_oracle_calls: int = int(1e7)
    max_oracle_workers: int = 5

    # Mutation
    fp_mutate_prob: float = 0.5
    fp_mutate_frac: float = (24 / 4096)
    bt_mutate_edits: int = 3

    children_per_couple: int = 2
    children_strategy: Literal["edits", "flips", "topk"] = "edits"

    # Restrict skeleton prediction to max number of reactions
    max_num_rxns: int = -1

    # Early stopping
    early_stop: bool = False
    early_stop_delta: float = 0.01
    early_stop_warmup: int = 30
    early_stop_patience: int = 10

    # IO
    initialize_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    background_set_file: str

    # WandB
    wandb: bool = False
    wandb_project: str = "syntreenet"
    wandb_entity: str = "lpft"
    wandb_dir: Optional[str] = None


@dataclass
class Individual:
    """Individual object for a genetic search.
    """

    # Search space objects
    fp: np.ndarray
    bt: nx.DiGraph

    # Set by fitness function
    fitness: Optional[float] = None
    smiles: Optional[str] = None