from dataclasses import dataclass
from typing import Literal, Optional

import networkx as nx
import numpy as np
import pydantic
import pydantic_cli


class GeneticSearchConfig(pydantic.BaseModel):
    """Configuration object for a genetic search.
    """

    seed: int = 10

    # Individuals
    fp_bits: int = 2048
    bt_nodes_min: int = 2
    bt_nodes_max: int = 10

    # Search parameters
    generations: int = 200
    population_size: int = 128
    offspring_size: int = 512

    freeze_fp: bool = False
    freeze_bt: bool = False

    # Mutation
    fp_mutate_prob: float = 0.5
    fp_mutate_frac: float = (24 / 4096)

    bt_mutate_prob: float = 0.5
    bt_mutate_edits: int = 2

    # Early stopping
    early_stop_delta: float = 0.01
    early_stop_warmup: int = 30
    early_stop_patience: int = 10

    # IO
    initialize_path: str = None
    checkpoint_path: str = None
    background_set_file: str

    # WandB
    wandb: bool = False
    wandb_project: str = "syntreenet_ga"
    wandb_dir: Optional[str] = None

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


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