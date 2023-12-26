"""Central place for all configuration, paths, and parameter."""
import multiprocessing

# Multiprocessing
MAX_PROCESSES = min(100, multiprocessing.cpu_count()) - 1
MP_MIN_COMBINATIONS = 1000000

NUM_THREADS = 1

# TODO: Remove these paths bit by bit

# Intermediates
PRODUCT_DIR = "/lfs/local/0/msun415/program_cache-bb=10000_keep-prods=2"
PRODUCT_JSON = True

# Pre-processed data
DATA_PREPROCESS_DIR = "data/pre-process"

# Prepared data
DATA_FEATURIZED_DIR = "data/featurized"

# Results
DATA_RESULT_DIR = "results"

# Checkpoints (& pre-trained weights)
CHECKPOINTS_DIR = "checkpoints"
