"""Central place for all configuration, paths, and parameter."""
import multiprocessing

# Multiprocessing
MAX_PROCESSES = min(100, multiprocessing.cpu_count()) - 1
# MP_MIN_COMBINATIONS = 10000000
MP_MIN_COMBINATIONS = 10000000000000000000000000000000000

NUM_THREADS = 1

# TODO: Remove these paths bit by bit

# Intermediates
# PRODUCT_DIR = "/ssd/msun415/program_cache-bb=10000-prods=1_new_product_map"
PRODUCT_DIR = "/dccstor/graph-design/program_cache_keep-prods=2/"
PRODUCT_JSON = True
DELIM = '_____'
MAX_DEPTH = 2
NUM_POSS = 91

# Pre-processed data
DATA_PREPROCESS_DIR = "data/pre-process"

# Prepared data
DATA_FEATURIZED_DIR = "data/featurized"

# Results
DATA_RESULT_DIR = "results"

# Checkpoints (& pre-trained weights)
CHECKPOINTS_DIR = "checkpoints"
