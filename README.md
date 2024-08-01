# SynTreeNet

This repo contains the code and analysis scripts for our Syntax-Guided approach for the Procedural Synthesis of Molecules. Similar to SynNet, our model serves both Synthesizable Analog Generation and Synthesizable Molecular Design applications. Upon acceptance for publication, we will include a link to the preprint with the full details of our method.

### Environment

```bash
conda env create -f environment.yml
```

### Data

Go to Enmaine's [catalog](https://enamine.net/building-blocks/building-blocks-catalog) to obtain the building blocks. We used the "Building Blocks, US Stock" data. You need to first register and then request access to download the dataset. The result is a .sdf file.

The reaction templates (from Hartenfeller-Button) are in data/assets/hb.txt.

1) Extract SMILES from the .sdf file from enamine.net.
```bash
python scripts/00-extract-smiles-from-sdf.py \
    --input-file="data/assets/building-blocks/enamine-us.sdf" \
    --output-file="data/assets/building-blocks/enamine-us-smiles.csv.gz"
``````

2) Filter building blocks. 
```bash
python scripts/01-filter-building-blocks.py \
    --building-blocks-file "data/assets/building-blocks/enamine-us-smiles.csv.gz" \
    --rxn-templates-file "data/assets/reaction-templates/hb.txt" \
    --output-bblock-file "data/assets/building-blocks/enamine_us_matched.csv" \
    --output-rxns-collection-file "data/assets/reaction-templates/reactions_hb.json.gz" --verbose
```

3) Embed building blocks.
```bash
python scripts/02-compute-embeddings.py \
    --building-blocks-file "data/assets/building-blocks/enamine_us_matched.csv" \
    --output-file "data/assets/building-blocks/enamine_us_emb_fp_256.npy" \
    --featurization-fct "fp_256"
```

It's helpful to set the following environmental variables from now on.

```bash
export BUILDING_BLOCKS_FILE=data/assets/building-blocks/enamine_us_matched.csv
export RXN_TEMPLATE_FILE=data/assets/reaction-templates/hb.txt
export RXN_COLLECTION_FILE=data/assets/reaction-templates/reactions_hb.json.gz
export EMBEDDINGS_KNN_FILE=data/assets/building-blocks/enamine_us_emb_fp_256.npy
```

### Overview

![overview](./data/assets/figs/fig1.png "terminologies")

Our innovation is to model synthetic pathways as *programs*. This section overviews the basic concepts for understanding the core ideas of our work. Terminologies from program synthesis are italicized.
In computers, programs are first parsed into a tree-like representation called a *syntax tree*. The syntax tree is closely related to synthetic trees (see [SynNet](https://github.com/wenhao-gao/SynNet)) where:
- Each leaf node is a *literal*: chemical building block (*B*)
- Each intermediate node is an *operator*: chemical reaction template (*R*)
- Each root node stores the *output*: product

Syntax arises from derivations of a *grammar*. A grammar  Our grammar contain basic chemical building blocks, reactions (uni-molecular and bi-molecular) and, more insightfuly, *syntactic templates* (*T*) to constrain the space of derivations.

*Syntactic templates* are the skeletons of a complete syntax tree. They are known as user-provided *sketches* and are used by program synthesis techniques to constrain the search space. Our framework allows users to provide these skeletons, but we can automatically extract them by first sampling a large number of synthetic trees, filter them, then extracting the skeletons present among them.

Sample 600000 synthetic trees.
```bash
python scripts/03-generate-syntrees.py --building-blocks-file $BUILDING_BLOCKS_FILE --rxn-templates-file $RXN_TEMPLATE_FILE --output-file "data/pre-process/syntrees/synthetic-trees.json.gz" --number-syntrees "600000"
```

Filter to only those that produce chemically valid molecules and are pass a QED threshold. You can customize with additional filters.
```bash
python scripts/04-filter-syntrees.py --input-file "data/pre-process/syntrees/synthetic-trees.json.gz" --output-file "data/pre-process/syntrees/synthetic-trees-filtered.json.gz" --verbose
```

Split into training, valid, test sets.
```bash
python scripts/05-split-syntrees.py --input-file "data/pre-process/syntrees/synthetic-trees-filtered.json.gz" --output-dir "data/pre-process/syntrees/" --verbose
```

Extract the skeletons and perform exploratory data analysis.
```bash
mkdir results/viz/
python scripts/analyze-skeletons.py \
    --skeleton-file results/viz/skeletons.pkl \
    --input-file data/pre-process/syntrees/synthetic-trees-filtered.json.gz \
    --visualize-dir results/viz/
```

This creates a canonical numbering over skeleton classes. skeletons.pkl is a dictionary, mapping each present syntactic template to a list of synthetic trees isomorphic to that template. Each key is a reference synthetic tree, and the value is a list of synthetic trees conforming to the same class.

Partition the skeletons.pkl dictionary into train, valid, test while keeping the canonical keys the same.
```bash
for split in {'train','valid','test'}; do
    python scripts/analyze-skeletons.py \
        --skeleton-file results/viz/skeletons-${split}.pkl \
        --input-file data/pre-process/syntrees/synthetic-trees-filtered-${split}.json.gz \
        --visualize-dir results/viz/ \
        --skeleton-canonical-file results/viz/skeletons.pkl
done
```

### Synthesizable Analog Generation

![overview](./data/assets/figs/fig2.png "model scheme")

Our surrogate ($F$) tackles the following search problem: given a *specification* in the form of a Morgan Fingerprint (over domain $X$), synthesize a program whose output molecule has that fingerprint. We introduce a bi-level solution, with an outer level proposing syntactic templates and the inner level inferencing a policy network to fill in the template.

### Training a Surrogate Model
![overview](./data/assets/figs/fig3.png "model scheme")

In summary, our inner surrogate model takes as input a skeleton (in $T$) and fingerprint (in $X$), and fills in the holes to infer a complete syntax tree. This amortizes over solving the finite horizon MDP induced by the template, with the goal state being a complete syntax tree whose output molecule has the fingerprint.

Our supervised policy network is a GNN that takes as input a partially filled in syntax tree, and predicts an operator/literal for the nodes on the frontier. To train it, we construct a dataset of partial syntax trees via imposing masks over the synthetic trees in our data.

```bash
max_depth=3
criteria=rxn_target_down_interm
dataset=gnn_featurized_${criteria}_postorder
for split in {'train','valid','test'}; do
    # Make the directories
    mkdir -p data/${dataset}_max_depth=${max_depth}_${split}
    mkdir -p data/${dataset}_max_depth=${max_depth}_split_${split}    

    # Enumerate partial trees
    python scripts/process-for-gnn.py \
        --determine_criteria ${criteria} \
        --output-dir data/${dataset}_max_depth=${max_depth}_${split} \
        --anchor_type target \
        --visualize-dir results/viz/ \
        --skeleton-file results/viz/skeletons-${split}.pkl \
        --max_depth ${max_depth} \
        --ncpu 100 \
        --num-trees-per-batch 5000

    # Partition each batch into individual files, so data loading becomes I/O-bound
    python scripts/split_data.py \
        --in-dir data/${dataset}_max_depth=${max_depth}_${split}/ \
        --out-dir data/${dataset}_max_depth=${max_depth}_split_${split}/ \
        --partition_size 1        
done;
``````

### Synthesizable Analog Generation

### Synthesis planning

This task is to infer the synthetic pathway to a given target molecule.
We formulate this problem as generating a synthetic tree such that the product molecule it produces (i.e., the molecule at the root node) matches the desired target molecule.

For this task, we can take a molecular embedding for the desired product, and use it as input to our model to produce a synthetic tree.
If the desired product is successfully recovered, then the final root molecule will match the desired molecule used to create the input embedding.
If the desired product is not successully recovered, it is possible the final root molecule may still be *similar* to the desired molecule used to create the input embedding, and thus our tool can also be used for *synthesizable analog recommendation*.

![the generation process](./figures/generation_process.png "generation process")

### Synthesizable molecular design

This task is to optimize a molecular structure with respect to an oracle function (e.g. bioactivity), while ensuring the synthetic accessibility of the molecules.
We formulate this problem as optimizing the structure of a synthetic tree with respect to the desired properties of the product molecule it produces.

To do this, we optimize the molecular embedding of the molecule using a genetic algorithm and the desired oracle function.
The optimized molecule embedding can then be used as input to our model to produce a synthetic tree, where the final root molecule corresponds to the optimized molecule.

## Setup instructions

### Environment

Conda is used to create the environment for running SynNet.

```bash
# Install environment from file
conda env create -f environment.yml
```

Before running any SynNet code, activate the environment and install this package in development mode:

```bash
source activate synnet
pip install -e .
```

The model implementations can be found in `src/syn_net/models/`.

The pre-processing and analysis scripts are in `scripts/`.

### Train the model from scratch

Before training any models, you will first need to some data preprocessing.
Please see [INSTRUCTIONS.md](INSTRUCTIONS.md) for a complete guide.

### Data

SynNet relies on two datasources:

1. reaction templates and
2. building blocks.

The data used for the publication are 1) the *Hartenfeller-Button* reaction templates, which are available under  [data/assets/reaction-templates/hb.txt](data/assets/reaction-templates/hb.txt) and 2) *Enamine building blocks*.
The building blocks are not freely available.

To obtain the data, go to [https://enamine.net/building-blocks/building-blocks-catalog](https://enamine.net/building-blocks/building-blocks-catalog).
We used the "Building Blocks, US Stock" data. You need to first register and then request access to download the dataset. The people from enamine.net manually approve you, so please be nice and patient.

## Reproducing results

Before running anything, set up the environment as decribed above.

### Using pre-trained models

We have made available a set of pre-trained models at the following [link](https://figshare.com/articles/software/Trained_model_parameters_for_SynNet/16799413).
The pretrained models correspond to the Action, Reactant 1, Reaction, and Reactant 2 networks, trained on the *Hartenfeller-Button* dataset and *Enamine* building blocks using radius 2, length 4096 Morgan fingerprints for the molecular node embeddings, and length 256 fingerprints for the k-NN search.
For further details, please see the publication.

To download the pre-trained model to `./checkpoints`:

```bash
# Download
wget -O hb_fp_2_4096_256.tar.gz https://figshare.com/ndownloader/files/31067692
# Extract
tar -vxf hb_fp_2_4096_256.tar.gz
# Rename files to match new scripts (...)
mv hb_fp_2_4096_256/ checkpoints/
for model in "act" "rt1" "rxn" "rt2"
do
  mkdir checkpoints/$model
  mv "checkpoints/$model.ckpt" "checkpoints/$model/ckpts.dummy-val_loss=0.00.ckpt"
done
rm -f hb_fp_2_4096_256.tar.gz
```

The following scripts are run from the command line.
Use `python some_script.py --help` or check the source code to see the instructions of each argument.

### Prerequisites

In addition to the necessary data, we will need to pre-compute an embedding of the building blocks.
To do so, please follow steps 0-2 from the [INSTRUCTIONS.md](INSTRUCTIONS.md).
Then, replace the environment variables in the commands below.

#### Synthesis Planning

To perform synthesis planning described in the main text:

```bash
python scripts/20-predict-targets.py \
    --building-blocks-file $BUILDING_BLOCKS_FILE \
    --rxns-collection-file $RXN_COLLECTION_FILE \
    --embeddings-knn-file $EMBEDDINGS_KNN_FILE \
    --data "data/assets/molecules/sample-targets.txt" \
    --ckpt-dir "checkpoints/" \
    --output-dir "results/demo-inference/"
```

python scripts/20-predict-targets.py \
    --building-blocks-file $BUILDING_BLOCKS_FILE \
    --rxns-collection-file $RXN_COLLECTION_FILE \
    --embeddings-knn-file $EMBEDDINGS_KNN_FILE \
    --data "data/pre-process/syntrees/top_1000/synthetic-trees-filtered-test.json.gz" \
    --ckpt-dir "results/logs/" \
    --output-dir "results/top-1000-inference/"

This script will feed a list of ten molecules to SynNet.

#### Synthesizable Molecular Design

To perform synthesizable molecular design, run:

```bash
python scripts/optimize_ga.py \
    --ckpt-dir "checkpoints/" \
      --building-blocks-file $BUILDING_BLOCKS_FILE \
    --rxns-collection-file $RXN_COLLECTION_FILE \
    --embeddings-knn-file $EMBEDDINGS_KNN_FILE \
    --input-file path/to/zinc.csv \
    --radius 2 --nbits 4096 \
    --num_population 128 --num_offspring 512 --num_gen 200 --objective gsk \
    --ncpu 32
```

This script uses a genetic algorithm to optimize molecular embeddings and returns the predicted synthetic trees for the optimized molecular embedding.

Note: `input-file` contains the seed molecules in CSV format for an initial run, and as a pre-saved numpy array of the population for restarting the run. If omitted, a random fingerprint will be chosen.
