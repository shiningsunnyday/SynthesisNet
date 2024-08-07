# SynNet

This repo contains the code and analysis scripts for our amortized approach to synthetic tree generation using neural networks.
Our model can serve as both a synthesis planning tool and as a tool for synthesizable molecular design.

The method is described in detail in the publication "Amortized tree generation for bottom-up synthesis planning and synthesizable molecular design" available on the [arXiv](https://arxiv.org/abs/2110.06389) and summarized below.

## Summary

We model synthetic pathways as tree structures called *synthetic trees*.
A synthetic tree has a single root node and one or more child nodes.
Every node is chemical molecule:

- The root node is the final product molecule
- The leaf nodes consist of purchasable building blocks.
- All other inner nodes are constrained to be a product of allowed chemical reactions.

At a high level, each synthetic tree is constructed one reaction step at a time in a bottom-up manner, that is starting from purchasable building blocks.

### Overview

The model consists of four modules, each containing a multi-layer perceptron (MLP):

1. An *Action Type* selection function that classifies action types among the four possible actions (“Add”, “Expand”, “Merge”, and “End”) in building the synthetic tree. Each action increases the depth of the synthetic tree by one.

2. A *First Reactant* selection function that selects the first reactant. A MLP predicts a molecular embedding and a first reactant is identified from the pool of building blocks through a k-nearest neighbors (k-NN) search.

3. A *Reaction* selection function whose output is a probability distribution over available reaction templates. Inapplicable reactions are masked based on reactant 1. A suitable template is then sampled using a greedy search.

4. A *Second Reactant* selection function that identifies the second reactant if the sampled template is bi-molecular. The model predicts an embedding for the second reactant, and a candidate is then sampled via a k-NN search from the masked set of building blocks.

![the model](./figures/network.png "model scheme")

These four modules predict the probability distributions of actions to be taken within a single reaction step, and determine the nodes to be added to the synthetic tree under construction.
All of these networks are conditioned on the target molecule embedding.

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
