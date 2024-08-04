# SynTreeNet

This repo contains the code and analysis scripts for our Syntax-Guided approach for the Procedural Synthesis of Molecules. Similar to SynNet, our model serves both Synthesizable Analog Generation and Synthesizable Molecular Design applications. Upon acceptance for publication, we will include a link to the preprint with the full details of our method.

### Environment

```bash
conda env create -f environment.yml
source activate syntreenet
pip install -e .
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

### Training a Surrogate Model
![model](./data/assets/figs/fig3.png "model scheme")

In summary, our surrogate model takes as input a skeleton (in $T$) and fingerprint (in $X$), and fills in the holes to infer a complete syntax tree. This amortizes over solving the finite horizon MDP induced by the template, with the goal state being a complete syntax tree whose output molecule has the fingerprint.

Our supervised policy network is a GNN that takes as input a partially filled in syntax tree, and predicts an operator or literal for the nodes on the frontier. To train it, we construct a dataset of partial syntax trees via imposing masks over the synthetic trees in our data then preprocessing the target y for each frontier node.

There is a tradeoff between expressiveness and usefulness when selecting which skeleton classes to use. For our final results, we find limiting to only the skeletons with at most 4 reactions (max_depth=4) is a good compromise between model coverage and real-world cost considerations.

```bash
max_depth=4
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

Now we can train a GNN surrogate. By default, all skeleton classes from the previous step are used, but you can train separate checkpoints for separate classes. For example, if you only want a model over partial trees belonging to skeleton classes 0 and 1, you can add --gnn-datasets 0 1. We train two models ($F_B$ and $F_R$), one for predicting only building blocks and one for predicting only reactions. 

```bash
ckpt_dir=/ssd/msun415/surrogate/ # replace with your directory
num_cpu=50 # change as needed
if [[ $1 -eq 1 ]]; # train F_B
then
        metric=nn_accuracy_loss;
        loss=mse;
else # train F_R
        metric=accuracy_loss;
        loss=cross_entropy;
fi
python src/synnet/models/gnn.py \
        --gnn-input-feats data/${dataset}_max_depth=${max_depth}_split \
        --results-log ${ckpt_dir} \
        --mol-embedder-file $EMBEDDINGS_KNN_FILE \
        --gnn-valid-loss ${metric} \
        --gnn-loss ${loss} \
        --gnn-layer Transformer \
        --lazy_load \
        --ncpu ${num_cpu} \
        --prefetch_factor 0 \
        --feats-split \
        --cuda 0 \
        --pe sin
``````    

The results and checkpoints will be stored in versioned folders. See gnn.py for the hyperparameters that affect performance. You can iterate as needed. For example, if your best checkpoints are in version_42 (for $F_B$) and version_43 (for $F_R$), you should rename the folders for the downstream applications:

```bash
mv ${ckpt_dir}/version_42/ ${ckpt_dir}/4-NN
mv ${ckpt_dir}/version_43/ ${ckpt_dir}/4-RXN
```

### Synthesizable Analog Generation

The task of synthesizable analog generation is to find a synthetic pathway to produce a molecule that's as similar to a given target molecule as possible. We define similarity between molecules as Tanimoto similarity over their fingerprints.

![analog](./data/assets/figs/fig2.png "model scheme")

Our surrogate procedure ($F$) tackles the following problem: given a *specification* in the form of a Morgan Fingerprint (over domain $X$), synthesize a program whose output molecule has that fingerprint. We introduce a bi-level solution, with an outer level proposing syntactic templates and the inner level inferencing our trained policy network to fill in the template.


See our bash script for the hyperparameters. Remember to specify the surrogate model checkpoint directory correctly.
```bash
./scripts/mcmc-analog.sh
``````

This script assumes you have listener processes in the background to parallelize across the batch. Each listener process can be called by running. Make sure the hyperparameters you want to try are also specified in it.
```bash
./scripts/mcmc-analog-listener-ccc.sh ${NUM_PROCESSES}
``````

The listeners coordinate via sender-filename and receiver-filename in mcmc-analog.sh. You can remove those args if you don't want to launch listener processes.

### Synthesizable molecular design
![ga](./data/assets/figs/ga.png "GA")

The task is to optimize over synthetic pathways with respect to the property of its output molecule. We use the property oracles in [PMO](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html) and [GuacaMol](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839), as implemented by [TDC](https://tdc.readthedocs.io/en/main/). 

We implement our discrete optimization procedure using a bilevel Genetic Search + Bayesian Optimization strategy. In the outer level, we follow a similar crossover and mutation strategy as [SynNet](https://github.com/wenhao-gao/SynNet) over fingerprints *X* but exercising our novel capability of controlling the skeleton. In the inner level, we introduce operators over skeletons *$\mathcal{T}$*. The inner procedure's goal is to explore the synthesizable analog space of a given fingerprint $X$ (as in the Synthesizable Analog Generation task) by trying different syntactic templates. Since MCMC is prohibitively expensive, we offer various ways to amortize over it. By default, we use the top k skeletons proposed by our trained recognition model. This gives us $T_1, \ldots, T_k$.

We use our surrogate model to decode $F(X, T_1), \ldots, F(X, T_K)$ with the given fingerprint and the proposed skeletons to obtain a sibling pool of molecules (and their fingerprints $\hat{X}_1, \ldots, \hat{X}_k$). Then, we inference the Gaussian Process regressor and apply a standard EI acquisition function to select the fingerprint to evaluate with the oracle. We refit the regressor after each generation with the history of oracle calls.
