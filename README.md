# Triplet Matching Network

The source code used for self-supervised taxonomy completion method [TMN](), published in AAAI 21.

Please cite the following work if you find the code useful.

```
@inproceedings{zhang2021tmn,
	Author = {Zhang, Jieyu and Song, Xiangchen and Zeng, Ying and Chen, Jiaze and Shen, Jiaming and Mao, Yuning and Li, Lei},
	Booktitle = {AAAI},
	Title = {Taxonomy Completion via Triplet Matching Network},
	Year = {2021}
}	
```
Contact: Jieyu Zhang (jieyuz2@cs.washington.edu)

## Install Guide

### Install DGL 0.4.0 version with GPU suppert using Conda

From following page: [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html)

```
conda install -c dglteam dgl-cuda10.0
```

### Other packages

```
ipdb tensorboard gensim networkx tqdm more_itertools
```

## Data Preparation

For dataset used in our paper, you can directly download all input files below and skip this section.

For expanding new input taxonomies, you need to read this section and format your datasets accordingly.

[**MAG-CS**](https://drive.google.com/file/d/11bxzqM8qznI-Qx1aAImd_JDBytw-_rjc/view?usp=sharing)

[**MAG-Psy**](https://drive.google.com/file/d/1joeKI9qvtHl8VX9Dfs9uy2G3EUu0PLDs/view?usp=sharing)

[**WordNet-Noun** ](https://drive.google.com/file/d/1S6ijwV7phg6ZlJbUgSZjPuJTcN98bWwe/view?usp=sharing)

[**WordNet-Verb**](https://drive.google.com/file/d/13LqeaaPq6vS8ah-dgkJO2eWGp107eSfT/view?usp=sharing)

### Step 0.a (Required): Organize your input taxonomy along with node features into the following 3 files

**1. <TAXONOMY_NAME>.terms**, each line represents one concept in the taxonomy, including its ID and surface name

```
taxon1_id \t taxon1_surface_name
taxon2_id \t taxon2_surface_name
taxon3_id \t taxon3_surface_name
...
```

**2. <TAXONOMY_NAME>.taxo**, each line represents one relation in the taxonomy, including the parent taxon ID and child taxon ID

```
parent_taxon1_id \t child_taxon1_id
parent_taxon2_id \t child_taxon2_id
parent_taxon3_id \t child_taxon3_id
...
```

**3. <TAXONOMY_NAME>.terms.<EMBED_SUFFIX>.embed**, the first line indicates the vocabulary size and embedding dimension, each of the following line represents one taxon with its pretrained embedding

```
<VOCAB_SIZE> <EMBED_DIM>
taxon1_id taxon1_embedding
taxon2_id taxon2_embedding
taxon3_id taxon3_embedding
...
```

The embedding file follows the gensim word2vec format.

Notes:

1. Make sure the <TAXONOMY_NAME> is the same across all the 3 files.
2. The <EMBED_SUFFIX> is used to chooose what initial embedding you will use. You can leave it empty to load the file "<TAXONOMY_NAME>.terms.embed". **Make sure you can generate the embedding for a new given term.**

### Step 0.b (Optional): Generate train/validation/test partition files

You can generate your desired train/validation/test parition files by creating another 3 separated files (named <TAXONOMY_NAME>.terms.train, <TAXONOMY_NAME>.terms.validation, as well as <TAXONOMY_NAME>.terms.test) and puting them in the same directory as the above three required files.

These three partition files are of the same format -- each line includes one taxon_id that appears in the above <TAXONOMY_NAME>.terms file.

### Step 1: Generate the binary dataset file

1. create a folder "./data/{DATASET_NAME}"
2. put the above three required files (as well as three optional partition files) in "./data/{DATASET_NAME}"
3. under this root directory, run

```
python generate_dataset_binary.py \
    --taxon_name <TAXONOMY_NAME> \
    --data_dir <DATASET_NAME> \
    --embed_suffix <EMBED_SUFFIX> \
    --existing_partition 0 \
    --partition_pattern internal \
```

This script will first load the existing taxonomy (along with initial node features indicated by `embed_suffix`) from the previous three required files.
Then, if `existing_partition` is 0, it will generate a random train/validation/test partitions, otherwise, it will load the existing train/validation/test partition files.
Notice that if `partition_pattern` is `internal`, it will randomly sample both internal and leaf nodes for validation/test, which makes it a taxonomy completion task; if it is set `leaf`, it will become a taxonomy expansion task.
Finally, it saves the generated dataset (along with all initial node features) in one pickle file for fast loading next time.

## Model Training

### Simplest training

Write all the parameters in an config file, let's say **./config_files/config.universal.json**, and then start training.

Please check **./config_files/config.explain.json** for explanation of all parameters in config file

There are four config files under each sub dirs of **./config_files**:

1. **baselineex**: baselines for taxonomy expansion;
2. **tmnex**: TMN for taxonomy expansion;
3. **baseline**: baselines for taxonomy completion;
4. **tmn**: TMN for taxonomy completion;

```
python train.py --config config_files/config.universal.json
```

### Specifying parameters in training command

For example, you can indicate the matching method as follow:

```
python train.py --config config_files/config.universal.json --mm BIM --device 0
```

Please check **./train.py** for all configurable parameters.


### Running one-to-one matching baselines

For example, BIM method on MAG-PSY:

```
python train.py --config config_files/MAG-PSY/config.test.baseline.json --mm BIM
```

### Running Triplet Matching Network

For example, on MAG-PSY:

```
python train.py --config config_files/MAG-PSY/config.test.tmn.json --mm TMN
```

### Supporting multiple feature encoders

Although we only use initial embedding as input in our paper, our code supports combinations of complicated encoders such as both GNN and LSTM.

Check out the `mode` parameter, there are three symbols for `mode`: `r`, `p` and `g`, representing initial embedding, LSTM and GNN respectively. 

If you want to replace initial embedding with a GNN encoder, plz set `mode` to `g`; 

If you want to use a combination of initial embedding and GNN encoder, plz set `mode` to `rg`, and then the initial embedding and embedding output by GNN encoder will be concatenated for calculating matching score; 

For GNN encoder, we defer user to Jiaming's WWW'20 paper [TaxoExpan](https://arxiv.org/abs/2001.09522);

For LSTM encoder, we collect a path from root to the anchor node, and them use LSTM to encoder it to generate representation of anchor node;


## Model Inference

Predict on completely new taxons.

### Data Format

We assume the input taxon list is of the following format:

```
term1 \t embeddings of term1
term2 \t embeddings of term2
term3 \t embeddings of term3
...
```

The term can be either a unigram or a phrase, as long as it doesn't contain "\t" character.
The embedding is space-sperated and of the same dimension of trained model.

### Infer

```
python infer.py --resume <MODEL_CHECKPOINT.pth> --taxon <INPUT_TAXON_LIST.txt> --save <OUTPUT_RESULT.tsv> --device 0
```


### Model Organization

For all implementations, we follow the project organization in [pytorch-template](https://github.com/victoresque/pytorch-template).
