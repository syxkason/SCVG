# SCVG
Tensorflow implementation for our CIKM 2021 paper:

>Semi-deterministic and Contrastive Variational Graph Autoencoder for Recommendation  .

## Usage

The required packages can be found in *requirement.txt*:

- Python == 3.6.8
- Tensorflow == 1.8.0
- numpy == 1.15.3
- scipy == 1.1.0
- pandas == 0.25.2
- cython == 0.29.15

## Preparation
**Firstly**, compline the evaluator of cpp implementation with the following command line:

```bash
python setup.py build_ext --inplace
```

If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

Note that the cpp implementation is much faster than python.

Further details, please refer to [NeuRec](https://github.com/wubinzzu/NeuRec/)

**Secondly**, specify dataset and recommender in configuration file *NeuRec.properties*.

**Thirdly**, Model specific hyperparameters are in configuration file *./conf/SCVG.properties*.

Note that we need 40GB+ of memory space to load the whole graph.

## Dataset Parameters

### Yelp2018, Amazon-Book

```
lr=0.001
embedding_size=200
epochs=1000
adj_type=pre
keep_prob=0.5
dropout=True
n_hidden=600
```

## Run

```bash
python main.py
```

