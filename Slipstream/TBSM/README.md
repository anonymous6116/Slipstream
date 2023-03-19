# TBSM - Slipstream implementation for Time Base Sequence Model


TBSM consists of an embedding layer and time series layer (TSL).

The embedding layer is implemented through (DLRM). Within this layer
all sparse features pass through embeddings, while dense features
pass through MLP. The MLP maps dense features into the vector space of
the same dimension as the embedding dimension for all sparse features.
In the next step the vector of all pairwise inner products between
embedded sparse and mapped dense features is formed. Finally, this
vector of inner products is concatenated with an output vector from MLP
and passed through the top MLP, which produces a vector z of dimensions n.

Let us denote z_i the history of items, and z_t the last item. The TSL layer
computes one or more context vectors c. It ressembles an attention mechanism,
and contains its own MLP network with trainable parameters. The attention MLP
takes a vector of inner products between normalized z_i and z_t and outputs
the vector of coefficients a which is applied to z_i  to obtain the context
vector c. In this way, c measures the significance of each of the z_i with respect
to vector z_t. For example, if the first component of a is 1 while the rest are 0,
then c = z_0. The distinction with standard attention mechanisms lies in the
normalization (spherical projection) use of well defined inner products and
use of individual rather than shared MLP for multiple context vectors.

The final step takes vectors [z_t, c_j] and passes them through MLPs
resulting in the probability of a click.

```
   model:
               probability of a click
                         |
                        /\
                       /__\
                         |
              _________ op ___________
            /            |            \
  (context) c                 (candidate) z_t
  (vector )                   (embedding)
              \                     /
                 \                 /
                     attention    /
                     /       \   /
                    H           z_t
                  /             |
                 /              |
              DLRM            DLRM
              /                 |
   user history             new item

 model components
 i) Embedding layer (DLRM)
                    item tower output vector z
                               |
                              /\
                             /__\
                               |
       _____________________> Op  <___________________
     /                         |                      \
    /\                        /\                      /\
   /__\                      /__\           ...      /__\
    |                          |                       |
    |                         Op                      Op
    |                    ____/__\_____           ____/__\____
    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
 item input:
 [ dense features ]     [sparse indices] , ..., [sparse indices]

 ii) TSL layer, processing history (a sequence of vectors used in the attention)

          z1,       z2,   ...   zk   -->   H = [z1, z2, ..., zk]
          |         |           |
         DLRM,     DLRM, ...   DLRM
          |         |           |
 previous i1       i2   ...    ik features

 TSL functions similarly to attention mechanism, with three important distinctions
 a. The vectors are normalized (spherical projection)
 b. The product between vectors can be done using: dot v'w, indefinite v' A w and
    positive semi-definite v' (A'A) w inner product
 c. The multiple TSL heads have individual rather than shared output MLPs
```
Dataset Pre-processing
----------------------
```
     cd TBSM

```
The code supports interface with the [Taobao User Behavior Dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1).
- Please do the following to prepare the dataset for use with TBSM code:
    - Download UserBehavior.csv.zip and UserBehavior.csv.zip.md5 into directory
      ```
      mkdir data
      ./data/taobao_data
      ```
    - Check the md5sum hash and unzip
       ```
       md5sum UserBehavior.csv.zip
       unzip UserBehavior.csv.zip
       ```
    - Run preprocessing to create input files (*taobao_train.txt* and *taobao_test.txt*)
       ```
       python ./tools/taobao_prepare.py
       ```
    - Copy input files (*taobao_train.txt* and *taobao_test.txt*) to ./input
       ```
       mkdir input
       mkdir output
       cp ./data/taobao_data/*.txt ./input/.
       ```
    - Run preprocessing to create processed files (*taobao_train_t20.npz* *taobao_val_t20.npz* *train.npz* *val.npz*)
       ```
       ./tbsm_processing.sh
       ```
Running TBSM Baseline
----------------------

TBSM baseline can be run on a hybrid CPU-GPU system using following script
```
     ./run_tbsm_baseline.sh
```

Input Segregation - Hot/Cold
-----------------------------

Input training dataset requires to be segregated into hold and cold inputs and hot and cold embeddings required for Slipstream.  Based on available GPU memory for hot embedding entries, $\Lambda$ parameter is selected that defines if an embedding entry is popular or not -- based on embeddings access frequency and further training dataset is segregated into hot and cold inputs.

Input segregation can be executed on a CPU system using following script
```
     ./run_input_segregation.sh
```

Running FAE-TBSM Baseline
--------------------------

FAE baseline can be run on a hybrid CPU-GPU system using following script
```
     ./run_tbsm_fae.sh
```

Running Slipstream
-------------------

Slipstream identifies the stale embeddings via threshold ($T$) found automatically based on the target drop percentage ($D$) along with certain additional parameters like $\alpha$, which determines the condition under which an input is dropped given the number of accessed embeddings by input are stale.

Slipstream can be run on a hybird CPU-GPU system using following script
```
     ./run_tbsm_slipstream.sh
```

Requirements
-------------

This project requires **Python** $\geq 3.7$, with below dependencies.

pytorch

scikit-learn

numpy

pandas

onnx (*optional*)

pydot (*optional*)

torchviz (*optional*)

tqdm

cPickle


License
-------
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.



