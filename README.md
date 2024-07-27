# Deciphering the Language of Protein-DNA Interactions:<br> A Deep Learning Approach Combining Contextual Embeddings and Multi-Scale Sequence Modeling
|[ 🧬&nbsp;Abstract](#abstract) |[ 🚀&nbsp;Quick Start](#quickstart) | [ 💻&nbsp;Prediction from colab](#colab)| [ 💡&nbsp;MCNN Training](#train)|
|-------------|-----------------------|-----------------------|-----|
## 🧬&nbsp;Abstract <a name="abstract"></a>
Deciphering the mechanisms governing protein-DNA interactions is crucial for understanding key cellular processes and disease pathways. In this work, we present a powerful deep learning approach that significantly advances the computational prediction of DNA-interacting residues from protein sequences.

Our method leverages the rich contextual representations learned by pre-trained protein language models, such as ProtTrans, to capture intrinsic biochemical properties and sequence motifs indicative of DNA binding sites. We then integrate these contextual embeddings with a multi-window convolutional neural network architecture, which scans across the sequence at varying window sizes to effectively identify both local and global binding patterns.

Comprehensive evaluation on curated benchmark datasets demonstrates the remarkable performance of our approach, achieving an area under the ROC-AUC of 0.89 and a PR-AUC of 0.57- substantial improvements over previous state-of-the-art sequence-based predictors. This showcases the immense potential of pairing advanced representation learning and deep neural network designs for uncovering the complex syntax governing protein-DNA interactions directly from primary sequences.

Our work not only provides a robust computational tool for characterizing DNA-binding mechanisms, but also highlights the transformative opportunities at the intersection of language modeling, deep learning, and protein sequence analysis. The publicly available code and data further facilitate broader adoption and continued development of these techniques for accelerating mechanistic insights into vital biological processes and disease pathways.

   
<br>

![workflow](https://github.com/B1607/DIRP/blob/226a6de582f96e115c0fff30b3fd2fe4dce60ca7/other/Figure.jpg)
## Dataset <a name="Dataset"></a>

| Dataset        | Protein Sequence | DNA Interacting Residues | Non-Interacting Residues |
|----------------|------------------|--------------------------|--------------------------|
| Training data  | 646              | 15636                    | 298503                   |
| Testing data   | 46               | 965                      | 9911                     |
| Total          | 692              | 16601                    | 308414                   |

## 


##  🚀&nbsp;Quick start <a name="quickstart"></a>

### Step 1: Environment Setup

We recommend using Anaconda to manage the project environment. You can create the necessary environment using the following command:
```bash
conda env create -n MCNN_DNA_Pred -f environment.yml
```
Activate the conda environment
```bash
conda activate MCNN_DNA_Pred
```
### Step 2: Download the GitHub Repository

Clone the GitHub repository using the following command:
```bash
git clone https://github.com/B1607/DIRP.git
```
Navigate to the repository folder:
```bash
cd DIRP
```
### Step 3: Start the Prediction !

Open the prediction program and enter your protein sequence:
```bash
DNA_Prediction.ipynb
```

## 💻&nbsp;Prediction from colab <a name="colab"></a>
We also provide a colab notebook for the DNA Interacting Residue Prediction from protein sequence.
[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/drive/1vNAAfziLS5XYl4zm-uEZD1L28pr_rNbU?usp=sharing)

## 💡&nbsp;MCNN Training <a name="train"></a>

### Step 1: Environment Setup

We recommend using Anaconda to manage the project environment. You can create the necessary environment using the following command:
```bash
conda env create -n MCNN_DNA_Pred -f environment.yml
```
Activate the conda environment
```bash
conda activate MCNN_DNA_Pred
```
### Step 2: Download the GitHub Repository and Dataset

Clone the GitHub repository using the following command:
```bash
git clone https://github.com/B1607/DIRP.git
```
Navigate to the repository folder:
```bash
cd DIRP
```
Download and extract the dataset:
```bash
wget -O ./dataset/ProtTrans.zip http://140.138.155.214/~user4/DIRP/ProtTrans.zip
```
```bash
unzip ./dataset/ProtTrans.zip
```
### Step 3: Navigate to the "code" folder
```bash
cd DIRP
```
### Step 4: Run the Training code
```bash
python MCNN_npy.py

"""
you can also change the arguments to training model by your self
-n_fil , --num_filter
      The number of filters in the convolutional layer.
-n_hid , --num_hidden
      The number of hidden units in the dense layer.
-bs , --batch_size
      The batch size for training the model.
-ws , --window_sizes
      The window sizes for convolutional filters.
-vm , --validation_mode
      The validation mode. Options are 'cross', 'independent'.
-d , --data_type,
      The type of data. Options are 'ProtTrans', 'tape', 'esm2'
-n_feat , --num_feature
      The number of data feature dimensions. 1024 for ProtTrans, 768 for tape, 1280 for esm2.
-length , --n_length
```
      


## Requirements <a name="requirement"></a>
```bash
h5py==3.11.0
tqdm==4.66.4
numpy==1.26.4
scikit-learn==1.4.2
tensorflow==2.10.1
transformers==4.40.1
torch==2.3.0+cu118
fair-esm==2.0.0
```
