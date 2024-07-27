# Deciphering the Language of Protein-DNA Interactions:<br> A Deep Learning Approach Combining Contextual Embeddings and Multi-Scale Sequence Modeling
|[ 🧬&nbsp;Overview](#Overview) |[📃&nbsp;Dataset](#Dataset) |[ 🚀&nbsp;Quick Start](#quickstart) | [ 💻&nbsp;Prediction With Colab](#colab)| [ 💡&nbsp;MCNN Training](#train)|[ 💾&nbsp;Requirements](#Requirements)|[ 📚&nbsp;License](#License)|
|-------------------------------|-----------------------------|------------------------------------|------------------------------------------|---------------------------------|---------------------|---------------|
## 🧬&nbsp;Overview <a name="Overview"></a>
This project implements a deep learning model to predict DNA-interacting residues from amino acid sequences. Utilizing a pre-trained ProtTrans model, it generates embeddings for protein sequences and predicts which residues are likely to interact with DNA.

The primary purpose of this tool is to assist researchers and bioinformaticians in identifying potential DNA binding sites in protein sequences. This is valuable for understanding protein-DNA interactions, which are crucial in various biological processes such as gene regulation, DNA repair, and replication.
   
<br>

![workflow](https://github.com/B1607/DIRP/blob/226a6de582f96e115c0fff30b3fd2fe4dce60ca7/other/Figure.jpg)
## 📃&nbsp;Dataset <a name="Dataset"></a>

| Dataset        | Protein Sequence | DNA Interacting Residues | Non-Interacting Residues |
|----------------|------------------|--------------------------|--------------------------|
| Training data  | 646              | 15636                    | 298503                   |
| Testing data   | 46               | 965                      | 9911                     |
| Total          | 692              | 16601                    | 308414                   |




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
### Step 3: Open the Notebook and Start the Prediction !

Open the prediction program
```bash
DNA_Prediction.ipynb
```
Follow the instructions to execute the program cells and input your protein sequence.
<br>For example: 
```bash
MHHHHHHSSGRENLYFQGSNKKRKRCGVCVPCLRKEPCGACYNCVNRSTSHQICKMRKCEQLKKKRVVPMKG.
```
After submitting the sequence, the script will first convert it into ProtTrans embeddings and then make predictions for each residue.
```
## Example Output
Amino acid:  MHHHHHHSSGRENLYFQGSNKKRKRCGVCVPCLRKEPCGACYNCVNRSTSHQICKMRKCEQLKKKRVVPMKG
Prediction:  000000000000000000011111100000011100000000000001111100001111000000000000

1 indicates the amino acid is predicted to be a DNA interacting residue.
0 indicates the amino acid is predicted to be a non-DNA interacting residue.
```


## 💻&nbsp;Prediction With Colab <a name="colab"></a>
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
      The length of the input sequence in residues (amino acids).
```
      


## 💾&nbsp;Requirements <a name="requirement"></a>
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

## 📚&nbsp;License <a name=" License"></a>
Licensed under the Academic Free License version 3.0
