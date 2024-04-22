# Deciphering the Language of Protein-DNA Interactions: A Deep Learning Approach Combining Contextual Embeddings and Multi-Scale Sequence Modeling


## Abstract <a name="abstract"></a>
Deciphering the mechanisms governing protein-DNA interactions is crucial for understanding key cellular processes and disease pathways. In this work, we present a powerful deep learning approach that significantly advances the computational prediction of DNA-interacting residues from protein sequences.

Our method leverages the rich contextual representations learned by pre-trained protein language models, such as ProtTrans, to capture intrinsic biochemical properties and sequence motifs indicative of DNA binding sites. We then integrate these contextual embeddings with a multi-window convolutional neural network architecture, which scans across the sequence at varying window sizes to effectively identify both local and global binding patterns.

Comprehensive evaluation on curated benchmark datasets demonstrates the remarkable performance of our approach, achieving an area under the ROC curve (AUC) of 0.97 - a substantial improvement over previous state-of-the-art sequence-based predictors. This showcases the immense potential of pairing advanced representation learning and deep neural network designs for uncovering the complex syntax governing protein-DNA interactions directly from primary sequences.

Our work not only provides a robust computational tool for characterizing DNA-binding mechanisms, but also highlights the transformative opportunities at the intersection of language modeling, deep learning, and protein sequence analysis. The publicly available code and data further facilitate broader adoption and continued development of these techniques for accelerating mechanistic insights into vital biological processes and disease pathways.
   
<br>
![workflow](https://github.com/B1607/DIRP/blob/226a6de582f96e115c0fff30b3fd2fe4dce60ca7/other/Figure.jpg)
## Dataset <a name="Dataset"></a>

| Dataset        | Protein Sequence | DNA Interacting Residues | Non-Interacting Residues |
|----------------|------------------|--------------------------|--------------------------|
| Training data  | 646              | 15636                    | 298503                   |
| Testing data   | 46               | 965                      | 9911                     |
| Total          | 692              | 16601                    | 308414                   |


## Quick start <a name="quickstart"></a>

### Step 1: Generate Data Features

Navigate to the data folder and utilize the FASTA file to produce additional data features, saving them in the dataset folder.

Example usage:
```bash
python get_Binary_Matrix.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_mmseqs2.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"
```
"Note: Ensure to update the path to your protein sequence database within get_mmseqs2.py as necessary."
### Step 2: Generate Dataset Using Data Features

Transition to the dataset folder and utilize the data features to produce a dataset.

Example usage:
```bash
python batch_get_series_feature.py -in "Your data feature Folder" -out "The destination folder of your output" -script get_series_feature.py -num 10 -old_ext "The data format of your data feature" -new_ext ".set" -w "num_dependent"
```
Alternative example:
```bash
python batch_get_series_feature.py -in Test -out Series11\ProtTrans\Test -script get_series_feature.py -num 10 -old_ext ".porttrans" -new_ext ".set" -w 5
```

### Step 3: Execute Prediction

Navigate to the code folder to execute the prediction.

Command-line usage:
```bash
python main.py -d "ProtTrans" -n_dep 5 -n_fil 256 -n_hid 1000 -bs 1024 -ws 2 4 6 8 10 -n_feat 1024 -e 20 -val "independent"
```
Alternatively, utilize the Jupyter notebook:
```bash
main.ipynb
```


