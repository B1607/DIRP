# Predicting DNA Interacting Residues in DNA-Binding Proteins from Amino Acid Sequences using Pre-trained Language Models and Multiple Window Scanning Convolutional Neural Networks


## Abstract <a name="abstract"></a>
Background: Protein-DNA binding is essential for key cellular processes. Identifying DNA interacting residues from sequence remains challenging. Recent advances in pre-trained language models and deep learning provide new opportunities.   

Method: We developed a multi-window convolutional neural network model using pre-trained protein language model embeddings as input features. The model scans across pre-trained 1024-dim contextual embeddings of each residue with parallel 1D convolutional layers having varying window sizes from 2-10 residues.    

Results: On curated benchmark datasets of DNA-binding proteins, our model achieves AUCs of 0.97, significantly outperforming previous sequence-based models as well as CNN and machine learning baselines. Multi-scale analysis of pre-trained embeddings enables effective representation of key protein binding properties directly from sequence.   
 
Conclusion: This work demonstrates the utility of pre-trained language models and multi-window CNN architectures for improved prediction of DNA interacting residues from sequence. Our approach provides a promising new direction for characterization of protein-DNA binding mechanisms and interactions.   
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


