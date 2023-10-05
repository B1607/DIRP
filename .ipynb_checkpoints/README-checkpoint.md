# DIRP

# 1
Into the "data" folder and use FASTA file to generate other data feature to "dataset" folder
example:
        python get_Binary_Matrix.py -in "Your FASTA file folder" -out "The destination folder of your output"
        python get_mmseqs2.py -in "Your FASTA file folder" -out "The destination folder of your output"
        *Please change the path of your proteins sequence database
        python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"

# 2
Into the "dataset" folder and use the data feature to generate dataset
example:
        python batch_get_series_feature.py -in "Your data feature Folder" -out "The destination folder of your output" -script get_series_feature.py -num 10 -old_ext "The data format of your data feature" -new_ext ".set" -w "num_dependent"
        *python batch_get_series_feature.py -in Test -out Series11\ProtTrans\Test -script get_series_feature.py -num 10 -old_ext ".porttrans" -new_ext ".set" -w 5

# 3
Into the "code" folder and do the prediction

python main.py -d "ProtTrans" -n_dep 5 -n_fil 256 -n_hid 1000 -bs 1024 -ws 2 4 6 8 10 -n_feat 1024 -e 20 -val "independent"
or
main.ipynb
*Please change the dataset path in "loading_function.py" first!


