### Get the repository [SAITS repository](https://github.com/WenjieDu/SAITS) (Du, W., Côté, D., & Liu, Y. (2022). SAITS: Self-Attention-based Imputation for Time Series. arXiv preprint arXiv:2202.08516.)

### Locate into the datasets generation directory
```
cd dataset_generating_scripts
```

### Download only the electricity dataset (as their bash files process many datasets at the time)
```
cd .. && mkdir Electricity && cd Electricity
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
unzip LD2011_2014.txt.zip
```

### Pre-process only the electricity dataset. 
```
python gene_UCI_electricity_dataset.py \
  --file_path RawData/Electricity/LD2011_2014.txt \
  --artificial_missing_rate 0.1 \
  --seq_len 100 \
  --dataset_name Electricity_seqlen100_01masked \
  --saving_path ../generated_datasets
```


### Note: we provide this final version of the dataset in our get_data.py file, however, if you do the steps above, remember to feature sample the data.
```
dataset = np.split(dataset, 10, 2) # shape = 10, x, 100, 37
```

