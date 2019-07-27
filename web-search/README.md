# Data
The original dataset of MQ2007 can be downloaded from [here](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0). The preprocessed data files (after removing queries with no relevant documents) can be downloaded from [here](https://drive.google.com/file/d/13IPgtDq7YNiBoFGV_LXuxAPKIQLyAu_Y/view?usp=sharing). Place the downloaded files inside the directory *MQ2007*.

# Train

The command for training the model on the mq2007 dataset using default parameters
```
python main.py 
```

# Evaluate 

For evaluating the model trained on the mq2007 dataset, use the below command
```
python evaluate.py   --data_dir MQ2007/Fold   --model_file   best_models/
```

# Results from the paper  
A link for downloading the pre trained model with best scores for reproducibility will be provided soon.
