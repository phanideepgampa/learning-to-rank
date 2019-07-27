# Data

The training instance for the model is the question paired with all the candidate answers. For WikiQa, the data from [here](https://github.com/castorini/NCE-CNN-Torch/tree/master/data/WikiQA) was preprocessed using the file *data_preprocessing.py*. For InsuranceQA, the data was first converted into the format as the WikiQA and the same procedure is repeated.

You can skip the above process by directly downloading the following files. Download and place the preprocessed data pickle files along with the vocabulary in the respective folders. The link for downloading 

- [wiki_qa](https://drive.google.com/file/d/1ycnrSk5lYMS6ozSu2WFeXqCDs-7Mz3fo/view?usp=sharing)
- [insurance_qa](https://drive.google.com/file/d/1qDD0spHNTa0pKOJ_dbGydwECfo-0Em9f/view?usp=sharing)

I have not provided the data files preprocessed with features from pre-trained *BERT* model because of huge size of the pickle files (__10GB+__). The file *bert_preprocessing.py* contains the code for pickling the data after the features from BERT are extracted.

__Uncomment line no 62-66  and comment 68-73 lines in evaluate.py__ for training or evaluating model without BERT features. Similarly, __comment line no 62-66  and uncomment 68-73 lines in evaluate.py__ for training or evaluating model with BERT features

# Train

*main.py* consists the code for training the model with text matching architecture (MCAN). *main2.py* consists the code for training the model on features obtained from a pre-trained _BERT_ language model.

## For training the default model 

The below command trains the model on hybrid loss with *gamma=0.75*. __Run this command after you have placed the preprocessed data files in the respective folders__. Change the data directory and the vocabulary file for training for training on insurance_qa dataset.

```
python main.py --highway --lr 8e-5  --lr_2 1e-4   --batch_size 20 --train_batch 3 --gamma 0.75  --beta 0.9  --epochs_ext 25 --device 1  --vocab_file wiki_qa/vocab.42b.300d.p  --data_dir wiki_qa/  --lr_sch 2  --max_num_of_ans 3  --compression_type SM 
```

## For training the model using features from BERT

__This command requires the features of the text data extracted from BERT language model__. Change the input dimensions accordingly. For example, if you take the features only from the last layer from *BERT-large*, the input dimensions would be 1024.

```
python main2.py --highway --lr 3e-4  --lr_2 1e-3   --batch_size 20 --train_batch 1 --gamma 1.0  --beta 0.   --epochs_ext 25 --device 1  --vocab_file wiki_qa/pickle_data/vocab.42b.300d.p  --data_dir wiki_qa/pickle_data  --lr_sch 2   --dropout 0.3 --input_dim 1024
```

# Evaluate

Provide the path to the trained model using the command --model_file. Change the data directory and vocabulary path accordingly.
```
python evaluate.py --vocab_file wiki_qa/pickle_data/vocab.42b.300d.p  --data_dir wiki_qa/   --model_file best_models/
```
# Results from the paper
A link for downloading the pre trained model with best scores for reproducibility will be provided soon.




