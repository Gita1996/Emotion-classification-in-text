#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, RobertaForSequenceClassification, RobertaTokenizerFast
from sklearn.model_selection import train_test_split

from datasets import Dataset

# Load your dataset
#data = pd.read_csv("TEC2.csv")
#class_names=sorted(list(set(data['labels'])))

# Split the dataset into training and validation sets
#data = pd.read_csv("TEC3.csv")

train_data=pd.read_csv("ISEAR_dataset_train.csv")
valid_data=pd.read_csv("ISEAR_dataset_val.csv")



# Load the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli", ignore_mismatched_sizes=True)
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large-mnli", ignore_mismatched_sizes=True)

# Tokenize the dataset
train_encodings = tokenizer(list(train_data["text"]), truncation=True, padding=True)
valid_encodings = tokenizer(list(valid_data["text"]), truncation=True, padding=True)

train_encodings['labels'] = list(train_data["labels"])
valid_encodings['labels'] = list(valid_data["labels"])

# Create a Dataset object
train_dataset = Dataset.from_dict(train_encodings)
valid_dataset = Dataset.from_dict(valid_encodings)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
valid_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])



#train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], 
#                                   "attention_mask": train_encodings["attention_mask"],
#                                   "labels": train_data["labels"]})

#valid_dataset = Dataset.from_dict({"input_ids": valid_encodings["input_ids"], 
#                                   "attention_mask": valid_encodings["attention_mask"],
#                                   "labels": valid_data["labels"]})

# Load the pre-trained model for sequence classification

#id2label = {i: label for i, label in enumerate(class_names)}
#id2label={0:'anger',1:'disgust',2:'fear',4:'joy',7:'surprise'}
id2label={0:'anger', 1:'disgust', 2:'fear', 3:'guilt', 4:'joy', 5:'sadness', 6:'shame'}
# Update the model's configuration with the id2label mapping
config = AutoConfig.from_pretrained("roberta-large-mnli")
config.update({"id2label": id2label})
model =  RobertaForSequenceClassification.from_pretrained("roberta-large-mnli", config=config,ignore_mismatched_sizes=True)

# Fine-tuning parameters
training_args = TrainingArguments(
    output_dir="./my_fine_tuned_roberta_model_isear2",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_strategy="steps",
    logging_steps=10,
    logging_dir="./logs",
)

# Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_roberta_model_isear2")

# Save the tokenizer for the fine-tuned model
tokenizer.save_pretrained("./fine_tuned_roberta_model_isear2")

