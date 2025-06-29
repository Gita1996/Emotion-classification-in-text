# -*- coding: utf-8 -*-
"""attention_visualization_summary2_isear.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zrca9NpMrxpejla7wnAZpLp-q2BYN4iM
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import softmax
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


model_name='microsoft/deberta-v2-xlarge-mnli'
model = AutoModel.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def analyze_token_alignment(premise, emotion):
    # Load model and tokenizer

    # Scenario 1: Without template
    inputs_without_template = tokenizer.encode_plus(premise, f"This text expresses {emotion}", return_tensors='pt')
    outputs_without_template = model(**inputs_without_template)

    input_ids = inputs_without_template['input_ids']

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Extract attention and hidden states
    last_layer_attention_without_template = outputs_without_template.attentions[-1]
#    print(last_layer_attention_without_template.shape)

    last_layer_hidden_state_without_template = outputs_without_template.last_hidden_state
    tokens_without_template = tokenizer.convert_ids_to_tokens(inputs_without_template['input_ids'][0].tolist())

    # Get target token index (last token before [SEP])
    target_token_index = len(tokens_without_template) - 2
#    print(tokens_without_template)
#    target_token_index = tokens.index(emotion)

#    print(tokens[target_token_index])

    # Compute attention weights
#    print("last_layer_attention_without_template")
#   print(last_layer_attention_without_template)
    attention_scores = last_layer_attention_without_template[0, :, :, target_token_index]
#    print("attention_scores")
#    print(attention_scores)
#    for head in range(attention_scores.shape[0]):
#      head_attention = attention_scores[head].detach().numpy()
#      print("head_attention:", head)
#      print(head_attention)
    # visualize or analyze individual head_attention

#    print(attention_scores.shape)
#    print(attention_scores.detach())
#    attention_weights = np.mean(attention_scores.detach().numpy())
#    print(attention_scores.detach().numpy())
#    attention_weights = softmax(torch.tensor(attention_scores), dim=-1).detach().numpy()
#    attention_weights = np.mean(attention_weights.detach().numpy(), axis=0)
    attention_weights = np.mean(attention_scores.detach().numpy(), axis=0)

    return attention_weights[-2]
#    print(attention_weights.shape)
#    attention_weights = softmax(torch.tensor(attention_weights), dim=0).numpy()

#    print(attention_weights[target_token_index])

#sentence = "Close friends talking badly of other friends"
sentence= "I heard part of a conversation in which one talked very low about women."
#sentence="When I had an argument with my best friend and I thought that I was right and she was not."
#sentence="I had a discussion with my mother concerning my sister's divorce, we disagreed strongly."

#sentence = "I hate it when people think they can run other people's lives"
#sentence="In school I was very bad in running long distances and my class-mates laughed at me for this reason."
#similarities = analyze_token_alignment('microsoft/deberta-v2-xlarge-mnli', sentence)
#print(similarities)

#sentence ="At a gathering I found myself involuntarily sitting next to two people who expressed opinions that I considered very low and discriminating."
#prompt="This text expresses disgust"

#emotion='disgust'
#a=analyze_token_alignment('microsoft/deberta-v2-xlarge-mnli', sentence, emotion)
#print(a)

data_file='ISEAR-nli-prompt-expr_emo3-Deberta.csv'
data = pd.read_csv(data_file)
dt=data['text']
emotions=['anger', 'disgust', 'fear' , 'guilt', 'joy', 'sadness', 'shame']

count=0
anger=list()
disgust=list()
fear=list()
guilt=list()
joy=list()
sadness=list()
shame=list()
Max_at_score=list()
at_label=list()
for m in dt:
  l=list()
  for e in emotions:
    a=analyze_token_alignment(m, e)
    l.append(a)
  anger.append(l[0])
  disgust.append(l[1])
  fear.append(l[2])
  guilt.append(l[3])
  joy.append(l[4])
  sadness.append(l[5])
  shame.append(l[6])
  Max_at_score.append(max(l))
  v=np.argmax(l)
  at_label.append(emotions[v])

dict={'labels':data['labels'] , 'text':dt, 'expr_emo':data['expr_emo'],'anger':anger, 'disgust':disgust, 'fear':fear, 'guilt':guilt, 'joy':joy, 'sadness':sadness, 'shame':shame, 'Max_avg_attention':Max_at_score, 'high_attention':at_label}

df=pd.DataFrame(dict)

df.to_csv('attention_visualization_summary2_isear.csv')