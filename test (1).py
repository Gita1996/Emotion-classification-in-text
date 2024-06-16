from transformers import pipeline
import pandas as pd
from sklearn.metrics import classification_report

data=pd.read_csv('ISEAR_dataset_test2.csv')

classifier = pipeline('text-classification', model="fine_tuned_roberta_model_isear2",tokenizer="fine_tuned_roberta_model_isear2")
#roberta-large-mnli
#fine_tuned_roberta_model2

#classifier = pipeline('sentiment-analysis')
predictions=list()
for t in data['text']:
    p=classifier(t)
#    print(p[0]['label'])
    predictions.append(p[0]['label'])
#print(predictions)
print(classification_report(list(data['labels']),predictions))
l={'labels':data['labels'], 'text':data['text'], 'predictions':predictions}

d=pd.DataFrame(l)
d.to_csv('ISEAR_fine_tuned_Roberta2.csv', index=False)


#text = "Every time I imagine that someone I love or I could contact a  serious illness, even death."
#result = classifier(text)
#print(result)

#predicted_label = result[0]["label"]
#print(f"Predicted label: {predicted_label}")