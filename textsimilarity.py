from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers import SentenceTransformer, util
import csv
import torch

model = SentenceTransformer("xlm-r-distilroberta-base-paraphrase-v1")
if torch.cuda.is_available():
  model.cuda()
  print('Using GPU')
# model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
# model = SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")
# model = SentenceTransformer("distilbert-multilingual-nli-stsb-quora-ranking")
# model = SentenceTransformer("LaBSE")
tsv_file = open('train-hotels-es.csv')
read_tsv = csv.reader(tsv_file, delimiter=",")
src_sentences = []
trg_sentences = []

for row in read_tsv:
  src_sentences.append(row[1])
  trg_sentences.append(row[2])
batch_size = 500
embeddings1 = model.encode(src_sentences, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True)
embeddings2 = model.encode(trg_sentences, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True)
cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
print('cosine_scores', cosine_scores)

# name of csv file  
filename = "train-hotels-es-sts-scores.csv"
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    csvwriter.writerows(map(lambda x: [x], cosine_scores)) 
