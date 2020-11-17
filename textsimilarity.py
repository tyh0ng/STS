from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers import SentenceTransformer, util
import csv

model = SentenceTransformer("xlm-r-distilroberta-base-paraphrase-v1")
# model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
# model = SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")
# model = SentenceTransformer("distilbert-multilingual-nli-stsb-quora-ranking")
# model = SentenceTransformer("LaBSE")
tsv_file = open('eval.tsv')
read_tsv = csv.reader(tsv_file, delimiter="\t")
src_sentences = []
trg_sentences = []

for row in read_tsv:
  src_sentences.append(row[0])
  trg_sentences.append(row[1])
batch_size = 4
embeddings1 = model.encode(src_sentences, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True)
embeddings2 = model.encode(trg_sentences, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True)
cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
print('cosine_scores', cosine_scores)

# name of csv file  
filename = "final_spanish_results1.csv"
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    csvwriter.writerows(map(lambda x: [x], cosine_scores)) 
