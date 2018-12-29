import data
import torch
from torch.autograd import Variable

model_checkpoint = './model.pt'
with open(model_checkpoint, mode='rb') as file:
    model = torch.load(file)

corpus = data.Corpus('./data/wikitext')
ntokens = len(corpus.dictionary)
print(model)


