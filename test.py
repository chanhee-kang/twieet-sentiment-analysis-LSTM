import torch
import pickle
import spacy
from model import Sentiment

nlp = spacy.load('en')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_class(model, TEXT,sentence, min_len = 4):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    print(preds)
    max_preds = preds.argmax(dim = 0)

    return max_preds.item()



EMBEDDING_DIM = 400
HIDDEN_DIM = 400
EPOCH = 20
OUTPUT_DIM = 3

TEXT = pickle.load(open("text.pkl", "rb"))
LABEL = pickle.load(open("label.pkl", "rb"))
ix_to_label = {0:'negative', 1:'neutral',  2:'positive'}


model = Sentiment(len(TEXT.vocab), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 2, 0.5)
model.to(device)
model.load_state_dict(torch.load('best_model.pt'))

pred_class = predict_class(model,TEXT, "bitch")



print(f'Predicted class is:  {ix_to_label[pred_class]}')
