## Sentiment Analysis
Sentiment analysis with tweet datasets

## set up
1. Download this git for initiating this program.
```
$git clone https://github.com/chanhee-kang/Twieet-sentiment-analysis-kaggle.git
```
2. Install packages
```
$python -m spacy download en
```
3. CUDA and CUDNN (for GPU acceleration, faster if you do)

4, The rest of the add-on modules, if you don't want to run it, download 
```
pip install "Library Name".
```
(Search on Google if it doesn't download)

## File explaination
model.py (network structure)
data_loader.py (load data set)
test.py (load network weight and load vocabulary and inference)
label.pkl (stored label)
best_model.pt (model weight)
text.pkl (stored text)
