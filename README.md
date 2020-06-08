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

4. The rest of the add-on modules, if you don't want to run it, download 
```
pip install "Library Name".
```
(Search on Google if it doesn't download)

## File explaination
1. model.py (network structure)
2. data_loader.py (load data set)
3. test.py (load network weight and load vocabulary and inference)
4. label.pkl (stored label)
5. best_model.pt (model weight)
6. text.pkl (stored text)
