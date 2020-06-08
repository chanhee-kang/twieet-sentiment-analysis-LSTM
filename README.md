#The file will be upload after 6.20

## Sentiment Analysis
Sentiment analysis with tweet datasets

## set up
1. Git clone
```
$git clone https://github.com/chanhee-kang/Twieet-sentiment-analysis-kaggle.git
```
2. Install packages
```
$python -m spacy download en
```
3. CUDA and CUDNN (for GPU acceleration/OPTIONAL)
* ###### If you do not have GPU, it is perfectly ok with using CPU. CUDA and CUDNN is just option

4. Install the rest of modules. 
```
pip install "library Name".
```

## File explaination
1. model.py (network structure)
2. data_loader.py (load data set)
3. test.py (load network weight and load vocabulary and inference)
4. label.pkl (stored label)
5. best_model.pt (model weight)
6. text.pkl (stored text)

## Get Start

## Limitation

### Contact
If you have any requests, please contact: [https://ck992.github.io/](https://ck992.github.io/).

