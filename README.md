# !!! The file will be upload after 6.20 !!!

# Sentiment Analysis
Sentiment analysis with tweet datasets using LSTM

## Machine Learning Frame Work
![pytorch](https://user-images.githubusercontent.com/26376653/84051109-9de92d80-a9e9-11ea-887d-06113adab7c0.jpg)
If you do not installed pytorch, downlaod from [https://pytorch.org/get-started/locally/]

## set up
1. Git clone
```
$git clone https://github.com/chanhee-kang/Twieet-sentiment-analysis-kaggle.git
```
2. Install packages
```
$python -m spacy download en
```
3. CUDA and CUDNN (for GPU acceleration/OPTIONAL)<br><br>
![nvidia](https://user-images.githubusercontent.com/26376653/84051534-48f9e700-a9ea-11ea-8faf-bd162daec013.png)<br><br>
You need NVIDA GPU for CUDA. Please download from [https://developer.nvidia.com/cuda-toolkit-archive] <br>
If you do not have GPU, it is perfectly ok with using CPU. CUDA and CUDNN is just option)

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

