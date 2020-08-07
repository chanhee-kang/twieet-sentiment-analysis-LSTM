# Predcting Sentiment with LSTM
<h3>The model will predict your sentence whether it is postive, negative, or netural</h3>

## Datasets
You can download from: [https://www.kaggle.com/c/tweet-sentiment-extraction] <br>
The file is for twieet sentiment extraction but I used for predicting sentiment

## Deep Learning Frame Work
![pytorch](https://user-images.githubusercontent.com/26376653/84051109-9de92d80-a9e9-11ea-887d-06113adab7c0.jpg) <br>
If you do not installed pytorch, downlaod from [https://pytorch.org/get-started/locally/]

## set up
1. Git clone
```
$git clone https://github.com/chanhee-kang/twieet-sentiment-analysis-LSTM.git
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
1. USE <code>train.py</code> for training

2. USE <code>test.py</code> for testing
```
pred_class = predict_class(model,TEXT, "TYPE YOUT SENTENCE")
print(f'Predicted class is:  {ix_to_label[pred_class]}')
```
## Result
1. Training <br>
![epoch](https://user-images.githubusercontent.com/26376653/84519635-56c2ab80-ad0d-11ea-90e4-de21c8a9f4c7.PNG)

2. Sentiment Analysis <br>
LET'S type "I really really love you baby. my sweet heart" <br>
The model predicts the sentense as postive :) <br><br>
![test](https://user-images.githubusercontent.com/26376653/84519461-182cf100-ad0d-11ea-8d7e-8d15338b6c0a.PNG)

## Limitation
The Datasets is too low :sob: <br>
More datasets will increase the performance

### Contact
If you have any requests, please contact: [https://ck992.github.io/](https://ck992.github.io/).

