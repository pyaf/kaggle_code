## Digit Recognizer on MNIST Dataset

### Code

1. nn.ipynb implements ANN softmax classifier from scratch.
    * max score (accuracy) achieved on kaggle: 0.97042
    
2. tf-ann.ipynb uses tensorflow to implement MLP softmax classifier. 
    * max score (accuracy) achieved on kaggle: 0.96814

3. tf-cnn.ipynb uses tensorflow, implements CNN with 2 Conv , 2 maxpool, and 2 FC layers.
    * max score (accuracy) achieved on kaggle: 0.99028

4. keras-v1.ipynb uses keras, implements CNN with 2 Conv, 2 maxpool, 2 FC layers,
    * max score (accuracy) achieved on kaggle: 0.99185
    
5. keras-v2.ipynb uses data augmentation, batch normalization, dropout in keras.
    * max score (accuracy) achieved on kaggle: 0.99742
    
Checkout [digit-recognizer](https://www.kaggle.com/c/digit-recognizer) at Kaggle for data and to know more.
