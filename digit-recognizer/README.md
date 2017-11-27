## Digit Recognizer on MNIST Dataset

### Architecture Used:

#### MLP Softmax Classifier

- Input Layer: 784 units, with Relu,
- 1st Hidden Layer: 128 units, with Relu,
- 2nd Hidden Layer: 64 units, with Relu,
- Output Layer: 10 units, with softmax.
- Softmax Cross Entropy for computing cost.
- Gradient Descent and Adam Optimzation for optimization.

#### CNN Softmax classifier

- 2 Conv Layers
- 2 MaxPool Layers
- 2 FC Layers
- Softmax Cross entropy for computing cost.
- Adam Optimization for optimization.
    
### Code

1. nn.ipynb implements ANN softmax classifier from scratch.
    * max score (accuracy) achieved on kaggle: 0.97042
    
2. tf-ann.ipynb uses tensorflow to implement MLP softmax classifier. 
    * max score (accuracy) achieved on kaggle: 0.96814

3. tf-cnn.ipynb uses tensorflow, implements CNN with 2 Conv , 2 maxpool, and 2 FC layers.
    * max score (accuracy) achieved on kaggle: 0.99028

4. keras.ipynb uses keras, implements CNN with 2 Conv, 2 maxpool, 2 FC layers,
    * max score (accuracy) achieved on kaggle: 0.99185