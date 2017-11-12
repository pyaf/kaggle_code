## Digit Recognizer on MNIST Dataset

### Architecture Used:

ANN Softmax Classifier
    Input Layer: 784 units, with Relu,
    1st Hidden Layer: 128 units, with Relu,
    2nd Hidden Layer: 64 units, with Relu,
    Output Layer: 10 units, with softmax.

    Softmax Cross Entropy for computing cost.
    Gradient Descent and Adam Optimzation for optimization.

CNN classifier
    2 Conv Layers
    2 FC Layers
    Softmax Cross entropy for computing cost.
    Adam Optimization for optimization.
    
### Code

1. nn.ipynb implements ANN softmax classifier from scratch.
    max score achieved on kaggle: 0.97042
    
2. tf-ann.ipynb uses tensorflow to implement ANN softmax classifier. 
    max score achieved on kaggle: 0.96814

3. tf-cnn.ipynb uses tensorflow, implemets CNN with 2 Conv and 2 FC layers.
    max score achieved on kaggle: 0.99028
