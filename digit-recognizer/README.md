## Digit Recognizer on MNIST Dataset

### Architecture Used:

Input Layer: 784 units, with Relu,
1st Hidden Layer: 128 units, with Relu,
2nd Hidden Layer: 64 units, with Relu,
Output Layer: 10 units, with softmax.

Softmax Cross Entropy for computing cost.
Gradient Descent and Adam Optimzation as Optimization algorithms.

### Code

1. nn.ipynb implements Gradient Descent, from scratch.
    max score achieved on kaggle: 0.97042
    
2. tf.ipynb uses tensorflow, implements Adam Optimization.
    max score achieved on kaggle: 0.96814
    
No CNNs used so far.
