## Dogs vs Cats

Competition [link](https://www.kaggle.com/c/dogs-vs-cats/)

### Architecture Used:

#### CNN Softmax classifier

Layer (type)                 Output Shape              Param #   

input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 222, 222, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 111, 111, 32)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 109, 109, 32)      9248      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 54, 54, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 52, 52, 32)        9248      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 26, 26, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 21632)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 64)                1384512   
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
predictions (Dense)          (None, 2)                 130       

Total params: 1,404,034
Trainable params: 1,404,034
Non-trainable params: 0
_________________________________________________________________
    
### Code

1. [preprocess.ipynb](preprocess.ipynb) splits `train` data into `train`(75%) and `dev`(25%), resizes images to 224 X 224, saves image data in `.npy` files.
2. [keras-v1.ipynb](keras-v1.ipynb) implements above classifier using keras.

After 10 epochs => loss: 0.0920 - acc: 0.9606 - val_loss: 0.8658 - val_acc: 0.7868
    
