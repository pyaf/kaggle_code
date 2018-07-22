import numpy as np
import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

label_data = pd.read_csv('labels.csv')
train_label, dev_label = train_test_split(label_data, test_size=0.30, random_state=0)
unique_labels = pd.unique(label_data['breed'])

def load_data(type_):
    label_data = train_label if type_=='train' else dev_label
    X = np.zeros(([label_data.shape[0], 387, 443, 3]), dtype='float32')
    Y = np.zeros(([120, label_data.shape[0]]), dtype='float32')
    for i, id_ in enumerate(label_data['id']):
        file = 'train/' + id_ + '.jpg'
        X[i] = mpimg.imread(file)
        breed = label_data.loc[label_data['id']==id_]['breed'].as_matrix()[0]
        Y[np.where(unique_labels==breed)[0][0], i] = 1
    return (X, Y)


def load_test_data():
    X = np.zeros(([10357, 387, 443, 3]), dtype='float32')
    for i, filename in enumerate(os.listdir('test')):
        file = os.path.join('test', filename)
        X[i] = mpimg.imread(file)
    return X

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1,)
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.legend(['train', 'val'], loc='upper left')
    plt.title(' accuracy')


    plt.subplot(1, 2, 2)
    
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('loss')
    plt.show()
#if __name__=='main':

    
