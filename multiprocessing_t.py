import multiprocessing
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class models:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            layers.Input(1, ),
            layers.Dense(1, )
        ])

    def train_in_model(self):
        with tf.device('/cpu:0'):
            print("reached here!")
            testdata = np.asarray([[1], [2], [3], [4]], dtype=np.int)
            self.model.predict(testdata)
            print(f'finished')


class MultiTest(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)
        self.model = models()

    def train(self):
        self.model.train_in_model()

    def run(self):
        self.train()


multi = []
for i in range(4):
    multi.append(MultiTest())

for i in range(4):
    multi[i].start()

