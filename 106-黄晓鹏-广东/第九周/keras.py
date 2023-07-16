# py3.6tf1.14
from keras import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# load data
(tran_imgs, tran_labels), (test_imgs, test_labels) = mnist.load_date()

# normaliza data
tran_imgs = tran_imgs.reshape((60000, 28 * 28))
tran_imgs = tran_imgs.astype('float') / 255
test_imgs = test_imgs.reshape((10000, 28 * 28))
test_imgs = test_imgs.astype('float') / 255

# one-hot code labels
tran_labels = to_categorical(tran_labels)
test_labels = to_categorical(test_labels)

# create network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# fit network
network.fit(tran_imgs, tran_labels, epochs=5, batch_size = 128)

#evaluate network get loss && acc
test_loss, test_acc = network.evaluate(test_imgs, test_labels, verbose=1)

print('test_loss:',test_imgs)
print('test_acc:',test_acc)

#
test_imgs_2 = test_imgs[1]
test_imgs_2 = test_imgs_2.reshape((1, 28*28))
res = network.predict(test_imgs_2)

