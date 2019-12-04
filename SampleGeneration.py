import time as t

import numpy as np
import tensorflow as tf

# Load Entire Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255
size_of_subsets = 15000


# Define Classifier
class ClassifierModel(tf.keras.Model):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.c2 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.p1 = tf.keras.layers.MaxPool2D(2, 2)
        self.d1 = tf.keras.layers.Dropout(0.25)

        self.c3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                         activation='relu')
        self.c4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.p2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.d2 = tf.keras.layers.Dropout(0.25)

        self.f1 = tf.keras.layers.Flatten()

        self.l1 = tf.keras.layers.Dense(512, activation='relu')
        self.d3 = tf.keras.layers.Dropout(0.5)
        self.pred = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c3(x)
        x = self.c4(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.f1(x)

        x = self.l1(x)
        x = self.d3(x)
        return self.pred(x)

class Classifier:
    def __init__(self, batch_size, id):
        self.id = id
        self.model = ClassifierModel()
        self.idx = np.random.permutation(len(x_train))
        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train[0:size_of_subsets], y_train[0:size_of_subsets])).shuffle(size_of_subsets).batch(batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(batch_size)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def train_one_step(self, images, labels):
        with tf.GradientTape() as tape:
            pred = self.model(images)
            loss = self.loss_object(labels, pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_acc(labels, pred)

    def test_one_step(self, images, labels, type='test'):
        pred = self.model(images)
        t_loss = self.loss_object(labels, pred)
        if type == 'test':
            self.test_loss(t_loss)
            self.test_acc(labels, pred)
        if type == 'train':
            self.train_loss(t_loss)
            self.train_acc(labels, pred)

    def train_one_epoch(self, epoch_number):
        self.train_loss.reset_states()
        self.train_acc.reset_states()
        self.test_loss.reset_states()
        self.test_acc.reset_states()
        begin = t.time()
        for images, labels in self.train_ds:
            self.train_one_step(images, labels)
        for images, labels in self.test_ds:
            self.test_one_step(images, labels, type='test')
        template = 'ID: {}, Epoch {}, Train Loss: {:.2f}, Train Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}, Run Time: {:.2f}'
        print(template.format(self.id,
                              epoch_number + 1,
                              self.train_loss.result(),
                              self.train_acc.result() * 100,
                              self.test_loss.result(),
                              self.test_acc.result() * 100
                              , t.time() - begin))

    def evaluate(self):
        self.train_loss.reset_states()
        self.train_acc.reset_states()
        self.test_loss.reset_states()
        self.test_acc.reset_states()
        begin = t.time()
        for images, labels in self.train_ds:
            self.test_one_step(images, labels, type='train')
        for images, labels in self.test_ds:
            self.test_one_step(images, labels, type='test')
        template = 'ID: {}, Train Loss: {:.2f}, Train Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}, Run Time: {:.2f}'
        print(template.format(self.id,
                              self.train_loss.result(),
                              self.train_acc.result() * 100,
                              self.test_loss.result(),
                              self.test_acc.result() * 100
                              , t.time() - begin))

    def save(self, path):
        path_model = path + '/model_' + str(self.id) + '_weights'
        self.model.save_weights(path_model)
        np.save(path + '/model_' + str(self.id) + '_idx', self.idx)
        print('Saved information of classifier: ' + str(self.id))
    def load(self, path):
        path_model = path + '/model_' + str(self.id) + '_weights'
        self.model.load_weights(path_model)
        self.idx = np.load(path + '/model_' + str(self.id) + '_idx.npy')
        print('Loaded information of classifier: ' + str(self.id))


num_clsfr = 10
num_epoch = 15
clsfrs = []
for i in range(num_clsfr):
    clsfrs.append(Classifier(256, i))
for epoch in range(num_epoch):
    for i in range(num_clsfr):
        clsfrs[i].train_one_epoch(epoch)
        clsfrs[i].save('./models')
    print('----------------------------')

# Generate Samples for the Estimator
mie_samples_pred = []
mie_samples_lab = []
mie_labels = []
for i in range(num_clsfr):
    # for the train set
    idx = clsfrs[i].idx
    pred = clsfrs[i].model(x_train[idx[0:10000]]).numpy()
    mie_samples_pred.append(pred)
    mie_samples_lab.append(y_train[idx[0:10000]])
    mie_labels.append(np.ones((10000, 1)))
    # for the test set
    pred = clsfrs[i].model(x_test).numpy()
    mie_samples_pred.append(pred)
    mie_samples_lab.append(y_test)
    mie_labels.append(np.zeros((10000, 1)))
mie_samples_pred = np.vstack(mie_samples_pred)
mie_samples_lab = np.vstack(mie_samples_lab)
mie_labels = np.vstack(mie_labels)



