import time as t

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.enable_eager_execution()
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

# Adversary 
class Adversary(tf.keras.Model):
    def __init__(self):
        super(Adversary, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense_2_1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense_2_2 = tf.keras.layers.Dense(512, activation='relu')
        self.dense_3_1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_3_2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(256, activation='relu')
        self.dense_5 = tf.keras.layers.Dense(64, activation='relu')

        
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')
    def call(self, x):
        x_1 = x[:, 0:10]
        x_2 = x[:, 10:20]
        
        x_1 = self.dense_1(x_1)
        x_1 = self.dense_2_1(x_1)
        x_1 = self.dense_3_1(x_1)


        x_2 = self.dense_2_2(x_2)
        x_2 = self.dense_3_2(x_2)

        x = tf.concat((x_1, x_2), axis=1)
        x = self.dense_4(x)
        x = self.dense_5(x)
        
        return self.out(x)



class Classifier:
    def __init__(self, batch_size, id, lmbd):
        self.id = id
        self.model = ClassifierModel()
        self.idx = np.random.RandomState(seed=id).permutation(len(x_train))[0:size_of_subsets]
        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train[self.idx], y_train[self.idx])).shuffle(size_of_subsets).batch(batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(batch_size)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.lmbd = lmbd
        self.train_acc_arr = []
        self.train_loss_arr = []
        self.test_acc_arr = []
        self.test_loss_arr = []

    def train_one_step(self, images, labels, mie_model):
        loss_object_mie = tf.keras.losses.BinaryCrossentropy()
        with tf.GradientTape() as tape:
            pred = self.model(images)
            loss = self.loss_object(labels, pred)
            labels_one_hot = tf.one_hot(labels.numpy().reshape(-1), 10)
            input_to_mie = tf.concat((pred, tf.dtypes.cast(labels_one_hot, tf.double)), 1)
            log_loss = loss_object_mie(tf.ones(len(images)), mie_model(input_to_mie))
            mie = np.log(2) - log_loss
            total_loss = loss + self.lmbd*tf.dtypes.cast(mie, tf.double)
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
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

    def train_one_epoch(self, epoch_number, mie_model):
        self.train_loss.reset_states()
        self.train_acc.reset_states()
        self.test_loss.reset_states()
        self.test_acc.reset_states()
        begin = t.time()
        for images, labels in self.train_ds:
            self.train_one_step(images, labels, mie_model)
        for images, labels in self.test_ds:
            self.test_one_step(images, labels, type='test')
        self.train_acc_arr.append(self.train_acc.result() * 100)
        self.train_loss_arr.append(self.train_loss.result())
        self.test_acc_arr.append(self.test_acc.result() * 100)
        self.test_loss_arr.append(self.test_loss.result())
        template = 'ID: {}, Epoch {}, Train Loss: {:.2f}, Train Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}, Run Time: {:.2f}'
        print(template.format(self.id,
                              epoch_number + 1,
                              self.train_loss.result(),
                              self.train_acc.result() * 100,
                              self.test_loss.result(),
                              self.test_acc.result() * 100
                              , t.time() - begin))
        return self.test_acc.result() * 100

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
        return self.train_loss.result(), self.train_acc.result() * 100, self.test_loss.result(), self.test_acc.result() * 100

    def save(self, path):
        path_model = path + '/model_' + str(self.id) + '_weights'
        self.model.save_weights(path_model)
        print('Saved information of classifier: ' + str(self.id))
    def load(self, path):
        path_model = path + '/model_' + str(self.id) + '_weights'
        self.model.load_weights(path_model)
        print('Loaded information of classifier: ' + str(self.id))


def generate_samples_mie(clsfrs, num_samples_per_group):
  mie_samples_pred = []
  mie_samples_lab = []
  mie_labels = []
  source = np.arange(len(x_train))
  for i in range(len(clsfrs)):
      # for the train set
      idx = clsfrs[i].idx
      pred = clsfrs[i].model(x_train[idx[0:num_samples_per_group]]).numpy()
      mie_samples_pred.append(pred)
      mie_samples_lab.append(y_train[idx[0:num_samples_per_group]])
      mie_labels.append(np.ones((num_samples_per_group, 1)))
      # for the test set
      rest = np.setdiff1d(source, idx)
      pred = clsfrs[i].model(x_train[rest[0:num_samples_per_group]]).numpy()
      mie_samples_pred.append(pred)
      mie_samples_lab.append(y_train[rest[0:num_samples_per_group]])
      mie_labels.append(np.zeros((num_samples_per_group, 1)))
  mie_samples_pred = np.vstack(mie_samples_pred)
  mie_samples_lab = np.vstack(mie_samples_lab)
  mie_labels = np.vstack(mie_labels)
  input_data = np.concatenate([mie_samples_pred, tf.one_hot(mie_samples_lab.reshape(-1), 10).numpy()], axis=1)
  return input_data, mie_labels
