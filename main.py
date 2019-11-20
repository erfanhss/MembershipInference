import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import sys
import os
from keras import backend as K

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = 'cifar-10-batches-py'

    num_train_samples = 50000

    x_train_local = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train_local = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train_local[(i - 1) * 10000: i * 10000, :, :, :],
         y_train_local[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test_local, y_test_local = load_batch(fpath)

    y_train_local = np.reshape(y_train_local, (len(y_train_local), 1))
    y_test_local = np.reshape(y_test_local, (len(y_test_local), 1))

    if K.image_data_format() == 'channels_last':
        x_train_local = x_train_local.transpose(0, 2, 3, 1)
        x_test_local = x_test_local.transpose(0, 2, 3, 1)

    return (x_train_local, y_train_local), (x_test_local, y_test_local)


def classifier_model(input_image, image_labels):
    with tf.variable_scope("classifier_vars") as scope:
        c1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(input_image)
        c2 = tf.keras.layers.Conv2D(32, 3, activation='relu')(c1)
        p1 = tf.keras.layers.MaxPool2D(2, 2)(c2)
        d1 = tf.keras.layers.Dropout(0.25)(p1)

        c3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                    activation='relu')(d1)
        c4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(c3)
        p2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(c4)
        d2 = tf.keras.layers.Dropout(0.25)(p2)
        f1 = tf.keras.layers.Flatten()(d2)

        l1 = tf.keras.layers.Dense(512, activation='relu')(f1)
        d3 = tf.keras.layers.Dropout(0.5)(l1)
        logits = tf.keras.layers.Dense(10)(d3)

    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=image_labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy_loss)
    correct = tf.nn.in_top_k(logits, image_labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    return tf.nn.softmax(logits), loss, accuracy


def MIE(true_prob, pred_prob, indicator, idx=0):
    with tf.variable_scope('MIE_' + str(idx)) as scope:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(2))
        logit = model(tf.concat((true_prob, pred_prob), axis=1))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=indicator, logits=logit)
    mutual_info_estimate = -tf.reduce_mean(cross_entropy) + 0.69314718056
    correct = tf.nn.in_top_k(logit, indicator, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return mutual_info_estimate, accuracy


def perform_one_subset(train_x, train_y, test_x, test_y, sub_set_id, batch_size):
    print('Preparing data for subset '+str(sub_set_id))
    idx = np.random.permutation(len(train_x))
    clsfr_train_x = train_x[idx[0:15000]]
    clsfr_train_y = train_y[idx[0:15000]]
    # Train data for adversary
    adv_train_intrain_x = clsfr_train_x[0:10000]
    adv_train_intrain_y = clsfr_train_y[0:10000]
    adv_train_not_intrain_x = train_x[idx[15000:25000]]
    adv_train_not_intrain_y = train_y[idx[15000:25000]]
    # Test Data for Adversary
    adv_test_intrain_x = clsfr_train_x[10000:]
    adv_test_intrain_y = clsfr_train_y[10000:]
    adv_test_not_intrain_x = train_x[idx[25000:30000]]
    adv_test_not_intrain_y = train_y[idx[25000:30000]]
    # Training classifier
    print('Training Classifier subset '+str(sub_set_id))
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    Y = tf.placeholder(tf.int32, shape=[None])
    _, clsfr_loss, clsfr_acc = classifier_model(X, Y)
    clsfr_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
    clsfr_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier_vars')
    training_clsfr = clsfr_optimizer.minimize(clsfr_loss, var_list=clsfr_vars)
    num_epochs = 12
    saver = tf.train.Saver(var_list=clsfr_vars)
    path = './models/clsfr_subset' + str(sub_set_id) + '_best.clsfr'
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(num_epochs):
            rnd_idx = np.random.permutation(len(clsfr_train_x))
            print(epoch)
            for batch_idx in np.array_split(rnd_idx, len(clsfr_train_x // batch_size)):
                x_batch, y_batch = clsfr_train_x[batch_idx], clsfr_train_y[batch_idx]
                sess.run(training_clsfr, feed_dict={X: x_batch, Y: y_batch.reshape(-1)})
            test_loss, test_acc = sess.run([clsfr_loss, clsfr_acc], feed_dict={X: test_x, Y: test_y.reshape(-1)})
            train_loss, train_acc = sess.run([clsfr_loss, clsfr_acc],
                                             feed_dict={X: clsfr_train_x, Y: clsfr_train_y.reshape(-1)})
            print('Train Loss: ' + str(train_loss))
            print('Train Acc: ' + str(train_acc))
            print('Test Loss: ' + str(test_loss))
            print('Test Acc: ' + str(test_acc))
            saver.save(sess, path)
            print('saved variables!')
            print('----------------------------')
    print('Estimating Mutual Information for subset '+str(sub_set_id))
    MIE_subset = []
    for class_idx in range(10):
        print('Calculating for class '+str(class_idx))
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        Y = tf.placeholder(tf.int32, shape=[None])
        U = tf.placeholder(tf.int32, [None])
        clsfr_logits, clsfr_loss, clsfr_acc = classifier_model(X, Y)
        clsfr_prob = tf.nn.softmax(clsfr_logits)
        mutual_infor_estimate, attack_acc = MIE(tf.one_hot(Y, 10), clsfr_prob, U, class_idx)
        mie_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MIE_' + str(class_idx))
        attacker_optimizer = tf.train.AdamOptimizer(0.0001)
        attacker_operation = attacker_optimizer.minimize(-mutual_infor_estimate, var_list=mie_vars)

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path + '.meta')
            tf.global_variables_initializer().run()
            saver.restore(sess, path)
            print(sess.run([clsfr_loss, clsfr_acc], feed_dict={X: test_x, Y: test_y.reshape(-1)}))
            best_mie = 0
            for epoch in range(200):
                rnd_idx = np.random.permutation(len(adv_train_intrain_x))
                for batch_idx in np.array_split(rnd_idx, len(adv_train_intrain_x) // batch_size):
                    x_1 = adv_train_intrain_x[batch_idx]
                    y_1 = adv_train_intrain_y[batch_idx]
                    x_2 = adv_train_not_intrain_x[batch_idx]
                    y_2 = adv_train_not_intrain_y[batch_idx]
                    u = np.concatenate((np.ones(len(batch_idx)), np.zeros(len(batch_idx))), axis=0)
                    x = np.concatenate((x_1, x_2), axis=0)
                    y = np.concatenate((y_1, y_2), axis=0)
                    idx = np.where(y == class_idx)[0]
                    sess.run(attacker_operation, feed_dict={X: x[idx], Y: y[idx].reshape(-1), U: u[idx]})
                test_x_class = np.concatenate((adv_test_intrain_x, adv_test_not_intrain_x), axis=0)
                test_y_class = np.concatenate((adv_test_intrain_y, adv_test_not_intrain_y), axis=0)
                test_u = np.concatenate((np.ones(len(adv_test_intrain_x)), np.zeros(len(adv_test_not_intrain_x))), axis=0)
                idx = np.where(test_y_class == class_idx)[0]
                mie_class_test, attack_acc_class_test = sess.run([mutual_infor_estimate, attack_acc],
                                                                 feed_dict={X: test_x_class[idx],
                                                                            Y: test_y_class[idx].reshape(-1), U: test_u[idx]})
                if mie_class_test > best_mie:
                    best_mie = mie_class_test
                if epoch % 20 == 0:
                    print(epoch)
                    print('Test MIE: ' + str(mie_class_test))
                    print('Test Attacker Acc: ' + str(attack_acc_class_test))
                    print('----------------------------------------')

        MIE_subset.append(best_mie)
    return MIE_subset, test_acc, train_acc


print('Preparing data')
(train_x, train_y), (test_x, test_y) = load_data()
train_x = train_x / 255.
test_x = test_x / 255.
MIE_vec = []
test_acc_vec = []
train_acc_vec = []
for sub_set in range(15):
    mie_subset, test_acc_subset, train_acc_subset = perform_one_subset(train_x, train_y, test_x, test_y, sub_set, 512)
    MIE_vec.append(mie_subset)
    test_acc_vec.append(test_acc_subset)
    train_acc_vec.append(train_acc_subset)
MIE_vec = np.array(MIE_vec)
test_acc_vec = np.array(test_acc_vec)
train_acc_vec = np.array(train_acc_vec)
np.save('MIE_vector', MIE_vec)
np.save('test_accuracy_vector', test_acc_vec)
np.save('train_accuracy_vector', train_acc_vec)

MIE_vec = np.load('MIE_vector.npy')
acc_baseline = np.array([0.71, 0.64, 0.81, 0.87, 0.8, 0.84, 0.67, 0.72, 0.68, 0.7])
x_axis = np.arange(1, 11)
plt.plot(x_axis, MIE_vec.mean(0))
plt.plot(x_axis, acc_baseline)
plt.legend(['MI', 'Reported Accuracy'])
plt.show()