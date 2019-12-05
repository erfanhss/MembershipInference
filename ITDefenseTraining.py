from utils import *


num_clsfr = 10
num_epoch = 100
clsfrs = []
for i in range(num_clsfr):
    clsfrs.append(Classifier(512, i, 5))
mie_est = Adversary()
mie_arr = []
mie_est.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()])
best_acc = np.zeros(num_clsfr)
for epoch in range(num_epoch):
    if epoch > 3:
      mie_data, mie_lbl = generate_samples_mie(clsfrs, 10000)
      mie_res = mie_est.fit(mie_data, mie_lbl, batch_size=512, epochs=5, verbose=0)
      log_loss = mie_res.history['loss']
      print('Mutual Information Estimate: ' + str(-log_loss[-1] + np.log(2)))
      mie_arr.append(-log_loss[-1] + np.log(2))
    for i in range(num_clsfr):
        test_acc = clsfrs[i].train_one_epoch(epoch, mie_est)
        if test_acc > best_acc[i]:
          best_acc[i] = test_acc
          clsfrs[i].save('./models')
    print('----------------------------')
    
    
# Loading best Models
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for i in range(num_clsfr):
    clsfrs.append(Classifier(512, i, 5))
    clsfrs[i].load('./models')
    tr_ls, tr_acc, ts_ls, ts_acc = clsfrs[i].evaluate()
    train_loss.append(tr_ls)
    train_acc.append(tr_acc)
    test_loss.append(ts_ls)
    test_acc.append(ts_acc)
    
# Evaluating Membership Inference Accuracy  
history_arr = []
for i in range(num_clsfr):
  print('Classifier: ' + str(i))
  mie_samples, mie_labels = generate_samples_mie([clsfrs[i]], 10000)
  mie_train_x, mie_test_x, mie_train_y, mie_test_y = train_test_split(mie_samples, mie_labels, test_size=0.2)
  adv = Adversary()
  adv.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()])
  history_arr.append(adv.fit(mie_train_x, mie_train_y, batch_size=512, epochs=100, verbose=0, 
                             validation_data=[mie_test_x, mie_test_y]))
  print('Attack Accuracy: ' + str(np.max(history_arr[i].history['val_binary_accuracy'])))
  
  
  
# Estimation Mutual Information
mie_samples, mie_labels = generate_samples_mie(clsfrs, 10000)
mie_train_x, mie_test_x, mie_train_y, mie_test_y = train_test_split(mie_samples, mie_labels, test_size=0.2)
adv = Adversary()
adv.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
          metrics=[tf.keras.metrics.BinaryAccuracy()])
history = adv.fit(mie_train_x, mie_train_y, batch_size=512, epochs=100, verbose=0, 
                             validation_data=[mie_test_x, mie_test_y])
print('MIE:' + str(np.log(2) - np.min(history.history['val_loss'])))
  
