import os
data_path = os.path.join(os.getcwd(),'iBeacon_RSSI_Labeled.csv')
model_weights = os.path.join(os.getcwd(),'ble_rssi.h5')

n_features = 13
dense1 = 512
dense2 = 512
dense3 = 64
keep_prob = 0.3

batch_size= 64
num_epoches = 30
validation_split = 0.2