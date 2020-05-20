import os
data_path = os.path.join(os.getcwd(),'data/iBeacon_RSSI_Labeled.csv')
image_path = os.path.join(os.getcwd(),'data/iBeacon_Layout.jpg')
validation_split = 0.2

#MLP parameters
# n_features = 13
# dense1 = 512
# dense2 = 512
# dense3 = 64
# keep_prob = 0.3

# batch_size= 64
# num_epoches = 30
# mlp_weights = os.path.join(os.getcwd(),'data/ble_rssi_mlp.h5')


#ANN paramaters
# n_features = 13
# seed = 7
# dense = 50
# output = 2

# learning_rate=0.001
# batch_size= 64
# num_epoches = 1000
# ann_weights = os.path.join(os.getcwd(),'data/ble_rssi_ann.h5')
# cdf_error_img = os.path.join(os.getcwd(),'data/CDF_error.png')

#Random Forest parameters
seed = 42
# depth = 200
# rf_weights = os.path.join(os.getcwd(),'data/ble_rssi_rf.pickle')