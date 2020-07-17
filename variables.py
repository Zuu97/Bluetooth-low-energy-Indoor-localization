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
n_features = 13
seed = 7
dense = 100
output = 2
learning_rate=0.001
batch_size= 64
num_epoches = 500
ann_weights = os.path.join(os.getcwd(),'data/ble_rssi_ann.h5')
cdf_error_img = os.path.join(os.getcwd(),'data/CDF_error_cnn.png')

#Random Forest parameters
# seed = 42
# depth = 200

# CNN paramaters
# beacon_coords = {
#             "b3001": (5, 9),
#             "b3002": (9, 14),
#             "b3003": (13, 14),
#             "b3004": (18, 14),
#             "b3005": (9, 11),
#             "b3006": (13, 11),
#             "b3007": (18, 11),
#             "b3008": (9, 8),
#             "b3009": (2, 3),
#             "b3010": (9, 3),
#             "b3011": (13, 3),
#             "b3012": (18, 3),
#             "b3013": (22, 3),
#                  }

# batch_size = 32
# seed = 7
# num_epoches = 1000
# img_width = 25
# img_height = 25
# learning_rate=0.1
# cnn_weights = os.path.join(os.getcwd(),'data/ble_rssi_cnn.h5')
# cdf_error_cnn = os.path.join(os.getcwd(),'data/CDF_error_cnn.png')