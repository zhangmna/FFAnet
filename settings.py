import os
import logging

res_block_num=8  #4,6,8---0.8506   
connection_style = 'dense_connection'#symmetric_connection;no_connection;multi_short_skip_connection,'dense-connection'
feature_map_num=10 #8,10,12

pyramid_num=3#2,3,4
dilation_num=3#2,3,4
use_se=False
use_bn=False
use_tree=True
dilation=True
unit='my_unit'# 'traditional'49835,'res'49835,'dilation'52889,'GRU'53289,'my_unit'50045

aug_data = False # Set as False for fair comparison
batch_size = 10
patch_size = 128
lr = 1e-3

# data_dir = '/media/supercong/d277df79-f0d6-4f1d-979c-d79f956a61e5/congwang/dataset/rain100H'
data_dir = '/data4/zhangman/rain100H'                        
log_dir = './logdir'
show_dir = './showdir'
model_dir = './models'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest')
save_steps = 10000

num_workers = 8
num_GPU = 1
device_id = '0,1,2,3'
# device_id = '0'


logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


