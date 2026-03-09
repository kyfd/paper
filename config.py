import os
from easydict import EasyDict as edict
import time

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035
__C.DATASET = 'DeepFish'
__C.NAME = 'train'
__C.encoder = "VGG16_FPN"
__C.RESUME = False
__C.RESUME_PATH = ''
__C.PRE_TRAIN_COUNTER = '/root/autodl-tmp/FishCount_Project/datasets/dataset_prepare/MDC_pre_trained_counter_vgg_fpn_200.pth'

__C.GPU_ID = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = __C.GPU_ID

__C.cross_attn_embed_dim = 256
__C.cross_attn_num_heads = 4
__C.mlp_ratio = 4
__C.cross_attn_depth = 2
__C.USE_ENHANCE = False
__C.FEATURE_DIM = 256

# learning rate settings
__C.LR_Base = 1e-5
__C.WEIGHT_DECAY = 1e-6

__C.MAX_EPOCH = 150 
__C.VAL_INTERVAL = 1
__C.START_VAL = 0
__C.PRINT_FREQ = 20

# print
now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
    + '_' + __C.DATASET \
    + '_' + str(__C.LR_Base) \
    + '_' + __C.NAME

__C.VAL_VIS_PATH = './exp/'+__C.DATASET+'_val'
__C.EXP_PATH = os.path.join('./exp', __C.DATASET)
if not os.path.exists(__C.EXP_PATH ):
    os.makedirs(__C.EXP_PATH )