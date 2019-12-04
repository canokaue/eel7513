import time
import torch
import socket as _socket
import time as _time
from src.networks import WholeNetwork as _net
from src.lm_networks import LandmarkBranchUpsample as _lm_branch
from src.utils import Evaluator as _evaluator

#base_path = '/Users/canok/Documents/MachineLearningUFSC/DeepFashion/'
#base_path = '/DeepFashion/'
base_path = '/home/kauecano/'

_name = 'proj_2-nvidia'

# gpu-10k-full-we100-e20-b32-wo10

NUM_WORKERS = 10
NUM_EPOCH = 20
LEARNING_RATE = 0.0001
#BATCH_SIZE = 16
#VAL_BATCH_SIZE = 32

BATCH_SIZE = 32
VAL_BATCH_SIZE = 40

WEIGHT_ATTR_NEG = 0.01
WEIGHT_ATTR_POS = 1
WEIGHT_LANDMARK_VIS_NEG = 0.5
WEIGHT_LANDMARK_VIS_POS = 0.5


# LOSS WEIGHT
WEIGHT_LOSS_CATEGORY = 1
WEIGHT_LOSS_ATTR = 20
WEIGHT_LOSS_LM_POS = 100


_time = _time.strftime('%m-%d %H:%M:%S', _time.localtime())

# Dataset
gaussian_R = 8
DATASET_PROC_METHOD_TRAIN = 'BBOXRESIZE'
DATASET_PROC_METHOD_VAL = 'BBOXRESIZE'
########

# Network
USE_NET = _net
LM_SELECT_VGG = 'conv4_3'
LM_SELECT_VGG_SIZE = 28
LM_SELECT_VGG_CHANNEL = 512
LM_BRANCH = _lm_branch
EVALUATOR = _evaluator
#################

# Learning Scheme
# mudar modelo de learning rate
LEARNING_RATE_DECAY = 0.8
WEIGHT_LOSS_LM_POS = 10
#################

# auto
TRAIN_DIR = 'runs/%s/' % _name + _time.replace(":","-").replace(" ","_")
VAL_DIR = 'runs/%s/' % _name + _time.replace(":","-").replace(" ","_")

MODEL_NAME = '%s.pkl' % _name
#############

_hostname = str(_socket.gethostname())

name = time.strftime('%m-%d %H:%M:%S', time.localtime())

FASHIONET_LOAD_VGG16_GLOBAL = False

DATASET_PROC_METHOD_TRAIN = 'RANDOM'
DATASET_PROC_METHOD_VAL = 'LARGESTCENTER'

# 0: no sigmoid 1: sigmoid
VGG16_ACT_FUNC_IN_POSE = 0



# VAL
VAL_CATEGORY_TOP_N = (1, 3, 5)
VAL_ATTR_TOP_N = (3, 5)
VAL_LM_RELATIVE_DIS = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lm2name = ['L.Col', 'R.Col', 'L.Sle', 'R.Sle', 'L.Wai', 'R.Wai', 'L.Hem', 'R.Hem']
attrtype2name = {1: 'texture', 2: 'fabric', 3: 'shape', 4: 'part', 5: 'style'}

VAL_WHILE_TRAIN = True

USE_CSV = 'info.csv'

LM_TRAIN_USE = 'vis'
LM_EVAL_USE = 'vis'

TRAIN_SPLIT_LEN = 10000
VAL_SPLIT_LEN = 10000