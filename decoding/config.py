import os
from os.path import dirname
import numpy as np


REPO_DIR = dirname(dirname(os.path.abspath(__file__)))
if 'chansingh' in os.path.expanduser('~'):
    DATA_PATH_TO_DERIVATIVE_DS003020 = '/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/'
    DATA_PATH_TO_DERIVATIVE_DS004510 = '/home/chansingh/mntv1/fmri_decoding/ds004510/derivative/'
    BIG_DATA_DIR = '/home/chansingh/mntv1/fmri_decoding'
else:
    DATA_PATH_TO_DERIVATIVE_DS003020 = os.path.join(REPO_DIR, 'ds003020', 'derivative')
    DATA_PATH_TO_DERIVATIVE_DS004510 = os.path.join(REPO_DIR, 'ds004510', 'derivative')
    BIG_DATA_DIR = REPO_DIR


# DERIVATIVE stuff above is downloaded from openneuro
# these three subfolders are downloaded from the links at https://github.com/HuthLab/semantic-decoding
DATA_LM_DIR = os.path.join(BIG_DATA_DIR, "data_lm")
DATA_TRAIN_DIR = os.path.join(BIG_DATA_DIR, "data_train")
DATA_TEST_DIR = os.path.join(BIG_DATA_DIR, "data_test")
# for e2e_baseline, additionally download "released" folder under BIG_DATA_DIR

RESULT_DIR = os.path.join(REPO_DIR, "results")
MODEL_DIR = os.path.join(RESULT_DIR, "models")
SCORE_DIR = os.path.join(RESULT_DIR, "scores")



# GPT encoding model parameters
TRIM = 5
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(1, 3, 10)
NBOOTS = 10
VOXELS = 10000
CHUNKLEN = 40
GPT_LAYER = 9
GPT_WORDS = 5

# decoder parameters
RANKED = True
WIDTH = 200
NM_ALPHA = 2/3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.1
EXTENSIONS = 5

# evaluation parameters
WINDOW = 20

# devices
GPT_DEVICE = "cuda"
EM_DEVICE = "cuda"
SM_DEVICE = "cuda"


def map_to_uts_subject(subject):
    if subject.startswith('S0'):
        return 'UT' + subject
    elif subject.startswith('S'):
        return 'UTS0' + subject[1:]
    else:
        return subject

if __name__ == '__main__':
    print(REPO_DIR)