import sys
import os
import random
import pandas as pd
import numpy as np
import torch
from easydict import EasyDict

from dataloader import ModalDataset
from src.model.meta import PoolFormer
from src.model.text_model import TextModel
from src.model.modal import Modal 
from src.trainer import Trainer
from transformers import AutoTokenizer

save_path = os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# 전체
dir_root = r'E:\관광'
meta_df = pd.read_csv(dir_root + '/train.csv')

# cat weights
weights_c1 = np.load("cat1.npy")
weights_c2 = np.load("cat2.npy")
weights_c3 = np.load("cat3.npy")

args = EasyDict(
    {
    # Path settings
    'root':'train_dir',
    'dir_root':dir_root,
    'save_dict' : os.path.join(dir_root, 'cat1_modal_c'),
    'df':meta_df,
     
    # Model parameter settings
    'drop_path_rate':0.2,
    'model_class': Modal,
    'weight':weights_c1,
    'pretrained':False,#"E:\관광\cat3_pool_h_224_focal\model_poolformer_m36_0_0.0268.pth",
    
    ## modal settings
    'hidden_dim':256,
    'modal_path':False,
    'img_path':r"E:\관광\cat1_pool_h_224_adamw_c\model_poolformer_m36_0_-0.7047.pth", #이미지 기학습
    'text_path':r"E:\관광\model_save (2)\klue_bert_cat1_model.pth", #텍스트 기학습
    'atten_use':False,
    'head_dim':64, #atten head dim
    'modal_freeze':True, # image and text model freeze
    'layer_num':2,

    ## image model
    'img_model':PoolFormer,
    'CODER':'poolformer_m36', # 'regnety_040', 'efficientnet-b0' ,poolformer_m36
    'freeze':False,
    'img_size':224,
    'test_size':224,

    ## text model
    'text_model':TextModel,
    'tokenizer':AutoTokenizer.from_pretrained("klue/bert-base"),
    'max_len':512,

    # Training parameter settings
    ## Base Parameter
    'BATCH_SIZE':100,
    'epochs':200,
    'optimizer':'adamw', #adam,Lamb,SAM, adamw
    'lr':5e-6,
    'weight_decay':1e-3,
    'Dataset' : ModalDataset,
    'fold_num':1, 
    'bagging_num':1,
    'label':'cat1',
    'loss_type':'focal',
    'beta':0.9999,
    'gamma':2.0,

    ## Augmentation
    'pad':True,

    #scheduler 
    'scheduler':'cos', #cycle
    ## Scheduler (OnecycleLR)
    'warm_epoch':5,
    'max_lr':1e-3,

    ### Cosine Annealing
    'min_lr':5e-6,
    'tmax':145,

    ## etc.
    'patience':20,
    'clipping':None,

    # Hardware settings
    'amp':False,
    'multi_gpu':False,
    'logging':False,
    'num_workers':4,
    'seed':42,
    'device':device,
    })

# def seed_everything(seed):
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

if __name__ == '__main__': 
    # seed_everything(args.seed)
    print(args.CODER + " train..")
    trainer = Trainer(args)
    trainer.fit()