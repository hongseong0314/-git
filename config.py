from easydict import EasyDict
from transformers import AutoTokenizer
from dataloader import ModalDataset
from src.model.meta import PoolFormer
from src.model.text_model import TextModel
from src.model.modal import Modal 

def config():
    args = EasyDict(
        { 
        # Model parameter settings
        'drop_path_rate':0.0,
        'model_class': Modal,
        'pretrained':False,
        
        ## modal settings
        'hidden_dim':256,
        'output_dim':128,
        'modal_path':False,
        'img_path':False,#r"E:\관광\check\image_weight.pth", #이미지 기학습
        'text_path':False,#r"E:\관광\model_save\klue_bert_cat3_model.pth", #텍스트 기학습
        'atten_use':False,
        'head_dim':64, #atten head dim
        'modal_freeze':True, # image and text model freeze

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
        'BATCH_SIZE':128,
        'Dataset' : ModalDataset,
        'label':'cat3',

        ## Augmentation
        'pad':True,


        # Hardware settings
        'multi_gpu':False,
        })
    return args