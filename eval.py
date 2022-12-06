import torch
from src.model.meta import PoolFormer_test
import os
import pandas as pd
from dataloader import Dataset_test
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json

def get_model(model, encoder_name, dim, pretrained=False):
    mdl = torch.nn.DataParallel(model(encoder_name, dim)) if False else model(encoder_name, dim)
    if not pretrained:
        return mdl
    else:
        print("기학습 웨이트")
        mdl.load_state_dict(torch.load(pretrained))
        return mdl

def tester(device, path:os.path, cat1=False, batch_size=16):
    dir_root = r'E:\관광'
    df_test = pd.read_csv(dir_root + '/test.csv')
    test_datasetcat3 = Dataset_test(df_test, "cat3", 224)
    cat3_loader = DataLoader(
            test_datasetcat3,
            batch_size = batch_size,
            shuffle = False,
        )

    model3 = get_model(PoolFormer_test, "poolformer_m36",128, path)
    model3.to(device)
    model3.eval()

    cat3_list = []
    for batch_data in tqdm(cat3_loader):
        images = batch_data['image']
        images = images.to(device)
        with torch.no_grad():
            model_pred  = model3(images) 
            cat3_list.extend(model_pred.detach().cpu().numpy())
    cat3_list = np.vstack(cat3_list)
    
    submission = pd.read_csv(dir_root + '/sample_submission.csv')
    with open(os.path.join("endecoder", "cat3"), 'r') as rf:
        coder = json.load(rf)
        cat2en = coder['{:s}toen'.format("cat3")]
        en2cat = coder['ento{:s}'.format("cat3")]

    if not cat1:
        submission.cat3 = np.argmax(cat3_list, axis=-1)
        submission.cat3 = submission.cat3.apply(lambda x:en2cat[str(x)])
        save_name = os.path.join(dir_root, path.split("\\")[-2] + '.csv')
        submission.to_csv(save_name, index=False)

    else:
        test_datasetcat1 = Dataset_test(df_test, "cat1", 224)
        cat1_loader  = DataLoader(
            test_datasetcat1,
            batch_size = batch_size,
            shuffle = False,
        )

        model1 = get_model(PoolFormer_test, "poolformer_m36", 6, cat1)
        model1.to(device)
        model1.eval()

        cat1_list = []
        for batch_data in tqdm(cat1_loader):
            images = batch_data['image']
            images = images.to(device)
            with torch.no_grad():
                model_pred  = model1(images)

                model_pred = torch.argmax(model_pred, dim=1).detach().cpu()
                cat1_list.extend(model_pred.numpy())
        
        with open(os.path.join("endecoder", "cat1tocat3"), 'r') as rf:
            coder = json.load(rf)
        
        pred_label = []
        for c1, c3 in zip(cat1_list, cat3_list):
            c2c = coder[str(c1)]
            max_idx = np.argmax(c3[c2c])
            pred_label.append(c2c[max_idx])
        submission.cat3 = pred_label
        submission.cat3 = submission.cat3.apply(lambda x:en2cat[str(x)])
        save_name = os.path.join(dir_root, path.split("\\")[-2] + 'cat1.csv')
        submission.to_csv(save_name, index=False)
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    p1 = r"E:\관광\cat1_pool_h_224_adamw_focal\model_poolformer_m36_0_-0.6977.pth"
    p3 = r"E:\관광\check\checkpoint1.pth"
    tester(device, path=p3, cat1=p1)