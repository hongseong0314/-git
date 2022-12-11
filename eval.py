import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from src.utill import stable_softmax  
from config import config

def get_model(model, args, pretrained=False):
    mdl = torch.nn.DataParallel(model(args)) if args.multi_gpu else model(args)
    if not pretrained:
        return mdl
    else:
        print("기학습 웨이트")
        mdl.load_state_dict(torch.load(pretrained))
        return mdl

def condition_pdf(cat1_pred, cat3_pred, coder):
    cat1_prob = stable_softmax(cat1_pred)
    cat3_prob = stable_softmax(cat3_pred)
    
    for c1 in range(6):
        condition = cat1_prob[..., c1]
        cat3_prob[..., coder[str(c1)]] *= condition[..., np.newaxis]
        
    return np.argmax(cat3_prob, axis=-1)

def tester(args, path:os.path, cat1=False):
    dir_root = r'E:\관광'
    args.dir_root = dir_root
    df_test = pd.read_csv(dir_root + '/test.csv')
    test_datasetcat3 = args.Dataset(df_test, args, mode='test')
    cat3_loader = DataLoader(
            test_datasetcat3,
            batch_size = args.batch_size,
            shuffle = False,
        )

    model3 = get_model(args.model_class, args, path)
    model3.to(args.device)
    model3.eval()

    cat3_list = []
    for batch_data in tqdm(cat3_loader):
        batch_data['input_ids'] = batch_data['input_ids'].to(args.device)
        batch_data['attention_mask'] = batch_data['attention_mask'].to(args.device)
        batch_data['image'] = batch_data['image'].to(args.device)
                
        with torch.no_grad():
            model_pred  = model3(batch_data) 
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
        args.output_dim = 6
        args.label = 'cat1'
        test_datasetcat1 = args.Dataset(df_test, args, mode='test')
        cat1_loader  = DataLoader(
            test_datasetcat1,
            batch_size = args.batch_size,
            shuffle = False,
        )
        model1 = get_model(args.model_class, args, cat1)
        model1.to(device)
        model1.eval()

        cat1_list = []
        for batch_data in tqdm(cat1_loader):
            batch_data['input_ids'] = batch_data['input_ids'].to(args.device)
            batch_data['attention_mask'] = batch_data['attention_mask'].to(args.device)
            batch_data['image'] = batch_data['image'].to(args.device)

            with torch.no_grad():
                model_pred  = model1(batch_data)

                model_pred = torch.argmax(model_pred, dim=1).detach().cpu()
                cat1_list.extend(model_pred.numpy())
        
        with open(os.path.join("endecoder", "cat1tocat3"), 'r') as rf:
            coder = json.load(rf)

        submission.cat3 = condition_pdf(cat1_list, cat3_list, coder)
        submission.cat3 = submission.cat3.apply(lambda x:en2cat[str(x)])
        save_name = os.path.join(dir_root, path.split("\\")[-2] + 'cat1.csv')
        submission.to_csv(save_name, index=False)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    p1 = r"E:\관광\cat1_modal_c\model_poolformer_m36_0_-0.9882.pth"
    p3 = r"E:\관광\cat3_modal_c\model_poolformer_m36_0_-0.8934.pth"
    args = config()
    args.device = device
    args.batch_size = 64

    tester(args, path=p3, cat1=p1)