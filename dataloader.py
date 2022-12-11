import os
from PIL import Image
import torch
from torchvision import transforms
import json

def expand2square(pil_img):
    background_color = (0,0,0)
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class ModalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                files,
                args,
                mode='train', 
                ):
        self.files = files
        self.pad = args.pad
        self.mode = mode
        self.label = args.label
        self.root = args.dir_root
        self.tokenizer = args.tokenizer
        self.max_len = args.max_len
        self.img_size = args.img_size
        
        self.train_mode = transforms.Compose([
                    transforms.Resize((self.img_size,self.img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                            ])

        self.test_mode = transforms.Compose([
                        transforms.Resize((self.img_size,self.img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])

        # label encoder, decoder 
        with open(os.path.join("endecoder", self.label), 'r') as rf:
            coder = json.load(rf)
            self.cat2en = coder['{:s}toen'.format(self.label)]
            self.en2cat = coder['ento{:s}'.format(self.label)]
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = self.files.iloc[index, ]
        text = str(data.overview)
        
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding = 'max_length',
          truncation = True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        
        image = Image.open(os.path.join(self.root, data.img_path)).convert('RGB')
        if self.mode == 'train':
            labels = data[self.label]
            labels = self.cat2en[labels]
            if self.pad:
                image = expand2square(image)
            try:
                image = self.train_mode(image)
            except TypeError:
                print(data, image)
            return {
                  'input_ids': encoding['input_ids'].flatten(),
                  'attention_mask': encoding['attention_mask'].flatten(),
                  'image':image,
                  'labels': labels,
                }
        elif self.mode == self.mode == 'valid':
            labels = data[self.label]
            labels = self.cat2en[labels]
            if self.pad:
                image = expand2square(image)
            image = self.test_mode(image)
            return {
                  'input_ids': encoding['input_ids'].flatten(),
                  'attention_mask': encoding['attention_mask'].flatten(),
                  'image':image,
                  'labels':labels,
                }
            
        elif self.mode == 'test':
            if self.pad:
                image = expand2square(image)
            image = self.test_mode(image)
            return {
                  'input_ids': encoding['input_ids'].flatten(),
                  'attention_mask': encoding['attention_mask'].flatten(),
                  'image':image,
                }
    
class UnNormalize(object):
    """
    정규화 inverse 한다.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def tensor2img(img):
    """
    tensor 형태에서 numpy 배열로 바꿔서 반환
    """
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    a = unorm(img).numpy()
    a = a.transpose(1, 2, 0)
    return a