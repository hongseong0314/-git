import torch
from src.model.meta import PoolFormer

class Modal(torch.nn.Module):
    def __init__(self, args):
        super(Modal, self).__init__()
        self.args = args
        
        def get_model(model, args, pretrained=False):
            mdl = torch.nn.DataParallel(model(args)) if False else model(args)
            if not pretrained:
                return mdl
            else:
                print("기학습 웨이트")
                mdl.load_state_dict(torch.load(pretrained))
                return mdl
            
        # image model 구축
        if not self.args.img_path:
            self.img_model = get_model(self.args.img_model, args, self.args.img_path).encoder
            self.img_model.head = torch.nn.Linear(self.img_model.head.in_features, self.args.hidden_dim)
        else:
            self.img_model = get_model(self.args.img_model, args).encoder
        # text model 구축
        self.atte1 = torch.nn.MultiheadAttention(self.args.hidden_dim * 2, self.args.num_heads, dropout=0.2, bias=True)
        self.output = torch.nn.Linear(self.args.hidden_dim * 2, self.args.output_dim)
        
        # freeze
        if self.args.modal_freeze:
            # image model
            for name, param in self.img_model.named_parameters():
                if name not in list(self.img_model.state_dict().keys())[-2:]:
                    param.requires_grad = False
            # text model

        pass
    
    def forward(self, images, texts):
        x_img = self.img_model(images)
        x_text = self.text_model(texts)
        x_concat = torch.cat([x_text, x_img],1)
        """
        x = self.atte1(x_concat) Q, K, V
        activation
        dropout
        """
        logits = self.output(x)
        return logits