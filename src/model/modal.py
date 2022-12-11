import torch

class 이게Attention이맞을까(torch.nn.Module):
    """
    input [B, Feature]인 layer self attention 적용
    """
    def __init__(self, dim, head_dim=64, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0, '헤드 수 정수 아님'
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, F = x.shape # (Batch, Feature)
        
        # qkv = (3, B, H, H_dim)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3) 
        q, k, v = qkv.unbind(0)  
        
        # attention weight 계산
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # context vector 계산
        x = (attn @ v).reshape(B, F)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class AttentionLayer(torch.nn.Module):
    def __init__(self, dim=512, head_dim = 64, 
                 act_layer=torch.nn.GELU, 
                 norm_layer=torch.nn.BatchNorm1d, 
                 drop=0.,
                 use_layer_scale=True, 
                 layer_scale_init_value=1e-5):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.atten = 이게Attention이맞을까(dim=dim, head_dim=head_dim, 
                                    attn_drop=drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.fc1 = torch.nn.Linear(dim, dim, bias=True) 
        self.act = act_layer()
        self.drop1 = torch.nn.Dropout(drop)

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = torch.nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = torch.nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.layer_scale_1* self.atten(self.norm1(x))
            x = x + self.layer_scale_2* self.drop1(self.act(self.fc1(self.norm2(x))))
        else:
            x = x + self.atten(self.norm1(x))
            x = x + self.drop1(self.act(self.fc1(self.norm2(x))))
        return x

class Modal(torch.nn.Module):
    def __init__(self, args):
        super(Modal, self).__init__()
        self.args = args
        
        def get_model(model, args, pretrained=False):
            mdl = torch.nn.DataParallel(model(args)) if args.multi_gpu else model(args)
            if not pretrained:
                return mdl
            else:
                print("기학습 웨이트")
                mdl.load_state_dict(torch.load(pretrained))
                return mdl
            
        # image model 구축
        self.img_model = get_model(self.args.img_model, args, self.args.img_path).encoder
        self.img_model.head = torch.nn.Linear(self.img_model.head.in_features, self.args.hidden_dim)

        # text model 구축
        self.text_model = get_model(self.args.text_model, args, self.args.text_path)
        self.text_model.linear = torch.nn.Linear(self.text_model.linear.in_features, self.args.hidden_dim)
      
        if self.args.atten_use:
            self.attention1 = torch.nn.Sequential(*[AttentionLayer(self.args.hidden_dim * 2, 
                                                               self.args.head_dim, 
                                                               drop=args.drop_path_rate) for _ in range(args.layer_num)])
        
        self.output = torch.nn.Linear(self.args.hidden_dim * 2, self.args.output_dim)
        
        # freeze
        if self.args.modal_freeze:
            # image model
            for name, param in self.img_model.named_parameters():
                if name not in list(self.img_model.state_dict().keys())[-2:]:
                    param.requires_grad = False
            # text model
            for name, param in self.text_model.named_parameters():
                if name not in list(self.text_model.state_dict().keys())[-2:]:
                    param.requires_grad = False
        pass
    
    def forward(self, x):
        x_img = self.img_model(x['image'])
        x_text = self.text_model(x)
        x_concat = torch.cat([x_text, x_img],1)
        if self.args.atten_use:
            x_concat = self.attention1(x_concat) 
        logits = self.output(x_concat)
        return logits