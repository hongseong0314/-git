import timm
import torch

class PoolFormer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = timm.create_model(args.CODER, pretrained=True,
                                    drop_path_rate=args.drop_path_rate,
                                    )
        
        num_head = self.encoder.head.in_features
        self.encoder.head = torch.nn.Linear(num_head, args.output_dim)

    def forward(self, x):
        x = self.encoder(x)
        return x

class PoolFormer_test(torch.nn.Module):
    def __init__(self, encoder_name, output_dim=128):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=False,
                                    drop_path_rate=0,
                                    )
        
        num_head = self.encoder.head.in_features
        self.encoder.head = torch.nn.Linear(num_head, output_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        return x