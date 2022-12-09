import torch
from transformers import BertModel

class TextModel(torch.nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained('klue/bert-base')
        self.dropout = torch.nn.Dropout(args.drop_path_rate)
        self.linear = torch.nn.Linear(768, args.output_dim)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, 
                                    attention_mask=mask,
                                    return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output