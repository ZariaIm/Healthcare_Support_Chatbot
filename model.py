from transformers import DistilBertModel
import torch.nn as nn



class FineTunedModel(nn.Module):
    def __init__(self, output_size, model_name):
        super(FineTunedModel, self).__init__()
        self.base_model = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, output_size) 
        
    def forward(self, input_ids, attn_mask):
        #print(input_ids.shape, attn_mask.shape, labels.shape)
        outputs = self.base_model(input_ids, attention_mask=attn_mask)
        outputs = self.dropout(outputs.last_hidden_state)
        outputs = self.linear(outputs)
        return outputs

class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out