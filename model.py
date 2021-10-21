from transformers import DistilBertModel
import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################################
class FineTunedModel(nn.Module):
    def __init__(self, output_size, model_name, hidden_size, freeze_bert = True):
        super(FineTunedModel, self).__init__()
        self.base_model = DistilBertModel.from_pretrained(model_name)
        #self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(768, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, output_size) #output_size ~=263`

        # Freeze the BERT model
        if freeze_bert:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attn_mask):
        #print(input_ids.shape, attn_mask.shape, labels.shape)
        outputs = self.base_model(input_ids, attention_mask=attn_mask)
        #last hidden state is tensor of shape batch_size x seq length x hidden size
        outputs = self.fc1(outputs[0][:, 0, :]) # returns last hidden layer
        # was using outputs.last_hidden_layer before but shape was eh
        outputs = F.relu(outputs)
        #outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        #outputs = F.softmax(outputs[0][:, 0, :])
        return outputs
##################################################################
class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out
##################################################################        