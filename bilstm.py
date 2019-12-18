"""不知道lstm问题出在哪里了，acc一直很低，不论rand、static..."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BILSTM(nn.Module):
    def __init__(self, **kwargs):   #dict
        super(BILSTM, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)  #遇到padding_index时，输出0向量
        if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
            elif self.MODEL == "multichannel":
                self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2   #有两个输入通道，static和非static

        self.lstm = nn.LSTM(self.WORD_DIM,50,2,bidirectional=True,batch_first=True,dropout=self.DROPOUT_PROB)
        #self.dropout = nn.Dropout(self.DROPOUT_PROB)
        self.fc = nn.Linear(100, self.CLASS_SIZE)

   
    def forward(self, inp):
        #print(inp.size())    #(batch_size*max_sent_len)
        #print(self.embedding(inp).size())   #(...*dim)
        x = self.embedding(inp)#.view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)    #view是改变tensor的大小（batch_size,1,max_sent_len*dim）
        """transform from 2d to 1d，which is match with conv1d"""
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)
            #print(x.size())    #(50,2,max_sent_len*dim)
        
       
        #h0=torch.zeros(4,x.size(0),50)
        #c0=torch.zeros(4,x.size(0),50)
        h0 = c0 = Variable(torch.zeros(4,x.size(0),50), requires_grad=False)
        output,_=self.lstm(x,(h0,c0))
        output = F.dropout(output, p=self.DROPOUT_PROB, training=self.training)
        output = output[:,-1,:].view(-1,100)
        x = self.fc(output)
        #print(x.size())
        return x
