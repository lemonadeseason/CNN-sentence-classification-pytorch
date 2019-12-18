"""不太清楚 conv1d输入输出的格式和所代表含义https://blog.csdn.net/sunny_xsc1994/article/details/82969867"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, **kwargs):   #dict
        super(CNN, self).__init__()

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

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)    
            """一维卷积，步长现在就相当于是1"""
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')
    """单个channel时
     (50,max_len)->embedding,view->(50,1,max_len*embedding)->conv1d (num 100,size 300*3)->(50,100,max_len-kernel_size+1)->maxpool1d(stride=max_len-kernel_size    +1)->(50,100,1)->view(50,100)->all kernel size  cat->(50,300)->fc(300,6)->(50,6)"""
    """两个channel
      经过embedding(50,2,max_len*embedding)->conv2d(2,100,300*3)->(50,200,...)->pool(50,200,1)->view->(50,200)"""
    def forward(self, inp):
        #print(inp.size())    #(batch_size*max_sent_len)
        #print(self.embedding(inp).size())   #(...*dim)
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)    #view是改变tensor的大小（batch_size,1,max_sent_len*dim）
        """transform from 2d to 1d，which is match with conv1d"""
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)
            #print(x.size())    #(50,2,max_sent_len*dim)
        
        print(x.size())    #50,1,300*37
        print(self.get_conv(0)(x).size())    #50,100,35=37-3+1
        print(F.max_pool1d(self.get_conv(0)(x),self.MAX_SENT_LEN - self.FILTERS[0] + 1).size())   #50,100,1
        
        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])   #batch*100,原先三维中最后一维是1个数，去掉那个维度了
            for i in range(len(self.FILTERS))]
        x = torch.cat(conv_results, 1)    #每一个kernel_size都是100列，3个拼起来，一个样本是300列的特征
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)
        #print(x.size())
        return x
