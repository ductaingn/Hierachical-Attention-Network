import torch
import torch.nn as nn

class WordLevel(nn.Module):
    def __init__(self,input_size, hidden_gru_size):
        super(WordLevel,self).__init__()
        self.WordEncoder = nn.GRU(input_size,hidden_gru_size,num_layers=1,bidirectional=True)
        self.Alpha = nn.Sequential(
            nn.Linear(hidden_gru_size*2,hidden_gru_size*2),
            nn.Tanh(),
            nn.Linear(hidden_gru_size*2,hidden_gru_size*2),
            nn.Softmax()
        )
    
    # Input x is a word
    def forward(self,x):
        h_it,hidden = self.WordEncoder(x)
        a_it = self.Alpha(h_it)
        s_i = (h_it*a_it).sum(dim=0)
        return s_i

class SentenceLevel(nn.Module):
    def __init__(self,input_size, hidden_gru_size):
        super(SentenceLevel,self).__init__()
        self.SentenceEncoder = nn.GRU(input_size,hidden_gru_size,num_layers=1,bidirectional=True)
        self.Alpha = nn.Sequential(
            nn.Linear(hidden_gru_size*2,hidden_gru_size*2),
            nn.Tanh(),
            nn.Linear(hidden_gru_size*2,hidden_gru_size*2),
            nn.Softmax()
        )

    # Input x is a sentence (list of words) 
    def forward(self,x):
        h_it,hidden = self.SentenceEncoder(x)
        a_it = self.Alpha(h_it)
        v = (h_it*a_it).sum(dim=0)
        return v

class HierachicalAttentionNetwork(nn.Module):
    def __init__(self,word_emmbed_dim, word_gru_size, sentence_gru_size,output_dim):
        super(HierachicalAttentionNetwork,self).__init__()
        self.word_level = WordLevel(word_emmbed_dim,word_gru_size)
        self.sentence_level = SentenceLevel(word_gru_size*2,sentence_gru_size)
        self.classification = nn.Sequential(
            nn.Linear(sentence_gru_size*2,output_dim),
            nn.Softmax()
        )

    # Input x is an text (list of sentences)
    def forward(self,x):
        s = []
        for i in range(len(x)):
            s.append(self.word_level.forward(torch.FloatTensor(x[i])))
        s = torch.stack(s,dim=0) # Turn list of encoded sentence tensor into a tensor
        v = self.sentence_level.forward(s)
        p = self.classification(v)

        return p
