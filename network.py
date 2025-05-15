import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
from math import pi

class Rnn(Enum):
    ''' The available RNN units '''
    
    RNN = 0
    GRU = 1    
    LSTM = 2    
    
    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM        
        raise ValueError('{} not supported in --rnn'.format(name))        

class RnnFactory():
    ''' Creates the desired RNN unit. '''
    
    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)
                
    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'        
    
    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]
        
    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)

        
        
class User_Week_Distribution(nn.Module):
    def __init__(self,stamp_num):
        super().__init__()
        self.stamp_num=stamp_num
        self.sigma=nn.Parameter(torch.ones(self.stamp_num).view(self.stamp_num,1))

    def forward(self,x):
        self.sigma.data=torch.abs(self.sigma.data)
        learned_weight=1/torch.sqrt(2*pi*(self.sigma**2))*torch.exp(-(x**2)/(2*(self.sigma**2)))
        sum=torch.sum(learned_weight,dim=1,keepdim=True)
        return learned_weight/sum

    
    
class REPLAY(nn.Module):
 
    
    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory, week, day, week_weight_index, day_weight_index):
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t # function for computing temporal weight
        self.f_s = f_s # function for computing spatial weight
        self.week_matrix=week #168 *168 

        self.week_weight_index=week_weight_index


        
        self.encoder = nn.Embedding(input_size, hidden_size) # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size) # user embedding
        self.week_encoder=nn.Embedding(24*7,hidden_size//2)

        self.rnn = rnn_factory.create(hidden_size)
        self.fc = nn.Linear(3*hidden_size-hidden_size//2, input_size) # create outputs in lenght of locations
        self.fcpt= nn.Linear(2*hidden_size-hidden_size//2, hidden_size)
        self.week_distribution=User_Week_Distribution(168)


    def forward(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_user):        
        seq_len, user_len = x.size()
        
        week_weight=self.week_distribution(self.week_weight_index).view(168,168)
        # week_weight = torch.where(week_weight < 0.001, 0, week_weight).view(1,user_len,168,1)
        # day_weight=self.day_distribution(self.day_weight_index).view(24,24)
        # day_weight=torch.where(day_weight < 0.01, 0, day_weight).view(1,user_len,24,1)
        # assert not torch.isinf(day_weight).any()
        # assert not torch.isnan(day_weight).any()
        # t_day1=t_slot%24
        # t_day2=y_t_slot%24

        new_week_weight1=week_weight.index_select(0,t_slot.view(-1)).view(seq_len,user_len,168,1)
        new_week_weight2=week_weight.index_select(0,y_t_slot.view(-1)).view(seq_len,user_len,168,1)


        w_t1=self.week_matrix.index_select(0,t_slot.view(-1)).view(seq_len,user_len,-1)
        w_t1=self.week_encoder(w_t1).permute(0,1,3,2)#seq*batch_size*5*168

        w_t1=torch.matmul(w_t1,new_week_weight1).squeeze()

        t_emb1 = w_t1

        w_t2=self.week_matrix.index_select(0,y_t_slot.view(-1)).view(seq_len,user_len,-1)

        w_t2=self.week_encoder(w_t2).permute(0,1,3,2)#seq*batch_size*5*168

        w_t2=torch.matmul(w_t2,new_week_weight2).squeeze()

        t_emb2 = w_t2


        x_emb = self.encoder(x)        
        poi_time=self.fcpt(torch.cat((x_emb,t_emb1),dim=-1))
        out, h = self.rnn(poi_time, h)
        # comopute weights per user
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)
            for j in range(i+1):
                dist_t = t[i] - t[j]
                dist_s = torch.norm(s[i] - s[j], dim=-1)
                a_j = self.f_t(dist_t, user_len)
                b_j = self.f_s(dist_s, user_len)
                a_j = a_j.unsqueeze(1)
                b_j = b_j.unsqueeze(1)
                w_j = a_j*b_j + 1e-10 # small epsilon to avoid 0 division
                sum_w += w_j
                out_w[i] += w_j*out[j]
            # normalize according to weights
            out_w[i] /= sum_w
        
        # add user embedding:
        p_u = self.user_encoder(active_user)
        p_u = p_u.view(user_len, self.hidden_size)
        out_pu = torch.zeros(seq_len, user_len, 2*self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)
        out_pu=torch.cat((out_pu,t_emb2),dim=-1)
        y_linear = self.fc(out_pu)
        return y_linear, h


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:        
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))        
    else:        
        return FixNoiseStrategy(hidden_size)

class H0Strategy():
    
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
    
    def on_init(self, user_len, device):
        pass
    
    def on_reset(self, user):
        pass
    
    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization. '''
    
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1/self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu
    
    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)
    
    def on_reset(self, user):
        return self.h0

class LstmStrategy(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''
    
    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy
    
    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h,c)
    
    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h,c)
