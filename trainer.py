import torch
import torch.nn as nn
import numpy as np
from network import REPLAY

class ReplayTrainer():

    
    def __init__(self, lambda_t, lambda_s):

        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
    
    def __str__(self):
        return 'Use REPLAY training.'        
    
    def parameters(self):   
        return self.model.parameters()

    def generate_tensor_of_distribution(self,time):
        list1=[]
        temp=[i for i in range(time)]
        for i in range(time):
            if i == time//2:
                list1.append(temp)
            elif i<time//2:
                list1.append(temp[-(time//2-i):]+temp[:-(time//2-i)])
            else :
                list1.append(temp[(i-time//2):]+temp[:(i-time//2)])
        return torch.tensor(list1)

    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device):
        f_t = lambda delta_t, user_len: ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*self.lambda_t)) # hover cosine + exp decay
        f_s = lambda delta_s, user_len: torch.exp(-(delta_s*self.lambda_s)) # exp decay   
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.week=self.generate_tensor_of_distribution(168).to(device)
        self.day=self.generate_tensor_of_distribution(24).to(device)
        self.week_weight_index=torch.tensor([x-84 for x in range(168)]).repeat(168,1).to(device)
        self.day_weight_index=torch.tensor([x-12 for x in range(24)]).repeat(24,1).to(device)
        self.model = REPLAY(loc_count, user_count, hidden_size, f_t, f_s, gru_factory,self.week,self.day,self.week_weight_index,self.day_weight_index).to(device)
    
    def evaluate(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users):

        
        self.model.eval()
        out, h = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users)
        out_t = out.transpose(0, 1)
        return out_t, h 
    
    def loss(self, x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users):


        self.model.train()
        out, h= self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users)  
        out = out.view(-1, self.loc_count)  
        y = y.view(-1)  
        l = self.cross_entropy_loss(out, y)
        return l
