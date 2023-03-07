import torch
from torch.utils.data import DataLoader
import numpy as np
import time

from setting import Setting
from trainer import ReplayTrainer
from dataloader import PoiDataloader
from dataset import Split
from network import create_h0_strategy
from evaluation import Evaluation


setting = Setting()
setting.parse()
print(setting)

poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
poi_loader.read(setting.dataset_file,setting.offset_file)
dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(), 'batch size must be lower than the amount of available users'

trainer = ReplayTrainer(setting.lambda_t, setting.lambda_s)
h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory, setting.device)
evaluation_test = Evaluation(dataset_test, dataloader_test, poi_loader.user_count(), h0_strategy, trainer, setting)
print('{} {}'.format(trainer, setting.rnn_factory))

optimizer = torch.optim.Adam(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.2)

for e in range(setting.epochs):
    h = h0_strategy.on_init(setting.batch_size, setting.device)    
    dataset.shuffle_users() 
    
    losses = []
    epoch_start = time.time()
    for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(dataloader):
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])
        
        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        t_slot = t_slot.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)

        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_t_slot = y_t_slot.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)
        active_users = active_users.to(setting.device)
        
        optimizer.zero_grad()
        loss = trainer.loss(x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users)
        loss.backward(retain_graph=True)
        losses.append(loss.item())
        optimizer.step()
    
    scheduler.step()
    epoch_end = time.time()
    print('One training need {:.2f}s'.format(epoch_end - epoch_start))  

    if (e+1) % 1 == 0:
        epoch_loss = np.mean(losses)
        print(f'Epoch: {e+1}/{setting.epochs}')
        print(f'Used learning rate: {scheduler.get_last_lr()[0]}')
        print(f'Avg Loss: {epoch_loss}')


    if (e+1) % setting.validate_epoch == 0:        
        print(f'~~~ Test Set Evaluation (Epoch: {e+1}) ~~~')
        evaluation_test.evaluate()
    
