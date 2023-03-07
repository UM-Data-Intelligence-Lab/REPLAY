import random
from enum import Enum
import torch
from torch.utils.data import Dataset


class Split(Enum):

    TRAIN = 0
    TEST = 1


class Usage(Enum):


    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2


class PoiDataset(Dataset):


    def reset(self):
        # reset training state:
        self.next_user_idx = 0  # current user index to add
        self.active_users = []  
        self.active_user_seq = []  
        self.user_permutation = [] 

        # set active users:
        for i in range(self.batch_size):  
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)  
            self.active_users.append(i)  
            self.active_user_seq.append(0)  

        for i in range(len(self.users)):
            self.user_permutation.append(i)  

    def shuffle_users(self):
        random.shuffle(self.user_permutation)  

        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)  
            self.active_users.append(self.user_permutation[i])  
            self.active_user_seq.append(0)

    def __init__(self, users, times, time_slots, coords, locs, sequence_length, batch_size, split, usage, loc_count,
                 custom_seq_count):
        self.users = users
        self.locs = locs
        self.times = times
        self.time_slots = time_slots
        self.coords = coords

        self.labels = [] 
        self.lbl_times = []  
        self.lbl_time_slots = []
        self.lbl_coords = []

        self.sequences = []  
        self.sequences_times = []
        self.sequences_time_slots = []
        self.sequences_coords = []

        self.sequences_labels = []
        self.sequences_lbl_times = []
        self.sequences_lbl_time_slots = []
        self.sequences_lbl_coords = []

        self.sequences_count = []  
        self.Ps = []  
        self.Qs = torch.zeros(loc_count, 1) 
        self.usage = usage  
        self.batch_size = batch_size  
        self.loc_count = loc_count
        self.custom_seq_count = custom_seq_count  

        self.reset()


        for i in range(loc_count):
            self.Qs[i, 0] = i


        for i, loc in enumerate(locs):
            self.locs[i] = loc[:-1]  
            self.labels.append(loc[1:]) 
            self.lbl_times.append(self.times[i][1:])
            self.lbl_time_slots.append(self.time_slots[i][1:])
            self.lbl_coords.append(self.coords[i][1:])

            self.times[i] = self.times[i][:-1]
            self.time_slots[i] = self.time_slots[i][:-1]
            self.coords[i] = self.coords[i][:-1]

        for i, (time, time_slot, coord, loc, label, lbl_time, lbl_time_slot, lbl_coord) in enumerate(
                zip(self.times, self.time_slots, self.coords, self.locs, self.labels, self.lbl_times,
                    self.lbl_time_slots, self.lbl_coords)):
            train_thr = int(len(loc) * 0.8)
            if split == Split.TRAIN:
                self.locs[i] = loc[:train_thr]
                self.times[i] = time[:train_thr]
                self.time_slots[i] = time_slot[:train_thr]
                self.coords[i] = coord[:train_thr]

                self.labels[i] = label[:train_thr]
                self.lbl_times[i] = lbl_time[:train_thr]
                self.lbl_time_slots[i] = lbl_time_slot[:train_thr]
                self.lbl_coords[i] = lbl_coord[:train_thr]

            if split == Split.TEST:
                self.locs[i] = loc[train_thr:]
                self.times[i] = time[train_thr:]
                self.time_slots[i] = time_slot[train_thr:]
                self.coords[i] = coord[train_thr:]

                self.labels[i] = label[train_thr:]
                self.lbl_times[i] = lbl_time[train_thr:]
                self.lbl_time_slots[i] = lbl_time_slot[train_thr:]
                self.lbl_coords[i] = lbl_coord[train_thr:]

        # split location and labels to sequences:
        self.max_seq_count = 0
        self.min_seq_count = 10000000
        self.capacity = 0
        for i, (time, time_slot, coord, loc, label, lbl_time, lbl_time_slot, lbl_coord) in enumerate(
                zip(self.times, self.time_slots, self.coords, self.locs, self.labels, self.lbl_times,
                    self.lbl_time_slots, self.lbl_coords)):
            seq_count = len(loc) // sequence_length 
            assert seq_count > 0, 'fix seq-length and min-checkins in order to have at least one test sequence in a 80/20 split!'
            seqs = []  
            seq_times = []
            seq_time_slots = []
            seq_coords = []

            seq_lbls = []
            seq_lbl_times = []
            seq_lbl_time_slots = []
            seq_lbl_coords = []

            for j in range(seq_count):
                start = j * sequence_length
                end = (j + 1) * sequence_length
                seqs.append(loc[start:end])
                seq_times.append(time[start:end])
                seq_time_slots.append(time_slot[start:end])
                seq_coords.append(coord[start:end])

                seq_lbls.append(label[start:end])
                seq_lbl_times.append(lbl_time[start:end])
                seq_lbl_time_slots.append((lbl_time_slot[start:end]))
                seq_lbl_coords.append(lbl_coord[start:end])

            self.sequences.append(seqs)
            self.sequences_times.append(seq_times)
            self.sequences_time_slots.append(seq_time_slots)
            self.sequences_coords.append(seq_coords)

            self.sequences_labels.append(seq_lbls)
            self.sequences_lbl_times.append(seq_lbl_times)
            self.sequences_lbl_time_slots.append(seq_lbl_time_slots)
            self.sequences_lbl_coords.append(seq_lbl_coords)

            self.sequences_count.append(seq_count)
            self.capacity += seq_count
            self.max_seq_count = max(self.max_seq_count, seq_count)
            self.min_seq_count = min(self.min_seq_count, seq_count)

        # statistics
        if self.usage == Usage.MIN_SEQ_LENGTH:
            print(split, 'load', len(users), 'users with min_seq_count', self.min_seq_count, 'batches:', self.__len__())
        if self.usage == Usage.MAX_SEQ_LENGTH:
            print(split, 'load', len(users), 'users with max_seq_count', self.max_seq_count, 'batches:', self.__len__())
        if self.usage == Usage.CUSTOM:
            print(split, 'load', len(users), 'users with custom_seq_count', self.custom_seq_count, 'Batches:',
                  self.__len__())

    def sequences_by_user(self, idx):
        return self.sequences[idx]

    def __len__(self):
        """ Amount of available batches to process each sequence at least once.
        """

        if self.usage == Usage.MIN_SEQ_LENGTH:
            # min times amount_of_user_batches:
            return self.min_seq_count * (len(self.users) // self.batch_size)
        if self.usage == Usage.MAX_SEQ_LENGTH:
            # estimated capacity:
            estimated = self.capacity // self.batch_size
            return max(self.max_seq_count, estimated)
        if self.usage == Usage.CUSTOM:
            return self.custom_seq_count * (len(self.users) // self.batch_size)
        raise ValueError()

    def __getitem__(self, idx):


        seqs = []
        times = []
        time_slots = []
        coords = []

        lbls = []
        lbl_times = []
        lbl_time_slots = []
        lbl_coords = []

        reset_h = []
        for i in range(self.batch_size):
            i_user = self.active_users[i]  
            j = self.active_user_seq[i]  
            max_j = self.sequences_count[i_user]  
            if self.usage == Usage.MIN_SEQ_LENGTH:
                max_j = self.min_seq_count
            if self.usage == Usage.CUSTOM:
                max_j = min(max_j, self.custom_seq_count)  
            if j >= max_j: 
               
                i_user = self.user_permutation[self.next_user_idx]  
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                while self.user_permutation[self.next_user_idx] in self.active_users:  
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # TODO: throw exception if wrapped around!
            # use this user:
            reset_h.append(j == 0)  
            seqs.append(torch.tensor(self.sequences[i_user][j]))
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            time_slots.append(torch.tensor(self.sequences_time_slots[i_user][j]))
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))

            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_time_slots.append(torch.tensor(self.sequences_lbl_time_slots[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))

            self.active_user_seq[i] += 1  
            
        x = torch.stack(seqs, dim=1)
        t = torch.stack(times, dim=1)
        t_slot = torch.stack(time_slots, dim=1)
        s = torch.stack(coords, dim=1)

        y = torch.stack(lbls, dim=1)
        y_t = torch.stack(lbl_times, dim=1)
        y_t_slot = torch.stack(lbl_time_slots, dim=1)
        y_s = torch.stack(lbl_coords, dim=1)
        return x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, torch.tensor(self.active_users)
