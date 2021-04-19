import torch
import numpy as np

class lth:
    def __init__(self, model, keep_percentage, n_rounds, late_reset_iter):
        self.model = model
        self.keep_percentage = keep_percentage
        self.n_rounds = n_rounds
        self.late_reset_iter = late_reset_iter
        self.thresh = 0.0
        self.n_zeros = 0
        self.create_mask(self.model)
        print(self.get_lth_stats())
        self.init_state_dict = None
        self.init_opt_state_dict = None

    def apply_mask(self, model):
        new_state_dict = model.state_dict()
        for name in model.state_dict():
            if name in self.mask:
                new_state_dict[name] = model.state_dict()[name]*self.mask[name]
        model.load_state_dict(new_state_dict)
        self.model = model
        return model

    def create_mask(self, model):
        self.mask = {}
        self.n_mask_dims = 0
        for name, param in model.named_parameters():
            if 'weight' in name and 'conv' in name:
                self.mask[name] = torch.ones_like(param)
                self.n_mask_dims += param.numel()


    def generate_new_mask(self, model, pruning_round):
        new_mask = self.mask 
        mask_vec = torch.zeros(self.n_mask_dims).to(list(new_mask.values())[0])
        start_ind = 0
        for name, param in model.named_parameters():
            if name in new_mask:
                mask_vec[start_ind:start_ind+param.numel()] = torch.abs(param).reshape(-1)
                start_ind += param.numel()
        cur_keep_percentage = self.keep_percentage**(pruning_round/(self.n_rounds-1)) #todo
        self.thresh = torch.topk(mask_vec,int(cur_keep_percentage*self.n_mask_dims), sorted=True)[0][-1]
        self.n_zeros = 0
        new_state_dict = model.state_dict()
        for name in model.state_dict():
            if name in new_mask:
                param = model.state_dict()[name]
                self.n_zeros += torch.sum(torch.abs(param)==0.0).item()
                mask_val = new_mask[name]
                mask_val[torch.abs(param)<self.thresh] = 0
                new_mask[name] = mask_val
                new_state_dict[name] = param*mask_val

        self.mask = new_mask
        model.load_state_dict(new_state_dict)
        self.model = model
        return model


    def get_lth_stats(self):
        n_params = 0
        n_zeros = 0
        for name, param in self.model.named_parameters():
            if name in self.mask:
                n_params += param.numel()
                n_zeros += torch.sum(torch.abs(param)==0.0).item()
        print_str = f'Percentile threshold: {self.thresh:.4f}, Number of model weights: {n_params:1.2E}'+\
                    f', percentage pruned: {n_zeros*100.0/n_params:.4f}% '
        return print_str
