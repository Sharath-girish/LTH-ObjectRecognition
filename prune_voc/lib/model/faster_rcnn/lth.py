import torch
import numpy as np

class lth:
    def __init__(self, model, keep_percentage, n_rounds, late_reset_iter, module_list):
        self.model = model
        self.keep_percentage = keep_percentage
        self.n_rounds = n_rounds
        self.late_reset_iter = late_reset_iter
        self.thresh = 0.0
        self.n_zeros = 0
        self.module_list = module_list
        self.create_mask(self.model)
        print(self.get_lth_stats())
        self.init_state_dict = None
        self.init_opt_state_dict = None

    def check_modules(self, name):
        # (base_conv|top_conv|rpn_conv|top_fc|rpn_fc|downsample|bn)
        condition = False
        checklist = [['base','conv','weight'],\
                     ['top','conv','weight'],\
                     ['RPN','Conv','weight'],\
                     ['RCNN','cls','weight'],\
                     ['RCNN','bbox','weight'],\
                     ['RPN','cls','weight'],\
                     ['RPN','bbox','weight'],\
                     ['downsample','weight'],\
                     ['bn','weight']]
        map_dict = {0:[0],1:[1],2:[2],3:[3,4],4:[5,6],5:[7],6:[8]}
        for i in range(7):
            if self.module_list[i] == '0':
                continue
            for j in map_dict[i]:
                condition = condition or np.bitwise_and.reduce(np.array([check in name for check in checklist[j]]))
        return condition

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
            if self.check_modules(name):
                self.mask[name] = torch.ones_like(param)
                self.n_mask_dims += param.numel()

    def generate_new_mask_layerwise(self,model,pruning_round):
        new_mask = self.mask 

        cur_keep_percentage = self.keep_percentage**(pruning_round/(self.n_rounds-1))
        self.n_zeros = 0
        new_state_dict = model.state_dict()
        self.thresh = {} 

        for name in model.state_dict():
            if name in new_mask:
                param = model.state_dict()[name]
                self.thresh[name] = torch.topk(torch.abs(param).reshape(-1),int(cur_keep_percentage*param.numel()), sorted=True)[0][-1]
                mask_val = new_mask[name]
                mask_val[torch.abs(param)<self.thresh[name]] = 0
                new_mask[name] = mask_val
                new_state_dict[name] = param*mask_val                

        self.mask = new_mask
        model.load_state_dict(new_state_dict)
        self.model = model
        return model
        
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
        if not isinstance(self.thresh, dict):
            print_str = f'Percentile threshold: {self.thresh:.4f}, Number of model weights: {n_params:1.2E}, percentage pruned: {n_zeros*100.0/n_params:.4f}% '
        else:
            print_str = f'Number of model weights: {n_params:1.2E}, percentage pruned: {n_zeros*100.0/n_params:.4f}% '
        return print_str
