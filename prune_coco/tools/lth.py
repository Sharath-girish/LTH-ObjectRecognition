import torch
import numpy as np

import pdb #debugger


class lth:
	def __init__(self, model, keep_percentage, n_rounds, late_reset_iter=None, module_list=None):
		self.model = model
		self.keep_percentage = keep_percentage
		self.n_rounds = n_rounds
		self.late_reset_iter = late_reset_iter
		self.thresh = 0.0
		self.n_zeros = 0
		self.module_list = module_list
		self.create_mask(self.model)
		#print(self.get_lth_stats())
		self.init_state_dict = None
		self.init_opt_state_dict = None
		
		#For debug purposes.
		self.model_path = ''


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
			if 'weight' in name and (('conv' in name) or ('fpn' in name) \
				or ('fcn' in name) or ('fc1' in name) or ('fc2') in name ):

				#if self.check_modules(name):
				print('Pruning: ',name)
				self.mask[name] = torch.ones_like(param)
				self.n_mask_dims += param.numel()


	def generate_layer_wise_mask(self,model,pruning_round):
		new_mask = self.mask 

		cur_keep_percentage = self.keep_percentage**(pruning_round/(self.n_rounds-1))
		self.n_zeros = 0
		new_state_dict = model.state_dict()

		for name in model.state_dict():
				if name in new_mask:
					param = model.state_dict()[name]
					tensor = param.cpu().detach().numpy()
					alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
					if len(alive)==0:
						#kill all
						new_mask[name] = 0
					else:
						percentile_value = np.percentile(abs(alive), (1-cur_keep_percentage)*100 )
						weight_dev = param.device
						temp = np.where(abs(tensor) < percentile_value, 0, new_mask[name])		
						# Apply new weight and mask
						param.data = torch.from_numpy(tensor * temp).to(weight_dev)
						temp = torch.tensor(temp)
						new_mask[name] = temp 

					#Apply mask to state dict
					new_state_dict[name] = param*new_mask[name]
				

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
		# print(self.mask.keys())
		for name, param in self.model.named_parameters():
			if name in self.mask:
				# print(name)
				n_params += param.numel()
				n_zeros += torch.sum(torch.abs(param)==0.0).item()
		print_str = f'Percentile threshold: {self.thresh:.4f}, Number of model weights: {n_params:1.2E}, percentage pruned: {n_zeros*100.0/n_params:.4f}% '

		n_zeros_2 = 0
		for name, param in self.mask.items():
			n_zeros_2 += torch.sum(param==0).item()
		print_str += f' percentage pruned (mask): {n_zeros_2*100.0/n_params:.4f}% '
		return print_str


	def count_zeros(self,model,layer_wise=True):
		"""
			The goal is number of zeros should match in each layer!.
		"""
		
		# state_dict = model.state_dict()

		# z = 0
		# total_weights = 0
		# for k in state_dict.keys():
		# 	#print("Layer: ",k,"  ")
		# 	#if 'weight' in k and 'conv' in k:
		# 	w = state_dict[k]
		# 	total_weights += w.nelement()
		# 	z = z + len(w[w==0])
		# 	if layer_wise:
		# 		print("Layer: ",k,"  ",len(w[w==0])/w.nelement(),w.nelement())

		# #print('\n\n')
		# #print('Total weights: ')
		# #print(z,'/',total_weights,' = ',z/total_weights)
		# num_zero_percentage = z/total_weights
		# return num_zero_percentage,total_weights

		n_params = 0
		n_zeros = 0

		for name, param in model.named_parameters():
			#print(name)
			n_params += param.numel()
			n_zeros += torch.sum(param==0).item()
		print('Zero percentage: {}'.format(n_zeros/n_params))
		num_zero_percentage = n_zeros / n_params
		return num_zero_percentage,n_params

if __name__ == '__main__':

	import torchvision.models as m 

	model = m.resnet18()
	lt = lth(model,0.1,2,1,[1,3,4]) #Keeping only 10%
	pruned_model = lt.generate_layer_wise_mask(model,1)

	print("should see SAME number of zeros each layer")
	lt.count_zeros(pruned_model)
	print('\n\n')

	model_new = m.resnet18()
	lt_new = lth(model_new,0.1,2,1,[1,3,4]) #Keeping only 10%
	pruned_model_new = lt_new.generate_new_mask(model_new,1)

	print("should see different number of zeros each layer")
	lt_new.count_zeros(pruned_model_new)

