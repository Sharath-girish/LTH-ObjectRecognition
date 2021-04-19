import pickle
import torch 


def load(filename):
	with open(filename,'rb') as fout:
		model = pickle.load(fout)
	return model 	

def get_param_dict(model):
	d = {}
	for name,param in model.named_parameters():
		d[name] = param 
	return d 

ticket_model = load('check/ticket_model.pkl')
ticket_dict = get_param_dict(ticket_model)

before_model = load('check/before_model.pkl')
before_dict = get_param_dict(before_model)

after_model = load('check/after_model.pkl')
after_dict = get_param_dict(after_model)


#Is it because of batchnorm??

for t in after_dict.keys():
	#if t in before_dict.keys():
	print(before_dict[t].requires_grad,after_dict[t].requires_grad)

