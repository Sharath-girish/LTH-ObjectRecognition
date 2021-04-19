import glob,os 
import torch
import pickle
import argparse


def count_zeros(state,path):
	n_zeros = 0
	n_params = 0

	for name in state:
		n_zeros += len(state[name][state[name]==0])
		n_params += state[name].numel()

	print(path,' Zero percentage : ',n_zeros/n_params)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str,default='',help='path to folder to check')
parser.add_argument("--all",action='store_true',help='check all ')
args = parser.parse_args()


states_path = glob.glob(args.folder+'/*.pth')

path = args.folder+'model_final.pth'


if args.all:
	for path in states_path:
		state = torch.load(path)
		state = state['model']
		count_zeros(state,path)
		
elif path in states_path:
	state = torch.load(path)['model']
	count_zeros(state,path)
else:
	for path in states_path:
		state = torch.load(path)
		state = state['model']
		count_zeros(state,path)


