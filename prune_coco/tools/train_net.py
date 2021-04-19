#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.layers import FrozenBatchNorm2d

import pdb
import pickle

import lth 

from torchvision import models

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def count_zeros(model,conv_only=False):

    n_params = 0
    n_zeros = 0

    for name, param in model.named_parameters():
        # print(name, param.size())
        if conv_only and 'conv' not in name: 
            continue

        n_params += param.numel()
        n_zeros += torch.sum(param==0).item()
    print('Zero percentage: {}'.format(n_zeros/n_params))

def count_zeros_mask(mask):
    n_params = 0
    n_ones = 0
    for name in mask:
        n_params += mask[name].numel()
        n_ones += (mask[name].sum())
    print('Zero percentage: {}'.format(1 - n_ones/n_params))


def apply_mask(model,mask):
    new_state_dict = model.state_dict()
    for name in model.state_dict():
        if name in mask:
            new_state_dict[name] = model.state_dict()[name]*mask[name]
    model.load_state_dict(new_state_dict)
    
    print('applied mask')


def create_mask(model):
    mask = {}
    n_mask_dims = 0
    for name, param in model.named_parameters():
        if 'weight' in name and (('conv' in name) or ('fpn' in name) \
            or ('fcn' in name) or ('fc1' in name) or ('fc2') in name ):

            #if self.check_modules(name):
            print('Pruning: ',name)
            mask[name] = torch.ones_like(param)
            n_mask_dims += param.numel()

    return mask,n_mask_dims


def generate_new_mask_prune(model,mask,n_mask_dims,keep_percentage,only_backbone=False):
    new_mask = mask 
    mask_vec = torch.zeros(n_mask_dims).to(list(new_mask.values())[0])
    start_ind = 0
    for name, param in model.named_parameters():
        if name in new_mask:
            mask_vec[start_ind:start_ind+param.numel()] = torch.abs(param).reshape(-1)
            start_ind += param.numel()

    cur_keep_percentage = keep_percentage
    thresh = torch.topk(mask_vec,int(cur_keep_percentage*n_mask_dims), sorted=True)[0][-1]
    n_zeros = 0
    new_state_dict = model.state_dict()
    for name in model.state_dict():
        if only_backbone:
            if name in new_mask and 'bottom_up' in name:
                param = model.state_dict()[name]
                n_zeros += torch.sum(torch.abs(param)==0.0).item()
                mask_val = new_mask[name]
                mask_val[torch.abs(param)<thresh] = 0
                new_mask[name] = mask_val
                new_state_dict[name] = param*mask_val
        elif name in new_mask:
                param = model.state_dict()[name]
                n_zeros += torch.sum(torch.abs(param)==0.0).item()
                mask_val = new_mask[name]
                mask_val[torch.abs(param)<thresh] = 0
                new_mask[name] = mask_val
                new_state_dict[name] = param*mask_val

    mask = new_mask
    model.load_state_dict(new_state_dict)
    
    return mask

def get_binary_mask_task_transfer(state_dict):

    new_mask = {}
    for name in state_dict.keys():
        #if name in state_dict and  ('backbone' in name) and ('bottom_up' in name): 
        model_param = state_dict[name].clone()
        model_param[model_param!=0] = 1
        new_mask[name] = model_param.byte().float().cuda()

    return new_mask


def get_task_mask(og_mask,source_task_mask,source_task_name):

    if source_task_name =='det':
        flag_key = 'mask_head'
    elif source_task_name =='key':
        flag_key = 'keypoint_head'

    for name in source_task_mask:
        #if source_task_name =='det':
        if name in og_mask and 'box_predictor' not in name and flag_key not in name:
            og_mask[name] = source_task_mask[name]

    return og_mask

  
def get_binary_mask(model,mask):

    state_dict = model.state_dict()
    new_mask = {}
    for name in mask:
        if name in state_dict and  ('backbone' in name) and ('bottom_up' in name): 

            model_param = state_dict[name].clone()
            model_param[model_param!=0] = 1
            new_mask[name] = model_param.byte().float()

    return new_mask

def copy_bn(ticket_bn,model_bn):
    tick_state_dict = ticket_bn.state_dict()
    mod_state_dict = model_bn.state_dict()

    for key in mod_state_dict.keys():
        mod_state_dict[key] = tick_state_dict[key]

    return mod_state_dict    


def compare_bnorm(back_bnorm,tick_bnorm):
  
    back_norm_state_dict = back_bnorm.state_dict()
    tick_norm_state_dict = tick_bnorm.state_dict()

    for k in back_norm_state_dict.keys():
        if torch.all(tick_norm_state_dict[k] != back_norm_state_dict[k]):
            return False
    return True

def compare_block(back_block,ticket_block,imagenet_ticket_type,shortcut=True):


    for i in range(len(back_block)):

        print('set: ',i+1)

        assert torch.all(back_block[i].conv1.weight ==  ticket_block[i].conv1.weight)
        assert compare_bnorm(back_block[i].conv1.norm,ticket_block[i].bn1)

        assert torch.all(back_block[i].conv2.weight == ticket_block[i].conv2.weight)
        assert compare_bnorm(back_block[i].conv2.norm,ticket_block[i].bn2)

        if shortcut and i==0:
            #Shortcut only present in first sub-block!
            #Shortcut/downsample
            assert torch.all(back_block[i].shortcut.weight == ticket_block[i].downsample[0].weight)
            assert compare_bnorm(back_block[i].shortcut.norm,ticket_block[i].downsample[1])

        if imagenet_ticket_type=='res50':
            assert torch.all(back_block[i].conv3.weight == ticket_block[i].conv3.weight)
            assert compare_bnorm(back_block[i].conv3.norm,ticket_block[i].bn3)




    # #SET 1
    # #Conv1 and bnorm
    # assert torch.all(back_block[0].conv1.weight ==  ticket_block[0].conv1.weight)
    # assert compare_bnorm(back_block[0].conv1.norm,ticket_block[0].bn1)

    # #Conv2 and bnorm
    # assert torch.all(back_block[0].conv2.weight == ticket_block[0].conv2.weight)
    # assert compare_bnorm(back_block[0].conv2.norm,ticket_block[0].bn2)

    # if imagenet_ticket_type=='res50':
    #     assert torch.all(back_block[0].conv3.weight == ticket_block[0].conv3.weight)
    #     assert compare_bnorm(back_block[0].conv3.norm,ticket_block[0].bn3)


    # if shortcut:
    #     #Shortcut/downsample
    #     assert torch.all(back_block[0].shortcut.weight == ticket_block[0].downsample[0].weight)
    #     assert compare_bnorm(back_block[0].shortcut.norm,ticket_block[0].downsample[1])

    # #SET 2
    # #Conv1 and bnorm
    # assert torch.all(back_block[1].conv1.weight ==  ticket_block[1].conv1.weight)
    # assert compare_bnorm(back_block[1].conv1.norm,ticket_block[1].bn1)

    # #Conv2 and bnorm
    # assert torch.all(back_block[1].conv2.weight == ticket_block[1].conv2.weight)
    # assert compare_bnorm(back_block[1].conv2.norm,ticket_block[1].bn2)

    #breakpoint()
    # if imagenet_ticket_type =='res50':
    #     #Conv 3 and bnorm: set 2    
    #     assert torch.all(back_block[1].conv1.weight ==  ticket_block[0].conv1.weight)
    #     assert compare_bnorm(back_block[1].conv1.norm,ticket_block[0].bn1)


    print("TRUE: All layer transfer check complete")



def check_transfer(back,ticket_model,imagenet_ticket_type):

    mod_state_dict = back.state_dict()
    tick_state_dict = ticket_model.state_dict()

    mod_names = list(mod_state_dict.keys())

    tick_names = list(tick_state_dict.keys())
    tick_names = [x for x in tick_names if 'num_batches' not in x]
    tick_names = [x for x in tick_names if 'fc' not in x]

    #breakpoint()

    for i in range(len(tick_names)):
        mod_param = mod_state_dict[ mod_names[i]  ]
        tick_param = tick_state_dict[ tick_names[i] ]

        if 'res2' in mod_names[i]:
            break

        print(mod_names[i],tick_names[i],mod_param.shape,tick_param.shape)
        print(torch.all(mod_param==tick_param).item())

    if imagenet_ticket_type=='res18':
        compare_block(back.res2,ticket_model.layer1,imagenet_ticket_type,shortcut=False)
    else:
        compare_block(back.res2,ticket_model.layer1,imagenet_ticket_type)

    compare_block(back.res3,ticket_model.layer2,imagenet_ticket_type)
    compare_block(back.res4,ticket_model.layer3,imagenet_ticket_type)
    compare_block(back.res5,ticket_model.layer4,imagenet_ticket_type)


def replace_block(backbone_block,ticket_block,imagenet_ticket_type,shortcut=False):

    """ 
        Each block is indexed 0,1 and each has conv and bnorm. Need to replace those in backbone.
        The detectron2 backbone has relu. 
    """

    for i in range(len(backbone_block)):
        back = backbone_block[i]
        tick = ticket_block[i]

        tick_state_dict = tick.state_dict()
        back_state_dict = back.state_dict()

        print('Len: ',len(backbone_block))

        for i in range(len(backbone_block)):
            back = backbone_block[i]
            tick = ticket_block[i]

            if shortcut and i==0:
                #Downsample = shortcut.

                back_state_dict = back.shortcut.state_dict()
                tick_state_dict = tick.downsample[0].state_dict()

                #Copy cnn
                back_state_dict['weight'] = tick_state_dict['weight']
                back.shortcut.load_state_dict(back_state_dict)

                #copy bnorm
                back_bn_state_dict = back.shortcut.norm.state_dict()
                tick_bn_state_dict = tick.downsample[1].state_dict()

                for key in back_bn_state_dict:
                    back_bn_state_dict[key] = tick_bn_state_dict[key]

                back.shortcut.norm.load_state_dict(back_bn_state_dict)    


            #Set conv1 weights
            back_state_dict = back.conv1.state_dict()
            tick_state_dict = tick.conv1.state_dict()
            back_state_dict['weight'] = tick_state_dict['weight']
            back.conv1.load_state_dict(back_state_dict)
            #Set bn1
            #back.conv1.norm = FrozenBatchNorm2d.convert_frozen_batchnorm(tick.bn1)
            back_bn_state_dict = copy_bn(tick.bn1,back.conv1.norm)
            back.conv1.norm.load_state_dict(back_bn_state_dict)



            #Set conv2
            back_state_dict = back.conv2.state_dict()
            tick_state_dict = tick.conv2.state_dict()
            back_state_dict['weight'] = tick_state_dict['weight']
            back.conv2.load_state_dict(back_state_dict)
            #set bn2
            #back.conv2.norm = FrozenBatchNorm2d.convert_frozen_batchnorm(tick.bn2)
            
            back_bn_state_dict = copy_bn(tick.bn2,back.conv2.norm)
            back.conv2.norm.load_state_dict(back_bn_state_dict)


            if imagenet_ticket_type =='res50':
                #Additional conv  in each block.

                #Set conv3
                back_state_dict = back.conv3.state_dict()
                tick_state_dict = tick.conv3.state_dict()
                back_state_dict['weight'] = tick_state_dict['weight']
                back.conv3.load_state_dict(back_state_dict)
                #set bn3
                #back.conv3.norm = FrozenBatchNorm2d.convert_frozen_batchnorm(tick.bn3)

                back_bn_state_dict = copy_bn(tick.bn3,back.conv3.norm)
                back.conv3.norm.load_state_dict(back_bn_state_dict)

                #print("Still left for resnet 50")
                #breakpoint()


            backbone_block[i] = back


def transfer_ticket(model,imagenet_ticket,imagenet_ticket_type):


    """ 
        res18 backbone can be broken into:
        stem,res2,res3,res4,res5

        equivalent to layer1,2,3,4 in res block

    """

    if imagenet_ticket_type=='res18':
        ticket_model = models.resnet18()
    elif imagenet_ticket_type == 'res50':
        ticket_model = models.resnet50()

    ticket_model = torch.nn.DataParallel(ticket_model)
    ticket_model = ticket_model.cuda()
    state = torch.load(imagenet_ticket)
    ticket_model.load_state_dict(state['state_dict'])


    # with open('ticket_model.pkl','wb') as fout:
    #     pickle.dump(ticket_model.module,fout)        


    module_names = []
    for name,module in model.backbone.bottom_up.named_children():
        module_names.append(name)

    if module_names[0] =='stem':

        #Stem
        tick_state_dict = ticket_model.module.conv1.state_dict()
        mod_state_dict = model.backbone.bottom_up.stem.conv1.state_dict()
        mod_state_dict['weight'] = tick_state_dict['weight']

        #model.backbone.bottom_up.stem.conv1.load_state_dict(tick_state_dict)
        model.backbone.bottom_up.stem.conv1.load_state_dict(mod_state_dict)

        mod_bn_state_dict = copy_bn(ticket_model.module.bn1,model.backbone.bottom_up.stem.conv1.norm)
        model.backbone.bottom_up.stem.conv1.norm.load_state_dict(mod_bn_state_dict)
        

        #detectron2 backbone has relu in forward 
        # Line 88 out = F.relu_(out) detectron2/modelling/backbone/resnet.py


    #Replace res2
    if imagenet_ticket_type == 'res18':
        replace_block(model.backbone.bottom_up.res2,ticket_model.module.layer1,imagenet_ticket_type,shortcut=False)
    else:
        replace_block(model.backbone.bottom_up.res2,ticket_model.module.layer1,imagenet_ticket_type,shortcut=True)
    
    #In these blocks we need to replace the original downsample layers with "shortcut" in backbone
    #replace res3
    replace_block(model.backbone.bottom_up.res3,ticket_model.module.layer2,imagenet_ticket_type,shortcut=True)
    #replace res4
    replace_block(model.backbone.bottom_up.res4,ticket_model.module.layer3,imagenet_ticket_type,shortcut=True)
    #replace res5
    replace_block(model.backbone.bottom_up.res5,ticket_model.module.layer4,imagenet_ticket_type,shortcut=True)

    #Create mask only for res-18 backbone.
    og_mask,n_mask_dims = create_mask(model)


    mask = {}
    for k in og_mask.keys():        
        if 'backbone' and 'bottom_up' in k:
            mask[k] = og_mask[k]

    new_mask = get_binary_mask(model,mask)
    #new_mask = generate_new_mask_prune(model,og_mask,n_mask_dims,0.2)

    count_zeros(ticket_model.module, conv_only=True)
    n_zeros = 0
    n_params = 0
    n_zeros_mask = 0
    n_params_mask = 0
    total_params = 0
    total_zeros = 0
    for name, param in model.state_dict().items():
        if name in new_mask:
            n_zeros += torch.sum(param==0).item()
            n_zeros_mask += torch.sum(new_mask[name]==0).item()
            n_params += param.numel()
            n_params_mask += new_mask[name].numel()
            print(name)
        total_params += param.numel()
        total_zeros += torch.sum(param==0).item()

    print(f'Mask: {n_zeros_mask/n_params_mask} Total: {total_zeros/total_params} Params: {n_zeros/n_params}')

    #breakpoint()

    check_transfer(model.backbone.bottom_up,ticket_model.module,imagenet_ticket_type)    

    return new_mask

def input_constructor(inp):
    data = pickle.load(open('resources/sample_data.pkl','rb'))
    return data

def get_macs(trainer):

    print('Loading sample data to measure FLOPS')
    data = pickle.load(open('resources/sample_data.pkl','rb'))

    import detectron2.utils.analysis as anal
    flops = anal.flop_count_operators(trainer.model,[data[0]])

    from thop import profile,clever_format
    macs, params = profile(trainer.model, inputs=(data, ))
    print(macs,params)
    macs, params = clever_format([macs, params], "%.3f")

    from ptflops import get_model_complexity_info
    macs_2, params_2 = get_model_complexity_info(trainer.model, (3, 768, 1024), as_strings=True,print_per_layer_stat=True, \
        verbose=True,input_constructor=input_constructor)
    #breakpoint()
    print('ptflops results: ')
    print(macs_2,params_2)    

    print("detectron2 results: ")
    print(flops)

    print("thop results:")
    print(macs,params,'\n')


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """

    """
    mask transfer

    """
    trainer = Trainer(cfg)

    #This resumes weights.
    #If resume=True, it loads weights from cfg.model.weights
    trainer.resume_or_load(resume=args.resume)

    if cfg['MODE'] == 'lottery':
        late_reset_ckpt = torch.load(cfg['LATE_RESET_CKPT'])

        #gets the basic mask structure
        og_mask,n_mask_dims = create_mask(trainer.model.module)
        #prune and generate new mask
        print('generating new mask by pruning')
        new_mask = generate_new_mask_prune(trainer.model.module,og_mask,n_mask_dims,cfg['LOTTERY_KEEP_PERCENTAGE'])

        #Load late_reset_dict
        print("Loading ",cfg['LATE_RESET_CKPT'],' late reset to model as initial\n')
        trainer.model.module.load_state_dict(late_reset_ckpt['model'])

        #Apply mask.
        apply_mask(trainer.model.module,new_mask)    

        lth_pruner = lth.lth(trainer.model,keep_percentage=cfg['LOTTERY_KEEP_PERCENTAGE'],\
            n_rounds=cfg['NUM_ROUNDS'])
        lth_pruner.init_state_dict = late_reset_ckpt['model']
        lth_pruner.init_opt_state_dict = late_reset_ckpt['optimizer']
        lth_pruner.mask = new_mask

    elif cfg['MODE'] =='transfer_imagenet':

        before_grad = {}
        for name, param in trainer.model.named_parameters():
            before_grad[name] = param.requires_grad

        #This modifes trainer model module.
        new_mask = transfer_ticket(trainer.model.module,cfg['IMAGENET_TICKET'],\
           cfg['IMAGENET_TICKET_TYPE'])

        #We have to ensure the same stuff is frozen before and after transfer
        after_grad = {}
        #setting default grad.
        for name, param in trainer.model.named_parameters():
            param.requires_grad = before_grad[name]
            after_grad[name] = param.requires_grad

        lth_pruner = lth.lth(trainer.model,keep_percentage=cfg['LOTTERY_KEEP_PERCENTAGE'],\
            n_rounds=cfg['NUM_ROUNDS'])
        lth_pruner.init_state_dict = trainer.model.state_dict()
        lth_pruner.init_opt_state_dict = None
        lth_pruner.mask = new_mask

    elif cfg['MODE'] == 'mask_transfer':
        og_mask,n_mask_dims = create_mask(trainer.model.module)    
        
        print('generating new mask by pruning only for backbone')
        new_mask = generate_new_mask_prune(trainer.model.module,og_mask,\
            n_mask_dims,cfg['LOTTERY_KEEP_PERCENTAGE'],only_backbone=True)

        before_grad = {}
        for name, param in trainer.model.named_parameters():
            before_grad[name] = param.requires_grad

        #Apply the mask.
        apply_mask(trainer.model.module,new_mask)

        after_grad = {}
        #setting default grad.
        for name, param in trainer.model.named_parameters():
            param.requires_grad = before_grad[name]
            after_grad[name] = param.requires_grad


        lth_pruner = lth.lth(trainer.model,keep_percentage=cfg['LOTTERY_KEEP_PERCENTAGE'],\
            n_rounds=cfg['NUM_ROUNDS'])

        lth_pruner.init_state_dict = trainer.model.state_dict()
        lth_pruner.init_opt_state_dict = None
        lth_pruner.mask = new_mask

    elif cfg['MODE'] == 'task_transfer':
        before_grad = {}
        for name, param in trainer.model.named_parameters():
            before_grad[name] = param.requires_grad

        source_model_state = torch.load(cfg['SOURCE_MODEL'])['model']
        source_task_name = cfg['SOURCE_TASK']

        #get complete_mask
        source_task_mask = get_binary_mask_task_transfer(source_model_state)

        #Get mask Structure.
        og_mask,n_mask_dims = create_mask(trainer.model.module)    
        
        new_mask = get_task_mask(og_mask,source_task_mask,source_task_name)
        apply_mask(trainer.model.module,new_mask)

        print('deleting model to save mem\n')
        del source_model_state

        after_grad = {}
        for name, param in trainer.model.named_parameters():
            param.requires_grad = before_grad[name]
            after_grad[name] = param.requires_grad

        lth_pruner = lth.lth(trainer.model,keep_percentage=cfg['LOTTERY_KEEP_PERCENTAGE'],\
            n_rounds=cfg['NUM_ROUNDS'])
        lth_pruner.init_state_dict = trainer.model.state_dict()
        lth_pruner.init_opt_state_dict = None
        lth_pruner.mask = new_mask

    elif cfg['MODE'] =='prune_backbone':
        late_reset_ckpt = torch.load(cfg['LATE_RESET_CKPT'])
        og_mask,n_mask_dims = create_mask(trainer.model.module.backbone.bottom_up)

        print('generating new mask by pruning')
        new_mask = generate_new_mask_prune(trainer.model.module.backbone.bottom_up,og_mask,n_mask_dims,cfg['LOTTERY_KEEP_PERCENTAGE'])

        #Load late_reset_dict
        print("Loading ",cfg['LATE_RESET_CKPT'],' late reset to model as initial\n')
        trainer.model.module.load_state_dict(late_reset_ckpt['model'])

        #Apply mask.
        apply_mask(trainer.model.module.backbone.bottom_up,new_mask)    

        lth_pruner = lth.lth(trainer.model,keep_percentage=cfg['LOTTERY_KEEP_PERCENTAGE'],\
            n_rounds=cfg['NUM_ROUNDS'])

        lth_pruner.init_state_dict = late_reset_ckpt['model']
        lth_pruner.init_opt_state_dict = late_reset_ckpt['optimizer']

        lth_pruner.mask = new_mask

        #TODO: Add Prune backbone and imagenet transfer.



    #*************************************************************************#
        #Some printing of zero count and param count (module and layerwise)
    #*************************************************************************#
    num_zero_per,_ = lth_pruner.count_zeros(trainer.model)
    print('total: ',num_zero_per)

    total_params = 0
    module_wise_params = {}

    print("\nMODULE WISE param count")
    for name,module in trainer.model.module.named_children():
        num_zero_per,total_params = lth_pruner.count_zeros(module,layer_wise=False)
        print('module name: ',name,' total zeros: ',num_zero_per,' total params: ',total_params)
        temp = 0
        for p in module.parameters():
            temp += torch.numel(p)
        total_params += temp
        module_wise_params[name] = temp   
    print('\n\n')  
    for mod in module_wise_params:
        print(mod,module_wise_params[mod],' percentage of total: ',\
            module_wise_params[mod]/total_params)
    print('\n\n')

    #*********************************************************************#

    # print("Transferred ticket part: zeros ")
    # count_zeros(trainer.model.module.backbone.bottom_up,conv_only=True)
    # print('\n')

    #*********************************************************************#    

    torch.save({'model': trainer.model.state_dict()}, cfg['OUTPUT_DIR'] + '/model_0000000.pth')    

    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    trainer.lth_pruner = lth_pruner

    if cfg.TEST.COMPUTE_FLOPS:
        print("Get MACS")
        get_macs(trainer)


    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
