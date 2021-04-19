# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm

from .c2_model_loading import align_and_update_state_dicts

def convert_res18_keys(layer_name):
    for i in range(1,5):
        layer_name = layer_name.replace('layer{}'.format(i), 'res{}'.format(i+1))
    for i in range(1, 3):
        layer_name = layer_name.replace('bn{}'.format(i), 'conv{}.norm'.format(i))
    for i in range(2):
        layer_name = layer_name.replace('downsample.{}'.format(i), 'shortcut{}'.format('.norm' if i else ''))
    if layer_name.startswith('conv1'):
        layer_name = 'stem.' + layer_name
    layer_name = 'backbone.bottom_up.' + layer_name
    return layer_name

class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        if 'resnet18' in filename:
            keys_dict = {k:convert_res18_keys(k) for k in  loaded.keys()}  
            loaded = {keys_dict[k]:v for k,v in loaded.items()}      

        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)
        if incompatible is None:  # support older versions of fvcore
            return None

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible

