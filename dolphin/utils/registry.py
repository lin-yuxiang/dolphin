import importlib
import os
import copy
import sys
import warnings
import torch.nn as nn

from dolphin.utils import merge_subcfg, is_list_of
# from dolphin.utils import merge_subcfg


class Register:

    def __init__(self, name):
        self._dict = {}
        self._name = name

    def __len__(self):
        return len(self._dict)
    
    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            raise KeyError(f'Module not found: {e}')

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(name={self._name}, ' + f'item={self._dict})'
        return format_str

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception('Value of a Registry must be callable.')
        if key is None:
            key = value.__name__
        if key in self._dict:
            # warnings.warn(f'{key} is already registered '
                            # f'in {self._name}')
            pass
        self._dict[key] = value
    
    def keys(self):
        return self._dict.keys()

    def register(self, target):
        def add(key, value):
            self[key] = value
            return value
        
        if callable(target):
            return add(None, target)
        return lambda x: add(target, x)
    

class Registers:

    def __init__(self):
        raise RuntimeError('Registers is not intended to be instantiated.')

    algorithm = Register('algorithm')
    engine = Register('engine')

    backbone = Register('backbone')
    neck = Register('neck')
    head = Register('head')
    decoder = Register('decoder')
    loss = Register('loss')
    data = Register('data')
    pipeline = Register('pipeline')
    strategy = Register('strategy')
    generator = Register('generator')
    discriminator = Register('discriminator')
    filter = Register('filter')

    roi_head = Register('roi_head')
    roi_extractor = Register('roi_extractor')
    dense_head = Register('dense_head')
    bbox_assigner = Register('bbox_assigner')
    bbox_sampler = Register('bbox_sampler')
    bbox_head = Register('bbox_head')
    bbox_coder = Register('bbox_coder')


def build_module_from_registers(cfg, module_name=None, sub_cfg=None):
    if cfg is None:
        return None
    if module_name is None:
        key = list(cfg.keys())
        assert len(key) == 1
        module_name = key[0]
        module_cfg = copy.deepcopy(cfg[module_name])
    else:
        module_cfg = copy.deepcopy(cfg)
    if not hasattr(Registers, module_name):
        # TODO: the module_name must be inside the Registers!!!
        if module_name == 'train_cfg' or module_name == 'test_cfg':
            return module_cfg
        else:
            raise ValueError(f'No module {module_name} found in Registers.')
    if is_list_of(module_cfg, dict):
        module = nn.ModuleList()
        for item in module_cfg:
            register = getattr(Registers, module_name)
            if item.get('type') is not None:
                module_type = item.pop('type')
            else:
                raise ValueError(f'{item} should include key "type".')
            if sub_cfg is not None:
                item = merge_subcfg(item, sub_cfg)
            module.append(register[module_type](**item))
    elif isinstance(module_cfg, dict):
        register = getattr(Registers, module_name)
        if module_cfg.get('type') is not None:
            module_type = module_cfg.pop('type')
        else:
            raise ValueError(f'{module_cfg} should include key "type".')
        if sub_cfg is not None:
            module_cfg = merge_subcfg(module_cfg, sub_cfg)
        module = register[module_type](**module_cfg)
    else:
        raise ValueError(
        f'"module_cfg" must be type of "list of dict" or "dict"')

    return module


def _handle_errors(errors, log):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        log.warning("Module {} import failed: {}".format(name, err))
    raise RuntimeError("Please check these modules.")


def path_to_module_format(py_path):
    return py_path.replace("/", ".").rstrip(".py")


def add_custom_modules(all_modules, config=None):
    current_work_dir = os.getcwd()
    if current_work_dir not in sys.path:
        sys.path.append(current_work_dir)
    if config is not None and "custom_modules" in config:
        custom_modules = config["custom_modules"]
        if not isinstance(custom_modules, list):
            custom_modules = [custom_modules]
        all_modules += [
            ("", [path_to_module_format(module)]) for module in custom_modules
        ]


def import_all_modules_for_register(log, config=None):
    """Import all modules for register"""
    all_modules = ALL_MODULES

    add_custom_modules(all_modules, config)

    log.debug(f"All modules: {all_modules}")
    errors = []
    for base_dir, modules in all_modules:
        for name in modules:
            try:
                if base_dir != "":
                    full_name = base_dir + "." + name
                else:
                    full_name = name
                importlib.import_module(full_name)
                log.debug(f'{full_name} loaded.')
            except ImportError as error:
                errors.append((name, error))
    _handle_errors(errors, log)

# algorithms
ALGORITHM_MODULES = ['multi_target_tracking.fairmot', 'gan.cyclegan', 'depth.fcrn', 'activate_learning.activate_learning']

# model_modules
BACKBONE_MODULES = ['backbone.dla_seg', 'backbone.resnet', 'backbone.activate_learning_backbone']
DECODER_MODULES = ['decoder.fcrn_up_proj']
GENERATOR_MODULES = ['generator.unet']
DISCRIMINATOR_MODULES = ['discriminator.nlayer_discriminator']
HEAD_MODULES = ['head.mot.fairmot_det_head', 'head.mot.fairmot_reid_head', 'head.mot.fairmot_share_head', 'head.depth.fcrn_head', 'head.activate_learning.activate_learning_head']
QUERY_MODULES = ['query_strategy.entropy_sampling']
FILTER_MODULES = ['filter.kalman_filter']
MODEL_MODULES = BACKBONE_MODULES + DECODER_MODULES + HEAD_MODULES + GENERATOR_MODULES + DISCRIMINATOR_MODULES + QUERY_MODULES + FILTER_MODULES

# datasets
DATASET_MODULES = ['mot.joint_dataset', 'gan.unaligned_dataset', 'depth.nyu_depth_dataset', 'activate_learning.mnist_dataset']
PIPELINE_MODULES = ['augmentation', 'compose', 'formating', 'loading']

# losses
LOSS_MODULES = ['classification_loss', 'regression_loss', 'gan_loss']

# engines
ENGINE_MODULES = ['mot.mot_engine', 'gan.gan_engine', 'activate_learning.activate_learning_engine', 'depth.depth_engine']


ALL_MODULES = [('models.modules', MODEL_MODULES),
               ('models.algorithms', ALGORITHM_MODULES),
               ('dataset.datasets', DATASET_MODULES),
               ('loss', LOSS_MODULES),
               ('engine', ENGINE_MODULES),
               ('dataset.pipeline', PIPELINE_MODULES)]
