import random
import numpy as np
import torch
import os
import os.path as osp
import shutil
import argparse
import time

from dolphin import utils
from dolphin.utils import Registers, import_all_modules_for_register, logger


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='PCL CV Library')
    parser.add_argument('--config', help='config path')
    parser.add_argument('--work_dir', help='work directory')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic', 
        action='store_true', 
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--resume_from', help='resume checkpoint path')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    parser.add_argument(
        '--gpu_ids', 
        type=int, 
        nargs='+', 
        help='ids of gpus to ues')
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()

    if args.config != '':
        cfg = utils.load_config(args.config)
    else:
        raise ValueError('Please assign the config path.')

    engine_cfg = cfg['engine']
    algorithm = cfg['algorithm']
    train_cfg = cfg['train_cfg']
    test_cfg = cfg['test_cfg']
    data = cfg['data']
    runtime_cfg = cfg['runtime']

    if runtime_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        runtime_cfg['work_dir'] = args.work_dir
    elif runtime_cfg.get('work_dir', None) is not None:
        runtime_cfg['work_dir'] = osp.join(
            './work_dir', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(runtime_cfg['work_dir'], mode=0o777, exist_ok=True)

    shutil.copy(args.config, runtime_cfg['work_dir'])

    logger_cfg = runtime_cfg['log']
    logger_cfg['filename'] = osp.join(
        runtime_cfg['work_dir'], logger_cfg['filename'])
    logger.init_logger(logger_cfg)
    log = logger.get(logger_cfg['logger_name'])

    log.info('PCL platform start ...')

    log.info('Loading all modules ...')
    import_all_modules_for_register(log)
    log.info('Modules loaded sucessfully.')

    if args.resume_from is not None:
        runtime_cfg['resume_from']['filename'] = args.resume_from
    if args.gpu_ids is not None:
        runtime_cfg['gpu_ids'] = args.gpu_ids
    else:
        if args.gpus is None:
            runtime_cfg['gpu_ids'] = range(1)
        else:
            runtime_cfg['gpu_ids'] = range(args.gpus)

    utils.mkdir_or_exist(osp.abspath(runtime_cfg['work_dir']))

    meta = dict()
    env_info_dict = utils.collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    log.info(
        'Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    if args.seed is not None:
        log.info(
            f'Set random seed to {args.seed}, '
            f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, args.deterministic)
    runtime_cfg['seed'] = args.seed
    meta['seed'] = args.seed

    engine_type = engine_cfg['type']
    engine = Registers.engine[engine_type](
        algorithm=algorithm,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        data=data,
        runtime_cfg=runtime_cfg,
        meta=meta)

    if runtime_cfg['resume_from']['filename']:
        engine.resume(**runtime_cfg['resume_from'])
    elif runtime_cfg['load_from']['filename']:
        engine.load_checkpoint(**runtime_cfg['load_from'])

    engine.run()

    log.info('Mission Done!')


if __name__ == '__main__':
    main()