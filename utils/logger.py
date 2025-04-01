
import datetime
import os
import sys
import logging


def setup_logger(cfg):
    """
    Sets up the logging.
    """
    cfg.log_dir = 'logs/'+cfg.exp_name + '/' # cfg.DATASET + '/' 
    log_file_name = '{}{:s}-{date:%m-%d_%H-%M-%S}.log'.format(cfg.log_dir, cfg.MODE, date=datetime.datetime.now())
    
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
   
    # Set up logging format.
    FORMAT = "[%(levelname)s: %(filename)10s: %(lineno)4d]: %(message)s"

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO, format=FORMAT, stream=sys.stdout
    )
    logging.getLogger().addHandler(logging.FileHandler(log_file_name, mode='a'))


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def create_exp_name(cfg) -> str:
    """
    Create name of experiment using training parameters 
    """
    # splits = ''.join([split[0]+split[-1] for split in args.TRAIN_SUBSETS])
    name = '{:s}-e{:d}'.format(
        cfg.DATA.DATASET_NAME,
        cfg.TRAIN.NUM_EPOCH
        )

    # cfg.SAVE_ROOT += args.DATASET+'/'
    # args.SAVE_ROOT = args.SAVE_ROOT+'cache/'+args.exp_name+'/'
    # if not os.path.isdir(args.SAVE_ROOT):
    #     print('Create: ', args.SAVE_ROOT)
    #     os.makedirs(args.SAVE_ROOT)

    cfg.update(exp_name = name)
    setup_logger(cfg)

    return name
     
