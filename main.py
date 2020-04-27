import torch
import torch.cuda
import numpy as np
#import torchvision
import random
from datetime import datetime
import ast
import os
import logging
import click

from src.network.LeNet5 import LeNet5
from src.optim.LeNet5_trainer import LeNet5_trainer
from src.dataset.load_dataset import load_dataset
from src.utils.Config import Config

################################################################################
#                                   Settings                                   #
################################################################################

@click.command()
@click.argument('dataset_name', type=click.Choice(['MNIST', 'FashionMNIST', 'KMNIST']))
@click.argument('net_name', type=click.Choice(['LeNet5']))
@click.option('--exp_folder', type=str, default='Outputs/',
              help='Where to export outputs. Default: Outputs/')
@click.option('--data_path', type=str, default='data/',
              help='Where to find the data. Default: data/')
@click.option('--load_config', type=click.Path(exists=True),
              default=None, help='Config JSON-file path. Default: None')
@click.option('--load_model', type=click.Path(exists=True),
              default=None, help='Model pt-file path. Default: None')
@click.option('--batch_size', type=int, default=128,
              help='The size of batch to use. Default: 128')
@click.option('--num_workers', type=int, default=0,
              help='The number of CPU worker to use. Default: 0')
@click.option('--n_epochs', type=int, default=100,
              help='The number of epoch to train on. Default: 100')
@click.option('--lr', type=float, default=1e-3,
              help='The learning rate to use. Default: 1e-3')
@click.option('--lr_decay', type=float, default=0.95,
              help='The learning rate decay at each epoch. Default:0.95')
@click.option('--device', type=str, default='cuda',
              help='The device to train on. Default: cuda')
@click.option('--optimizer_name', type=click.Choice(['Adam', 'SGD']), default='Adam',
              help='The optimization method. Default: Adam')
@click.option('--seeds', type=str, default='[-1]',
              help='List of seeds. Perform as many training as seeds. Default: [-1]')
def main(**params):
    """

    """
    # make output dir
    OUTPUT_PATH = params['exp_folder'] + params['dataset_name'] + '_' + \
                  params['net_name'] + '_' + datetime.today().strftime('%Y_%m_%d_%Hh%M')+'/'
    if not os.path.isdir(OUTPUT_PATH+'model/'): os.makedirs(OUTPUT_PATH+'model/', exist_ok=True)
    if not os.path.isdir(OUTPUT_PATH+'results/'): os.makedirs(OUTPUT_PATH+'results/', exist_ok=True)

    # create the config file
    cfg = Config(params)

    # set up the logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    log_file = OUTPUT_PATH + 'LOG.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f'Export path : {OUTPUT_PATH}')

    # Load config if required
    if params['load_config']:
        cfg.load_config(params['load_config'])
        logger.info(f'Config loaded from {params["load_config"]}')

    if not torch.cuda.is_available():
        cfg.settings['device'] = 'cpu'

    logger.info('Config Parameters:')
    for key, value in cfg.settings.items():
        logger.info(f'|---- {key} : {value}')

    #loop over seeds:
    train_acc_list, test_acc_list = [], []
    seeds = ast.literal_eval(cfg.settings['seeds'])

    for i, seed in enumerate(seeds):
        logger.info('-'*25 + f' Training n°{i+1} ' + '-'*25)
        # set the seed
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(f'Set seed {i+1:02}/{len(seeds):02} to {seed}')

        # get dataset
        train_dataset = load_dataset(cfg.settings['dataset_name'],
                                     cfg.settings['data_path'], train=True,
                                     data_augmentation = True)

        test_dataset = load_dataset(cfg.settings['dataset_name'],
                                    cfg.settings['data_path'], train=False,
                                    data_augmentation = False)
        # get model
        net = LeNet5()
        LeNet = LeNet5_trainer(net, n_epoch=cfg.settings['n_epochs'], batch_size=cfg.settings['batch_size'],
                               num_workers=cfg.settings['num_workers'], lr=cfg.settings['lr'],
                               lr_decay=cfg.settings['lr_decay'], device=cfg.settings['device'],
                               optimizer=cfg.settings['optimizer_name'], seed=seed)

        # Load model if required
        if cfg.settings['load_model']:
            LeNet.load_model(cfg.settings['load_model'])
            logger.info(f'Model loaded from {cfg.settings["load_model"]}')

        # train model
        LeNet.train(train_dataset, test_dataset)
        LeNet.evaluate(test_dataset, last=True)

        # Save model and results
        LeNet.save_model(OUTPUT_PATH+f'model/model_{i+1}.pt')
        logger.info('Model saved at ' + OUTPUT_PATH+f'model/model_{i+1}.pt')
        LeNet.save_results(OUTPUT_PATH+f'results/results_{i+1}.json')
        logger.info('Results saved at ' + OUTPUT_PATH+f'results/results_{i+1}.json')
        cfg.save_config(OUTPUT_PATH + 'config.json')
        logger.info('Config saved at ' + OUTPUT_PATH+'config.json')

        train_acc_list.append(LeNet.train_acc)
        test_acc_list.append(LeNet.test_acc)

        # show results


    train_acc, test_acc = np.array(train_acc_list), np.array(test_acc_list)
    logger.info('\n'+'-'*60)
    logger.info(f"Performance of {cfg.settings['net_name']} on {cfg.settings['dataset_name']} over {len(seeds)} replicates")
    logger.info(f"|---- Train accuracy {train_acc.mean():.3%} +/- {1.96*train_acc.std():.3%}")
    logger.info(f"|---- Test accuracy {test_acc.mean():.3%} +/- {1.96*test_acc.std():.3%}")

if __name__ == '__main__':
    main()
