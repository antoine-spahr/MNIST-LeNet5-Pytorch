import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics
import time
import logging
import json

from src.utils.utils import print_progessbar

class LeNet5_trainer:
    """
    Trainer of LeNet5 archiecture.
    """
    def __init__(self, net, n_epoch=100, batch_size=128, num_workers=0, lr=1e-3,
                 lr_decay=0.95, device='cpu', optimizer=None, loss_fn=None,
                 seed=-1):
        """
        Built a LeNet5 trainer. It enables to train and evaluate a LeNet5 architecture.
        """
        # set seed for reproducibility
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.benchmark = False
            self.seed = seed
        # Trainig parameters
        self.device = device
        self.net = net.to(self.device)
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.lr_decay = lr_decay
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        else:
            raise ValueError('Only Adam and SGD are supported.')
        self.scheduler = optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda ep: self.lr_decay) # manage the change in learning rate
        self.loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn

        # Outputs
        self.train_acc = None
        self.test_acc = None
        self.test_pred = None
        self.train_time = None
        self.epoch_loss_list = []

    def train(self, train_dataset, test_dataset):
        """
        Train the LeNet5 on the provided dataset.
        """
        logger = logging.getLogger()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                            num_workers=self.num_workers)
        n_batch = train_loader.__len__()

        logger.info(f'>>> Start Training the LeNet5 with seed {self.seed}.')
        start_time = time.time()
        for epoch in range(self.n_epoch):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            # minibatch iteration
            for b, (train_input, train_label) in enumerate(train_loader):
                train_input = train_input.float().to(self.device)
                train_input.require_grad = True
                train_label = train_label.to(self.device)
                # update weight by backpropagation
                pred = self.net(train_input)
                loss = self.loss_fn(pred, train_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                print_progessbar(b, n_batch, Name='Train Batch', Size=40, erase=True)

            # evaluate on test set
            test_acc, _ = self.evaluate(test_dataset, last=False)
            # store epoch stat
            self.epoch_loss_list.append([epoch+1, epoch_loss / n_batch, test_acc])
            # print summary statistics
            logger.info(f'>>> | Epoch {epoch+1:03}/{self.n_epoch:03} '
                        f'| Loss {epoch_loss / n_batch:.7f} '
                        f'| Test Accuracy {test_acc:.3%} '
                        f'| Time {time.time() - epoch_start_time:.2f} [s] |')

            # update leanring rate
            self.scheduler.step()

        # Get results
        self.train_time = time.time() - start_time
        self.train_acc, _ = self.evaluate(train_dataset, last=False)

        logger.info(f'>>> Finished training of LeNet5')
        logger.info(f'>>> Train time {self.train_time:.0f} [s]')
        logger.info(f'>>> Train accuracy {self.train_acc:.3%}\n')

    def evaluate(self, dataset, last=True):
        """
        Evaluate the network with the porvided dataloader and return the accuracy score.
        """
        if last:
            logger = logging.getLogger()
            logger.info('>>> Start Evaluating the LeNet5.')

        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                            num_workers=self.num_workers)

        N = loader.__len__()
        with torch.no_grad():
            pred, label = [], []
            for b, (input_data, input_label) in enumerate(loader):
                input_data = input_data.float().to(self.device)
                input_label = input_label.to(self.device)
                # classify sample
                pred += self.net(input_data).argmax(dim=1).tolist()
                label += input_label.tolist()

                print_progessbar(b, N, Name='Evaluation Batch', Size=40, erase=True)

            acc = sklearn.metrics.accuracy_score(label, pred)

            if last:
                self.test_acc, self.test_pred = acc, pred
                logger.info(f'>>> Test accuracy {self.test_acc:.3%} \n')
            else:
                return acc, pred

    def save_model(self, export_path):
        """
        save the model at the given path.
        """
        torch.save({'net_dict': self.net.state_dict()}, export_path)

    def load_model(self, import_path, map_location='cuda'):
        """
        Load a model from the given file.
        """
        model = torch.load(import_path, map_location=map_location)
        self.net.load_state_dict(model['net_dict'])

    def save_results(self, export_path):
        """

        """
        results = {'train_time': self.train_time,
                   'loss': self.epoch_loss_list,
                   'train_acc': self.train_acc,
                   'test_acc': self.test_acc,
                   'test_pred': self.test_pred}

        with open(export_path, 'w') as fb:
            json.dump(results, fb)
