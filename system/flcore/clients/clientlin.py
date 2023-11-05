import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientLIN(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.init
    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_epochs = self.local_epochs

        for _ in range(max_local_epochs):
            initial_gradients = [param.grad.data for param in self.model.parameters()]
            for i, (x, y) in enumerate(trainloader):

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # if self.train_slow:
                #     time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                # self.optimizer.zero_grad()
                loss.backward()

                for cur_param, initial_param, global_param in zip(self.model.parameters(), initial_gradients, self.global_model.parameters()):
                    cur_param.grad.data -= (initial_param.data - global_param.grad.data.clone())

                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    def send_topk_gradients(self):
        uploaded_grads = [param.grad.data.clone() for param in self.model.paramters()]
        return uploaded_grads
    

        