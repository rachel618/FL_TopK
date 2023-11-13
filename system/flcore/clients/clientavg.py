import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
import copy


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs

        initial_params = {
            name: params.data.clone() for name, params in self.model.named_parameters()
        }

        for _ in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

        updated_param = self.subtract_params(initial_params)
        top_k_ = self.get_top_k(updated_param)  # current - initial 의 top k

        return top_k_

    def subtract_params(self, prev_params):
        current_params = {
            name: params.data.clone() for name, params in self.model.named_parameters()
        }
        assert prev_params.keys() == current_params.keys()

        for name in prev_params.keys():
            prev_params[name] = current_params[name] - prev_params[name]

        return prev_params

    def get_top_k(self, params):
        top_k = params
        if self.topk_algo == "global":
            top_k = self.global_topk(params)
        elif self.topk_algo == "chunk":
            top_k = self.chunk_topk(params)
        return top_k

    def chunk_topk(self, params):
        all_params = torch.cat([param.data.reshape(-1) for param in params.values()])
        chunks = all_params.chunk(self.topk, dim=-1)
        for chunk in chunks:
            local_max_index = torch.abs(chunk.data).argmax().item()
            zeroed_out = set(range(len(chunk))) - set([local_max_index])
            chunk.data[list(zeroed_out)] = 0

        top_k = torch.cat([chunk for chunk in chunks])
        start_idx = 0
        top_k_params = {}
        for name, param in self.model.named_parameters():
            end_idx = start_idx + param.data.numel()
            top_k_params[name] = top_k[start_idx:end_idx].view(param.data.shape)
            start_idx = end_idx
        return top_k_params

    def global_topk(self, params):
        all_params = torch.cat([param.data.reshape(-1) for param in params.values()])
        top_k = all_params.abs().topk(self.topk)
        mask = set(range(len(all_params))) - set(top_k.indices.tolist())

        all_params[list(mask)] = 0

        start_idx = 0
        top_k_params = {}

        for name, param in self.model.named_parameters():
            end_idx = start_idx + param.data.numel()
            top_k_params[name] = all_params[start_idx:end_idx].view(param.data.shape)
            start_idx = end_idx

        return top_k_params

    def get_min_of_topk(self, params):
        all_params = torch.cat([param.reshape(-1) for param in params])
        topk_params = torch.abs(all_params).topk(self.topk)[0]
        return torch.min(topk_params).item()
