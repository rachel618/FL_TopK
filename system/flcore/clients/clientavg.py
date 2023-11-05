import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)   
      
        
    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        #
        initial_params = [param.data.clone() for param in self.model.parameters()]
        for _ in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
            
        param_diff = [cur_param.data.clone() - initial_param for cur_param, initial_param in 
                zip(self.model.parameters(), initial_params)] 
        topk = self.get_top_k(param_diff)
        return topk  
  
    def get_top_k(self, params):
        if self.topk_algo == "global":
            topk_ = self.global_topk(params)
        elif self.topk_algo == "chunk":
            topk_ = self.chunk_topk(params) 
        else:
            topk_ = params
        return topk_

    def chunk_topk(self, parameter):
        all_params = torch.cat([param.reshape(-1) for param in parameter])
        chunks = all_params.chunk(self.topk, dim=-1)
        for chunk in chunks:
            local_max_index = torch.abs(chunk.data).argmax().item()
            zeroed_out = set(range(len(chunk))) - set([local_max_index])
            chunk.data[list(zeroed_out)] = 0

        topk_chunk_grad_flattened = torch.cat([chunk for chunk in chunks])
    
        topk_chunk_grad = []
        start_idx = 0
        for grad in parameter:
            end_idx = start_idx + grad.numel()
            topk_chunk_grad.append(
                torch.Tensor(topk_chunk_grad_flattened[start_idx:end_idx]).view(
                    grad.data.shape
                )
            )
            start_idx = end_idx

        return topk_chunk_grad

    def global_topk(self, parameter):
        all_grads = torch.cat([param.reshape(-1) for param in parameter])
        top_k_gradients = all_grads.abs().topk(self.topk)
        mask = set(range(len(all_grads))) - set([top_k_gradients.indices])
        all_grads[list(mask)] = 0
        
        topk_global_grad = []
        start_idx = 0
        for grad in parameter:
            end_idx = start_idx + grad.numel()
            topk_global_grad.append(
                torch.Tensor(all_gradsd[start_idx:end_idx]).view(
                    grad.data.shape
                )
            )
            start_idx = end_idx

        return topk_global_grad


    def get_min_grad(self, gradient):
        all_grads = torch.cat([grad.reshape(-1) for grad in gradient])
        topk_grads = torch.abs(all_grads).topk(self.topk)[0]
        return torch.min(topk_grads).item()
