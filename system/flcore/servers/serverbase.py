import torch
import os
import numpy as np
import h5py
import copy
import time

from sklearn.preprocessing import label_binarize
from sklearn import metrics

from utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

        self.topk = self.args.topk
        self.topk_algo = self.args.topk_algo
        self.test_loader = self.set_test_data()
        self.prev_epoch_global_param = []

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(
            range(self.num_clients), self.train_slow_clients, self.send_slow_clients
        ):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)

            client = clientObj(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_slow=train_slow,
                send_slow=send_slow,
            )
            self.clients.append(client)

    def set_test_data(self):
        test_data = []
        for i in range(self.num_clients):
            test_data.extend(read_client_data(self.dataset, i, is_train=False))

        return DataLoader(test_data, self.batch_size, drop_last=False, shuffle=True)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(
            np.random.choice(self.clients, self.current_num_join_clients, replace=False)
        )

        return selected_clients

    def send_models(self):
        assert len(self.clients) > 0

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0

        # active_clients = random.sample(
        #     self.selected_clients,
        #     int((1 - self.client_drop_rate) * self.current_num_join_clients),
        # )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"]
                    / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"]
                    / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def aggregate_gradients(self):
        self.global_model = copy.deepcopy(self.uploaded_models[0])

        for new_param, orig_param in zip(
            self.global_model.parameters(), self.uploaded_models[0].parameters()
        ):
            new_param.grad = orig_param.grad.clone()

        client_grads = []
        for client in self.uploaded_models:
            # grad = [param.grad.clone() for param in client.parameters()]
            grad = [param.grad.clone() for param in client.parameters()]
            client_grads.append(grad)

        topk_grads = self.get_top_k(client_grads, self.topk_algo)
        average_grads = [sum(element) / len(topk_grads) for element in zip(*topk_grads)]

        for idx, param in enumerate(self.global_model.parameters()):
            if param.grad == None:
                param.grad.data = torch.zeros_like(param.data)
            param.grad.data = average_grads[idx]

        optimizer = torch.optim.SGD(
            self.global_model.parameters(), lr=self.args.local_learning_rate
        )
        optimizer.step()

    def aggregate_param_diff(self):
        client_updated_params = []

        for client in self.uploaded_models:
            param = [
                client_param.data.clone() - global_param.data.clone()
                for client_param, global_param in zip(
                    client.parameters(), self.global_model.parameters()
                )
            ]
            client_updated_params.append(param)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        # for new_param, origin in zip(self.global_model.parameters(), self.prev_epoch_global_param):
        #     new_param.data = origin

        topk_updated_params = self.get_top_k(client_updated_params, self.topk_algo)
        for param_topk, w in zip(topk_updated_params, self.uploaded_weights):
            for server_param, client_param_diff in zip(
                self.global_model.parameters(), param_topk
            ):
                server_param.data += client_param_diff.data.clone() * w

        self.prev_epoch_global_param = [
            param.data.clone() for param in self.global_model.parameters()
        ]

    def get_top_k(self, aggregated_clients, topk_algo):
        if topk_algo == "global":
            topk_ = [self.global_topk(client) for client in aggregated_clients]
        elif topk_algo == "chunk":
            topk_ = [self.chunk_topk(client) for client in aggregated_clients]
        else:
            topk_ = aggregated_clients

        return topk_

    def chunk_topk(self, gradient):
        all_grads = torch.cat([grad.reshape(-1) for grad in gradient])
        chunks = all_grads.chunk(self.topk, dim=-1)
        for chunk in chunks:
            local_max_index = torch.abs(chunk.data).argmax().item()
            zeroed_out = set(range(len(chunk))) - set([local_max_index])
            chunk.data[list(zeroed_out)] = 0

        topk_chunk_grad_flattened = torch.cat([chunk for chunk in chunks])

        topk_chunk_grad = []
        start_idx = 0
        for grad in gradient:
            end_idx = start_idx + grad.data.numel()
            topk_chunk_grad.append(
                torch.Tensor(topk_chunk_grad_flattened[start_idx:end_idx]).view(
                    grad.data.shape
                )
            )
            start_idx = end_idx

        return topk_chunk_grad

    def global_topk(self, gradient):
        min_ = self.get_min_grad(gradient)

        for g in gradient:
            g.data = torch.where(torch.abs(g) >= min_, g, 0.0)

        return gradient

    def get_min_grad(self, gradient):
        all_grads = torch.cat([grad.reshape(-1) for grad in gradient])
        topk_grads = torch.abs(all_grads).topk(self.topk)[0]
        return torch.min(topk_grads).item()

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(
            self.global_model.parameters(), client_model.parameters()
        ):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert os.path.exists(model_path)
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("rs_test_acc", data=self.rs_test_acc)
                hf.create_dataset("rs_test_auc", data=self.rs_test_auc)
                hf.create_dataset("rs_train_loss", data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(
            item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt")
        )

    def load_item(self, item_name):
        return torch.load(
            os.path.join(self.save_folder_name, "server_" + item_name + ".pt")
        )

    def test_metrics(self):
        testloaderfull = self.test_loader

        self.global_model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

        return test_acc, test_num, auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []

        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        test_acc, test_num, test_auc = self.test_metrics()
        stats_train = self.train_metrics()
        test_acc /= test_num

        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))

        return train_loss, test_acc, test_auc

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = (
                    len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0]
                    > top_cnt
                )
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = (
                    len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0]
                    > top_cnt
                )
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(
                self.global_model.parameters(), client_model.parameters()
            ):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))

        if cnt > 0:
            print("PSNR value is {:.2f} dB".format(psnr_val / cnt))
        else:
            print("PSNR error")

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_slow=False,
                send_slow=False,
            )
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
