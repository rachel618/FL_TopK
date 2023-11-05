import time
from flcore.clients.clientlin import clientLIN
from flcore.servers.serverbase import Server
from threading import Thread
import wandb
import torch
from tqdm import tqdm

class FedLin(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # set logger
        config = {
            "optimizer": "topk SGD",
            "epochs": self.global_rounds,
            "batch size": self.batch_size,
            "lr": self.learning_rate,
            "top k method": self.topk_algo,
            "k": self.topk,
            "num clients": self.num_clients,
            "model": "mobilenet_v2",
            "dataset": self.dataset,
        }
        # logger = wandb.init(
        #     project="fl-smartnic", config=config, name="mobilenetv2_topk"
        # )
        # select slow clients
        # self.set_slow_clients()
        self.set_clients(clientLIN)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.uploaded_gradients = []
        
    def are_model_parameters_equal(self,params1,params2):
        if len(params1) != len(params2):
            print('different length')
            return False

        for param1, param2 in zip(params1, params2):
            if not torch.equal(param1.data, param2.data):
                print('different elements')
                return False

        return True

    def train(self):

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.uploaded_gradients = []
            

            for client in tqdm(self.selected_clients, desc = 'training clients...'):
                client.train()
                self.uploaded_gradients.append(client.send_gradients())

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            # self.aggregate_gradients(self.uploaded_gradients)
            
            self.send_models() # senf global model to local models
            self.update_clients()´
        
            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, test_acc, test_auc = self.evaluate()
                # time_cost = time.time() - s_t
                # wandb.log(
                #     {
                #         "train epochs": i + 1,
                #         "time cost for each epoch": self.Budget[-1],
                #         "averaged local train loss": train_loss,
                #         "global model test accuracy": test_acc,
                #         "global model test auc": test_auc,
                #     }
                # )

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientLIN)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
    def set_gradient(self):
        average_grads = [sum(element) / len(self.uploaded_gradients) for element in zip(*self.uploaded_gradients)]
        
    def update_clients(self):
        gradients = []
        for client in self.selected_clients:
            gradients.append(client.update_gradients())
    
    def global_topk(self, gradient):
        min_ = self.get_min_grad(gradient)

        for g in gradient:
            g.data = torch.where(torch.abs(g) >= min_, g, 0.0)

        return gradient
       