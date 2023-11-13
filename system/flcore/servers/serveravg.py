import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import wandb
import torch
from tqdm import tqdm


class FedAvg(Server):
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
        logger = wandb.init(
            project="fl-smartnic", config=config, name="mobilenetv2_topk"
        )
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.uploaded_params = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.clients
            self.send_models()  # send global model to local models

            if i % self.eval_gap == 0 and i > 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, test_acc, test_auc = self.evaluate()

                wandb.log(
                    {
                        "train epochs": i + 1,
                        "time cost for each epoch": self.Budget[-1],
                        "averaged local train loss": train_loss,
                        "global model test accuracy": test_acc,
                        "global model test auc": test_auc,
                    }
                )
                
            self.global_model.train()
            self.uploaded_params = []
            for client in tqdm(self.selected_clients, desc="training clients..."):
                uploded_params = client.train()
                self.uploaded_params.append(uploded_params)

            self.receive_models()
            self.aggregate_parameter_diff(self.uploaded_params)
            self.save_global_model()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
