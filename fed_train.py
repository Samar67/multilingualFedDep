from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import os
from datetime import datetime
import warnings

import json
import numpy as np
import pandas as pd

import flwr as fl
import torch

from torch.utils.data import DataLoader
import evaluate

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import logging


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


from fed_train_config import _C as cnfg
from fed_data_read import *

def configure():
    cfg = "fed_train_config.yaml"
    cnfg.defrost()
    cnfg.merge_from_file(cfg)
    cnfg.freeze()

def tokenization(tweets, model_name):
    Max_Len  = 150   #According to what is returned from the below code
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_feature = tokenizer.batch_encode_plus(
                            tweets, 
                            add_special_tokens = True,
                            padding = 'max_length',
                            truncation=True,
                            max_length = Max_Len, 
                            #is_split_into_words=True  
                            return_attention_mask = True,
                            return_tensors = 'pt'       
                   )
    # tokenized_feature_raw = tokenizer.batch_encode_plus(
    #                         tweets.astype(str).values.tolist(), 
    #                         add_special_tokens = True#,
                                
    #                )
    # token_sentence_length = [len(x) for x in tokenized_feature_raw['input_ids']]
    # print('max: ', max(token_sentence_length))
    # print('min: ', min(token_sentence_length))

    # plt.figure(figsize=(20, 8))
    # plt.hist(token_sentence_length, rwidth = 0.9)
    # plt.xlabel('Sequence Length', fontsize = 18)
    # plt.ylabel('# of Samples', fontsize = 18)
    # plt.xticks(fontsize = 14)
    # plt.yticks(fontsize = 14)

    return tokenized_feature

def prep_dataloader(tokens, labels):
    data = TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.tensor(labels))
    sampler = RandomSampler(data)
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, sampler= sampler)#, shuffle=True)
    return data_loader

def train(net, trainloader, epochs):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    losss = 0
    optimizer = AdamW(net.parameters(), lr=LR)
    net.train()
    for _ in range(epochs):
        for step, batch in enumerate(trainloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            outputs = net( b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            
            losss += outputs.loss.item()
            preds = torch.argmax(outputs.logits.to(DEVICE), dim=1)
            clf_metrics.add_batch(predictions=preds, references=b_labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    losss /= len(trainloader.dataset)
    mets = clf_metrics.compute()
    return losss, mets["accuracy"], mets["precision"], mets["recall"], mets["f1"]

def test(net, testloader):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    loss = 0
    net.eval()
    for step, batch in enumerate(testloader):
        batch = tuple(t.to(DEVICE) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = net(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        clf_metrics.add_batch(predictions=predictions, references=b_labels)

    loss /= len(testloader.dataset)
    mets = clf_metrics.compute()
    return loss, mets["accuracy"], mets["precision"], mets["recall"], mets["f1"]

def get_parameters(model) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def save_model(net, server_round):
    state = {'model': net.state_dict(),
             'round': server_round}
    torch.save(state, os.path.join(STORE_DIR, f'train-{server_round}.pth'))
    
class DepClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")

        self.set_parameters(parameters)
        print("Training Started...")
        loss, accuracy, precision, recall, f1 = train(self.net, self.trainloader, epochs=local_epochs)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.trainloader), {"accuracy": float(accuracy), "loss": float(loss), 
                                                   "precision": float(precision), "recall": float(recall), 
                                                   "f1": float(f1), "client fit": float(self.cid)}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.set_parameters(parameters)
        loss, accuracy, precision, recall, f1 = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "loss": float(loss), 
                                                   "precision": float(precision), "recall": float(recall), 
                                                   "f1": float(f1), "client evaluate": float(self.cid)}
    
def client_fn(cid):
    return DepClient(cid, net, trainloaders[int(cid)], val_loaders[int(cid)])

def weighted_fit_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    percs = [num_examples * m["precision"] for num_examples, m in metrics]
    recs = [num_examples * m["recall"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    num = sum(examples)
    acc = sum(accuracies) / num
    loss = sum(losses) / num
    perc =  sum(percs) / num
    rec = sum(recs) / num
    f1 = sum(f1s) / num

    RESULTS['agg_tr_accuracy_loss_precesion_recall_f1'].append([acc, loss, perc, rec, f1])

    return {"accuracy": acc, "loss": loss, 
            "precision": perc, "recall": rec,
            "f1": f1, "which?":  "fit"}

def weighted_eval_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    percs = [num_examples * m["precision"] for num_examples, m in metrics]
    recs = [num_examples * m["recall"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    num = sum(examples)
    acc = sum(accuracies) / num
    loss = sum(losses) / num
    perc =  sum(percs) / num
    rec = sum(recs) / num
    f1 = sum(f1s) / num

    RESULTS['agg_val_accuracy_loss_precesion_recall_f1'].append([acc, loss, perc, rec, f1])

    return {"accuracy": acc, "loss": loss, 
            "precision": perc, "recall": rec,
            "f1": f1, "which?":  "eval"}

def centralized_validation():
    ar_val = pd.read_csv("data/Arabic/ar_val_1.6k.csv")
    ru_val = pd.read_csv("data/Russian/ru_val_1.6k.csv")
    en_val = pd.read_csv("data/English CLPsych/en_val_160.csv")
    ko_val = pd.read_csv("data/Korean/ko_val_160.csv")
    sp_val = pd.read_csv("data/Spanish/sp_val_320.csv")
    val_df = pd.concat([ar_val, ru_val, en_val, ko_val, sp_val])


    val_df.pop(val_df.columns[0])
    val_df = val_df.dropna()
    tweets_val =  val_df.text.astype(str).values.tolist()
    label_val = val_df.label.values.tolist()

    tweets_val_tokens = tokenization(tweets_val, CHECKPOINT)
    val_dataloader = prep_dataloader(tweets_val_tokens, label_val)

    return val_dataloader

# The `evaluate` function will be by Flower called after every round
def server_side_evaluation(server_round: int, parameters: fl.common.NDArrays, 
                config: Dict[str, fl.common.Scalar],) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    
    # valloader = centralized_trial_validation()
    valloader = centralized_validation()
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy, precision, recall, f1 = test(net, valloader)
    RESULTS['server_val_accuracy_loss_precesion_recall_f1'].append([server_round, accuracy, loss, precision, recall, f1])
    save_model(net, server_round)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy} / precision {precision} / recall {recall} / f1 {f1}")
    return loss,  {"accuracy": float(accuracy), "loss": float(loss), 
                   "precision": float(precision), "recall": float(recall), 
                   "f1": float(f1), "server_side_eval": "hena"}

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local 
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else LOCAL_EPOCHS,  #
    }
    return config

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.set_verbosity(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TOKENIZERS_PARALLELISM']= 'false'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    torch.cuda.empty_cache() 
    warnings.simplefilter('ignore')

    configure()

    global trainloaders, val_loaders, LOCAL_EPOCHS, AGG_TR_LOSSES
    global RESULTS, CHECKPOINT, LR, net, STORE_DIR, DEVICE, BATCH_SIZE, folder_name

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device is {DEVICE}")


    resume_dir = cnfg.resume_dir
    CHECKPOINT = cnfg.model_name
    LR = cnfg.learning_rate
    NUM_CLIENTS = cnfg.clients_num
    NUM_ROUNDS = cnfg.rounds
    LOCAL_EPOCHS = cnfg.epochs
    BATCH_SIZE = cnfg.batch_size
    folder_name = cnfg.saving_folder_name
    STORE_DIR = os.path.join('results', folder_name) 
    os.mkdir(STORE_DIR)
    STORE_DIR = os.path.join(STORE_DIR, datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    os.mkdir(STORE_DIR)
    AGG_TR_LOSSES = []
    RESULTS = {
                'total_rounds': NUM_ROUNDS,
                'batch_size':BATCH_SIZE,
                'agg_tr_accuracy_loss_precesion_recall_f1':[],
                'agg_val_accuracy_loss_precesion_recall_f1':[],
                'server_val_accuracy_loss_precesion_recall_f1':[]
            }

    net = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2).to(DEVICE)

    if resume_dir:
        saved_mod =  torch.load(resume_dir)
        net.load_state_dict(saved_mod['model'])

    params = get_parameters(net)

    if cnfg.data_function in data_funcs_dict:
        trainloaders, val_loaders = data_funcs_dict[cnfg.data_function](CHECKPOINT)
    else:
        print("Wrong data function.")
        return

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        fit_metrics_aggregation_fn = weighted_fit_average,
        evaluate_metrics_aggregation_fn=weighted_eval_average,
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        evaluate_fn=server_side_evaluation,
        on_fit_config_fn=fit_config,
    )

    client_resources = {"num_cpus":1, "num_gpus": 0}
    if DEVICE.type == "cuda":
        client_resources["num_gpus"] = 1
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources= client_resources,
        ray_init_args={"log_to_driver": False, "num_cpus": 1, "num_gpus": 1}
    )

    json.dump(RESULTS, open(f"{folder_name}.json",'w'))

if __name__ == "__main__":
    main()