import os
import json
import warnings
from datetime import datetime
from sklearn.metrics import confusion_matrix

import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from transformers import get_scheduler

import torch.backends.cudnn as cudnn

from sklearn.metrics import accuracy_score

from testing_config import _C as cnfg
from utils import progress_bar
from cnt_lcl_data_read import *


def configure():
    cfg = "testing_config.yaml"
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

    return tokenized_feature

def prep_dataloader(tokens, labels):
    data = TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.tensor(labels))
    sampler = RandomSampler(data)
    data_loader = DataLoader(data, batch_size=batch_size, sampler= sampler)
    return data_loader

def model_build(model_name):
    #cudnn stuff
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels = 2, 
    output_attentions = False,
    output_hidden_states = False)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model).cuda()

    return model

def perf_measures(TN, FP, FN, TP):
    percision = round(float(TP / (TP + FP)),4)
    recall = round(float(TP / (TP + FN)),4)
    f_score = round(float(2 * ((percision*recall)/(percision+recall))),4)
    return percision, recall, f_score

def model_in_action(mode,loader, model):
    mode_loss = 0
    correct = 0
    total = 0
    all_preds = torch.tensor([], dtype=int, device=device)
    all_targets = torch.tensor([],dtype=int,device=device)
    print(f"<---------------{mode}--------------->")
    model.eval()
    for step, batch in enumerate(loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

        # print(outputs.loss)    
        mode_loss += outputs.loss.item() #Not computed as there are no labels passed to the model forward function
        
        all_targets = torch.cat((all_targets, b_labels),dim=0)
        preds = torch.argmax(outputs.logits.to(device), dim=1)
        all_preds = torch.cat((all_preds, preds.to(device)),dim=0)

        total += b_labels.size(0)
            
        correct += preds.eq(b_labels).sum().item()

        progress_bar(step, len(loader), f"Loss: {mode_loss/(step+1):.3f} | Acc: {100.*correct/total:.3f}%")
            
    epoch_acc = 100.*correct/total
    epoch_loss = mode_loss/len(loader)
    epoch_cm = confusion_matrix(all_targets.tolist(), all_preds.tolist())
    epoch_pr, epoch_rec, epoch_f1 = perf_measures(epoch_cm[0][0], epoch_cm[0][1], 
                                                  epoch_cm[1][0], epoch_cm[1][1])
    
    return epoch_acc, epoch_loss, epoch_cm.tolist(), epoch_pr, epoch_rec, epoch_f1 

def saved_model(model, trained_file):
    if resume_dir:
        print("Testing....")
        checkpoint = torch.load(os.path.join(resume_dir, trained_file))
        model.load_state_dict(checkpoint['model'])
    else:
        print("Kindly provide the model path")
    
    return model

def test_many_epochs_many_langs(langs, saving_file):
    print(resume_dir)
    res = dict()
    epochs = os.listdir(resume_dir)
    epochs.sort()
    for model_file in epochs:
        model = model_build(model_name)
        model = saved_model(model, model_file)
        res[model_file] = []
        print(model_file)
        for fn in langs:
            tr_df, val_df, ts_df = fn()
            tweets_ts =  ts_df.text.astype(str).values.tolist()
            label_ts= ts_df.label.values.tolist()
            tweets_ts_tokens = tokenization(tweets_ts, model_name)
            ts_dataloader = prep_dataloader(tweets_ts_tokens, label_ts)
            ts_acc, ts_loss, ts_cm, ts_pr, ts_rec, ts_f1 = model_in_action("test",ts_dataloader, model)
            print(fn.__name__ + " , " + str(ts_acc))
            res[model_file].append(ts_acc)
    res_df = pd.DataFrame.from_dict(res)
    res_df.to_csv(saving_file+".csv", index=False)

def test_one_epoch_many_langs(langs, model_file, saving_file):
    print(resume_dir)
    res = dict()
    model = model_build(model_name)
    model = saved_model(model, model_file)
    res[model_file] = []
    print(model_file)
    for fn in langs:
        tr_df, val_df, ts_df = fn()
        tweets_ts =  ts_df.text.astype(str).values.tolist()
        label_ts= ts_df.label.values.tolist()
        tweets_ts_tokens = tokenization(tweets_ts, model_name)
        ts_dataloader = prep_dataloader(tweets_ts_tokens, label_ts)
        ts_acc, ts_loss, ts_cm, ts_pr, ts_rec, ts_f1 = model_in_action("test",ts_dataloader, model)
        print(fn.__name__ + " , " + str(ts_acc))
        res[model_file].append(ts_acc)
    res_df = pd.DataFrame.from_dict(res)
    res_df.to_csv(saving_file+".csv", index=False)

def test_one_epoch_one_lang(lang, model_file):
    print(resume_dir)
    res = dict()
    model = model_build(model_name)
    model = saved_model(model, model_file)
    res[model_file] = []
    print(model_file)
    tr_df, val_df, ts_df = lang()
    tweets_ts =  ts_df.text.astype(str).values.tolist()
    label_ts= ts_df.label.values.tolist()
    tweets_ts_tokens = tokenization(tweets_ts, model_name)
    ts_dataloader = prep_dataloader(tweets_ts_tokens, label_ts)
    ts_acc, ts_loss, ts_cm, ts_pr, ts_rec, ts_f1 = model_in_action("test",ts_dataloader, model)
    print(lang.__name__ + " , " + str(ts_acc))
    
def test_many_epochs_one_lang(lang, saving_file):
    print(resume_dir)
    res = dict()
    tr_df, val_df, ts_df = lang()
    tweets_ts =  ts_df.text.astype(str).values.tolist()
    label_ts= ts_df.label.values.tolist()
    tweets_ts_tokens = tokenization(tweets_ts, model_name)
    ts_dataloader = prep_dataloader(tweets_ts_tokens, label_ts)
    epochs = os.listdir(resume_dir)
    epochs.sort()
    for model_file in epochs:
        print(model_file)
        model = model_build(model_name)
        model = saved_model(model, model_file)
        res[model_file] = []
        ts_acc, ts_loss, ts_cm, ts_pr, ts_rec, ts_f1 = model_in_action("test",ts_dataloader, model)
        print(lang.__name__ + " , " + str(ts_acc))
        res[model_file].append(ts_acc)
    res_df = pd.DataFrame.from_dict(res)
    res_df.to_csv(saving_file+".csv", index=False)

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.cuda.empty_cache()
    print(f"Starting time : {datetime.now()}")
    configure()
    global batch_size, model_name, device, resume_dir
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    testing_mode = cnfg.testing_mode
    resume_dir = cnfg.resume_dir
    resume_epoch = cnfg.resume_epoch
    results_file = cnfg.results_file
    model_name = cnfg.model_name
    batch_size = cnfg.batch_size
    data_func = cnfg.data_func

    print(testing_mode, resume_dir, resume_epoch, results_file, model_name, batch_size, data_func)
   
    tsting_fns = {
        'test_many_epochs_many_langs': test_many_epochs_many_langs,
        'test_one_epoch_many_langs': test_one_epoch_many_langs,
        'test_one_epoch_one_lang': test_one_epoch_one_lang,
        'test_many_epochs_one_lang': test_many_epochs_one_lang,
    }

    #Data Loading
    if data_func in data_funcs_dict:
        if testing_mode == "test_many_epochs_many_langs":
            tsting_fns['test_many_epochs_many_langs'](data_funcs_dict[data_func], results_file)
        elif testing_mode == "test_one_epoch_many_langs":
            tsting_fns['test_one_epoch_many_langs'](data_funcs_dict[data_func], resume_epoch, results_file)
        elif testing_mode == "test_one_epoch_one_lang":
            tsting_fns['test_one_epoch_one_lang'](data_funcs_dict[data_func], resume_epoch)
        elif testing_mode == "test_many_epochs_one_lang":
            tsting_fns['test_many_epochs_one_lang'](data_funcs_dict[data_func], results_file)
        else:
            print("Wrong testing mode.")
    else:
        print("Wrong data function.")

    print(f"Ending time : {datetime.now()}")

if __name__ == "__main__":
    main()