import os
import json
import warnings
from datetime import datetime
from sklearn.metrics import confusion_matrix

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AutoTokenizer

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


import torch.backends.cudnn as cudnn

from cnt_lcl_train_config import _C as cnfg
from utils import progress_bar
from cnt_lcl_data_read import *

def configure():
    cfg = "cnt_lcl_train_config.yaml"
    cnfg.defrost()
    cnfg.merge_from_file(cfg)
    cnfg.freeze()

def tokenization(tweets, model_name):
    Max_Len  = 150   #According to what is returned from the commented code below
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
    data_loader = DataLoader(data, batch_size=batch_size, sampler= sampler)
    return data_loader

def model_build(model_name, folder_name):
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

    optimizer = AdamW(model.parameters(), lr = learning_rate, eps = 1e-8)
    store_dir = os.path.join('results', folder_name) 
    os.mkdir(store_dir)
    store_dir = os.path.join(store_dir, datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    os.mkdir(store_dir)
    return model, optimizer, store_dir

def perf_measures(TN, FP, FN, TP):
    percision = round(float(TP / (TP + FP)),4)
    recall = round(float(TP / (TP + FN)),4)
    f_score = round(float(2 * ((percision*recall)/(percision+recall))),4)
    return percision, recall, f_score

def model_in_action(mode,loader,epoch=0, best_val_acc=0):
    mode_loss = 0
    correct = 0
    total = 0
    all_preds = torch.tensor([], dtype=int, device=device)
    all_targets = torch.tensor([],dtype=int,device=device)
    print(f"<---------------{mode}--------------->")
    if mode == 'train':
        # switch to train mode
        model.train()
        for step, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
           
            outputs = model( b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            mode_loss += outputs.loss.item()
            all_targets = torch.cat((all_targets, b_labels),dim=0)
            preds = torch.argmax(outputs.logits.to(device), dim=1)

            all_preds = torch.cat((all_preds, preds.to(device)),dim=0)

            total += b_labels.size(0)
                
            correct += preds.eq(b_labels).sum().item()

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar(step, len(loader), f"Loss: {mode_loss/(step+1):.3f} | Acc: {100.*correct/total:.3f}%")
    else:
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
    if mode == "val":
        if epoch_acc > best_val_acc:
            print('Saving..')
            state = {
                    'model': model.state_dict(),
                    'best_val_acc': epoch_acc,
                    'best_val_loss': epoch_loss,
                    'epoch': epoch+1,
                    'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(store_dir, f'best_ckpt.pth'))
            best_val_acc = epoch_acc
            best_val_loss = epoch_loss

        state = {
                'model': model.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'epoch': epoch+1,
                'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(store_dir, f"train-{epoch}.pth"))
    
    return epoch_acc, epoch_loss, epoch_cm.tolist(), epoch_pr, epoch_rec, epoch_f1 

def saved_model(model, optimizer=''):
    best_val_acc = 0  
    best_val_loss = 0
    start_epoch = 0
    if resume_dir:
        print('==> Resuming from checkpoint..')
        print("Training....")
        checkpoint = torch.load(os.path.join(resume_dir, resume_epoch))

        model.load_state_dict(checkpoint['model'])
        
        best_val_acc = checkpoint['best_val_acc']
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']

        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, optimizer, best_val_acc, best_val_loss, start_epoch

def check_early_stopping(tr_losses, min_delta = 0.01):
    count = 0
    for i in range(early_stopping_patience-1, 0,-1):
        # print(tr_losses[i],":",tr_losses[i-1],"-->", round(abs(tr_losses[i]-tr_losses[i-1]),2))
        if(round(abs(tr_losses[i]-tr_losses[i-1]),2))<= min_delta:
            count += 1
    if count == early_stopping_patience-1:
        return True
    else:
        return False

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.cuda.empty_cache()
    print(f"Starting time : {datetime.now()}")
    configure()
    global epochs, batch_size
    global device, model, optimizer, scheduler, learning_rate
    global best_val_acc, best_val_loss, store_dir, resume_dir, resume_epoch
    global early_stopping_patience
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    resume_dir = cnfg.resume_dir
    resume_epoch = cnfg.resume_epoch
    epochs = cnfg.epochs
    early_stopping_patience = cnfg.early_stopping_patience
    model_name = cnfg.model_name
    learning_rate = cnfg.learning_rate
    batch_size = cnfg.batch_size
    folder_name = cnfg.saving_folder_name
    data_func = cnfg.data_function
    
    #Data Loading
    if data_func in data_funcs_dict:
        tr_df, val_df, ts_df = data_funcs_dict[data_func]()
    else:
        print("Wrong data function.")

    tweets_tr =  tr_df.text.astype(str).values.tolist()
    label_tr = tr_df.label.values.tolist()
    tweets_val =  val_df.text.astype(str).values.tolist()
    label_val = val_df.label.values.tolist()

    tweets_tr_tokens = tokenization(tweets_tr, model_name)
    tweets_val_tokens = tokenization(tweets_val, model_name)

    tr_dataloader = prep_dataloader(tweets_tr_tokens, label_tr)
    val_dataloader = prep_dataloader(tweets_val_tokens, label_val)

    model, optimizer, store_dir = model_build(model_name, folder_name)
    model, optimizer, best_val_acc, best_val_loss, start_epoch = saved_model(model, optimizer)
    total_steps = len(tr_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    results = {
        'total_epochs': epochs,
        'batch_size':batch_size,
        'tr_val_accuracy':[],
        'tr_val_loss':[],
        'tr_confusion_matrix': [],
        'val_confusion_matrix': [],
        'tr_precesion_recall_f1': [],
        'val_precesion_recall_f1': []
    }
    tr_losses = []
    for epoch in range(start_epoch, start_epoch+epochs):
        print(f"\nEpoch: {epoch}")
    
        tr_acc, tr_loss, tr_cm, tr_pr, tr_rec, tr_f1 = model_in_action("train",tr_dataloader,epoch, best_val_acc)
        val_acc, val_loss, val_cm, val_pr, val_rec, val_f1 =  model_in_action("val",val_dataloader,epoch, best_val_acc)
        
        print("***************Training Results***************")
        print(f"Accuracy -----> {tr_acc:.4f}")
        print(f"Loss -----> {tr_loss:.4f}")
        print(f"Confusion Matrix -----> {tr_cm}")
        print(f"Percicion -----> {tr_pr}")
        print(f"Recall -----> {tr_rec}")
        print(f"F1-Score -----> {tr_f1}")

        print("***************Validation Results***************")
        print(f"Accuracy -----> {val_acc:.4f}")
        print(f"Loss -----> {val_loss:.4f}")
        print(f"Confusion Matrix -----> {val_cm}")
        print(f"Percicion -----> {val_pr}")
        print(f"Recall -----> {val_rec}")
        print(f"F1-Score -----> {val_f1}")

        results['tr_val_accuracy'].append([epoch, tr_acc, val_acc])
        results['tr_val_loss'].append([epoch, tr_loss, val_loss])
        results['tr_confusion_matrix'].append([epoch, tr_cm])
        results['val_confusion_matrix'].append([epoch, val_cm])
        results['tr_precesion_recall_f1'].append([epoch, tr_pr, tr_rec, tr_f1])
        results['val_precesion_recall_f1'].append([epoch, val_pr, val_rec, val_f1])

        json.dump(results, open(f"{folder_name}",'w'))
        #Saving after last epoch
        state = {
                    'model': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss,
                    'epoch': start_epoch+epochs,
                    'optimizer': optimizer.state_dict()
                }
        torch.save(state, os.path.join(store_dir, 'after_ckpt.pth'))

        tr_losses.append(tr_loss)
        if(len(tr_losses) >= early_stopping_patience):
            booli = check_early_stopping(tr_losses[-early_stopping_patience:])
            print(booli)
            if booli:
                print(f"Early stopped at epoch: {epoch}")
                break

    print(f"Ending time : {datetime.now()}")

if __name__ == "__main__":
    main()