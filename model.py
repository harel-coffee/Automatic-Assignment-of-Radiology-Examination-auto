
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME, AdamW,    BertTokenizer,  get_linear_schedule_with_warmup
from torch.utils.data import Dataset,RandomSampler,SequentialSampler,DataLoader
import torch
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from tqdm import tqdm
import datetime
import time
from sys import platform
from sklearn.metrics import precision_recall_fscore_support,classification_report
import os
import pickle
import json
from pathlib import Path

def flat_accuracy(preds, labels):
    return np.sum(preds == labels) / len(labels)

class BertForSequenceClassification_with_pooled_outputs(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        softlabels=None,
        teacher_labels_idx=None,

    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1  :
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))

            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits .view(-1, self.num_labels),
                                    labels .view(-1))

            outputs = (loss,) + outputs

        return outputs ,pooled_output


class BERTDataset(Dataset):

    def __init__(self, docs,labels,softlabels,tokenizer, teacher_labels=None,  maxlen=512):

        self.docs=docs
        self.labels=labels
        self.teacher_labels=teacher_labels
        self.maxlen = maxlen
        self.num_tokens=0
        self.inputIDs = []
        self.softlabels=softlabels

        for sent in self.docs:
            encoded_doc = tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=self.maxlen,
                pad_to_max_length=True,
                truncation=True
            )

            self.inputIDs.append(encoded_doc)

        self.attention_masks = []
        for sent in self.inputIDs:
            att_mask = [int(token_id > 0) for token_id in sent]
            self.attention_masks.append(att_mask)


    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index):
        if self.labels is not None:
            label = torch.tensor(self.labels [index ]).long()
        else:
            label= torch.zeros(len(self.inputIDs[index]))

        if self.softlabels is not None:
            softlabels = torch.tensor(self.softlabels[index])
        else:
            softlabels = torch.zeros(len(self.inputIDs[index]))

        tokens_ids_tensor = torch.tensor(self.inputIDs[index])
        attn_mask = torch.tensor(self.attention_masks[index])
        text = self.docs[index]

        return tokens_ids_tensor, attn_mask, label,softlabels,text,index



class BERT():

    def __init__(self,  X_train, X_validation, Y_train, Y_train_softlabels, Y_validation,args,groups,X_train_teacher=None ,
                 Y_train_teacher=None):
        self.args=args
        self.patience = args.patience
        self.groups=groups
        self.x_train= X_train
        self.x_validation = X_validation
        self.y_train_teacher=Y_train_teacher

        if args.distill:
            model_data = json.load(open(os.path.join(args.nn_teacher_model, 'model.json'), 'r'))
            self.unique_labels =  model_data['labels']
            self.label2id_map = {t: i for i, t in enumerate(self.unique_labels)}
            self.id2label_map = {i: t for i, t in enumerate(self.unique_labels)}
            self.teacher_labels_idx = len(Y_train_teacher)
            self.y_train = [self.label2id_map[t] for t in Y_train]
            self.y_train_softlabels =Y_train_softlabels
            self.y_validation = [self.label2id_map[t] for t in Y_validation]
            self.x_train_teacher = X_train_teacher
            self.y_train_teacher = [self.label2id_map[t] for t in Y_train_teacher]

        elif args.softlabels:
            model_data = json.load(open(os.path.join(args.nn_teacher_model, 'model.json'), 'r'))
            self.unique_labels =  model_data['labels']
            self.label2id_map = {t: i for i, t in enumerate(self.unique_labels)}
            self.id2label_map = {i: t for i, t in enumerate(self.unique_labels)}
            self.y_train = [self.label2id_map[t] for t in Y_train]
            self.y_validation = None
        else:
            self.unique_labels = np.unique(Y_train+Y_validation)
            self.label2id_map = {t: i for i, t in enumerate(self.unique_labels)}
            self.id2label_map = {i: t for i, t in enumerate(self.unique_labels)}
            self.y_train = [self.label2id_map[t] for t in Y_train]
            self.y_validation = [self.label2id_map[t] for t in Y_validation]

        self.batch=args.batch
        self.epochs =args.epochs if platform != "darwin" else 10

        bert_model = args.nn_model
        self.model = BertForSequenceClassification_with_pooled_outputs.from_pretrained(
            bert_model,
            num_labels=self.unique_labels.__len__(),
            output_attentions=False,
            output_hidden_states=False,
        )
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        if torch.cuda.device_count()>0:

            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
            self.device=torch.device("cuda")

        else:
            print('using CPU')
            self.device =  torch.device("cpu")

        self.trainingset = BERTDataset(self.x_train, self.y_train, self.y_train_softlabels if args.distill else None, self.tokenizer, self.y_train_teacher, self.args.maxlen)
        self.train_loader = DataLoader(self.trainingset, sampler=RandomSampler(self.trainingset),
                                         batch_size=self.batch )

        validset = BERTDataset(self.x_validation, self.y_validation,self.y_validation if args.distill else None, self.tokenizer,None, self.args.maxlen)
        print('validset',validset.__len__())
        self.val_loader = DataLoader(validset, sampler=SequentialSampler(validset),
                                         batch_size=self.batch)

        if X_train_teacher is not None:
            self.trainingset_teacher = BERTDataset(self.x_train_teacher, self.y_train_teacher,  None,
                                         self.tokenizer, self.y_train_teacher, self.args.maxlen)

            self.x_train_teacher_loader = DataLoader(self.trainingset_teacher, sampler=RandomSampler(self.trainingset_teacher),
                                         batch_size=self.batch )

    def init_sched_optm(self):
        params = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {
                'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            {
                'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        self.optimizer = AdamW(grouped_params, correct_bias=False, lr=self.args.learning_rate)


    def train(self,fold, distill=False):

        self.init_sched_optm()
        loss_values = []
        best_f1=0
        patience_count=0
        best_results={}

        loader = self.train_loader if not self.args.distill or distill else self.x_train_teacher_loader

        for epoch_i in range(0, self.epochs):

            print("")
            print('======== Epoch {:} / {:} / Fold {:} Size {:} ========'.format(epoch_i + 1, self.epochs,fold,len(self.x_train)))

            t0 = time.time()

            total_loss = 0
            self.model.train()
            for step, batch in enumerate(tqdm(loader,desc=('Training' if not distill or not self.args.distill else 'Distilling') + '..fold {} .epoch {} '.format(fold, epoch_i))):
                batch = tuple(p for p in batch)

                input_ids, attention_mask, labels,softlabels, _, indices = batch

                if distill:
                    teacher_labels_idx = [i for i,idx in enumerate(indices) if idx <self.teacher_labels_idx]

                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        labels = labels.cuda()
                        softlabels = softlabels.cuda()

                    outputs, pooledoutput  = self.model(input_ids ,
                                          token_type_ids=None,
                                          attention_mask=attention_mask,
                                          labels=labels,
                                          softlabels=softlabels ,
                                          teacher_labels_idx=teacher_labels_idx)

                    logits = outputs[1]
                    loss_fct = MSELoss()
                    distill_loss = loss_fct(logits.view(-1), softlabels.view(-1))

                    if len(teacher_labels_idx) >0:
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits[teacher_labels_idx].view(-1, self.unique_labels.__len__()), labels[teacher_labels_idx].view(-1))
                    else:
                        loss = 0

                    if self.args.distill_weight > 0:
                        distill_weight = self.args.distill_weight
                    else:
                        distill_weight = len(teacher_labels_idx) / len(logits)

                    loss = distill_weight * loss + (1 - distill_weight) * distill_loss

                else:

                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        labels = labels.cuda()

                    outputs, pooledoutput = self.model(input_ids,
                                                   token_type_ids=None,
                                                   attention_mask=attention_mask,
                                                   labels=labels,
                                                       softlabels=None)
                    loss = outputs[0]

                    if torch.cuda.device_count() > 0:
                        loss = loss.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss = loss.detach().item()

                total_loss += epoch_loss
 
            avg_train_loss = total_loss / len(self.train_loader)

            loss_values.append(avg_train_loss)

            print("")
            if(distill):
                print('distill_weight : {}'.format(distill_weight))

            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(str(datetime.timedelta(seconds=(time.time() - t0)))))


            # ========================================
            #               Validation
            # ========================================

            t0 = time.time()
            self.model.eval()

            eval_accuracy = 0
            eval_steps  = 0
            y_pred=[]
            y_prob = []
            y_validation=[]

            with torch.no_grad():
                for eval_steps, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                    batch = tuple(p for p in batch)
                    input_ids, attention_mask, labels, softlabels, _,  indices = batch

                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        labels = labels.cuda()

                    outputs, pooled_output = self.model(input_ids,
                                         token_type_ids=None,
                                         attention_mask=attention_mask)

                    logits = outputs[0]
                    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                    probs = torch.max(torch.softmax(logits, dim=-1),dim=-1)[0].detach().cpu().numpy()
                    y_prob.extend(probs)
                    y_pred.extend([self.id2label_map[y] for y in preds] )
                    label_ids = labels.to('cpu').numpy()
                    y_validation.extend([self.id2label_map[y] for y in label_ids] )

                    batch_accuracy = flat_accuracy(preds, label_ids)
                    eval_accuracy += batch_accuracy

                target_names = set(y_validation)
                target_names = sorted(target_names)

                print(classification_report(y_validation, y_pred, labels=self.groups, digits=4))
                results = (
                classification_report(y_validation, y_pred, labels=self.groups, output_dict=False, digits=4),
                classification_report(y_validation, y_pred, labels=self.groups, output_dict=True, digits=4))

                pickle.dump([y_validation, y_pred,y_prob], open("fold_{}_preds.pickle".format(fold), "wb"))

            # Report the final accuracy for this validation run.
            print("  Accuracy: {0:.2f}".format(eval_accuracy / eval_steps))
            print("  Validation took: {:}".format(str(datetime.timedelta(seconds=(time.time() - t0)))))

            precsions_all, recalls_all, f1_all, _ = precision_recall_fscore_support(y_validation, y_pred,
                                                                                    labels=self.groups)

            if self.args.earlystop:
                if best_f1 < np.mean(f1_all):
                    best_f1= np.mean(f1_all)
                    patience_count = 0
                    best_results={'y_validation':y_validation, 'y_pred':y_pred}
                    self.save(fold,self.epochs)
                else:
                    patience_count += 1
                    print(" last {} epochs show no improvement".format(patience_count))
                    if patience_count == self.patience:
                        print("*****************")
                        print("** Best result **")
                        print("*****************")
                        print(classification_report(best_results['y_validation'], best_results['y_pred'], labels=self.groups))

                        break
            else:
                best_results = {'y_validation': y_validation, 'y_pred': y_pred}
                self.save(fold, epoch_i)

        print("")
        results = (
            classification_report(best_results['y_validation'], best_results['y_pred'], labels=self.groups,
                                  output_dict=False),
            classification_report(best_results['y_validation'], best_results['y_pred'], labels=self.groups,
                                  output_dict=True))

        print("Training complete!")
        return results

    def save(self,fold,epoch):

        model_path = 'saved_model_fold_{}_epoch_{}'.format(fold,epoch)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        model_to_save = self.model.module if torch.cuda.device_count() > 0  else self.model
        torch.save(model_to_save.state_dict(), os.path.join(model_path, WEIGHTS_NAME))
        model_to_save.config.to_json_file(os.path.join(model_path, CONFIG_NAME))
        torch.save({'model_state_dict': model_to_save.state_dict()}, os.path.join(model_path, 'checkpoint'))
        self.tokenizer.save_vocabulary(model_path)
        model_meta =  { 'labels':  list(self.unique_labels)}
        json.dump(model_meta, open(os.path.join(model_path, 'model.json'), 'w'))


    def generate_softlabels(self,fold):

        ds = BERTDataset( self.x_train,
                       self.y_train , None, self.tokenizer, self.args.maxlen)

        print('data+augmented',ds.__len__())
        ds_loader = DataLoader(ds, sampler=SequentialSampler(ds),
                                         batch_size=self.batch)


        soft_labels=[]
        hard_labels=[]
        texts=[]
        with torch.no_grad():
            for step, batch in enumerate(
                        tqdm(ds_loader, desc='generate_softlabels ')):

                # Add batch to GPU
                batch = tuple(p for p in batch)
                input_ids, attention_mask, labels, _, text,_ = batch
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    labels = labels.cuda()

                outputs, pooled_output = self.model(input_ids,
                                                    token_type_ids=None,
                                                    attention_mask=attention_mask)

                logits = outputs[0]
                soft_labels.extend(logits.detach().cpu().numpy())
                texts.extend(text)
                preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                hard_labels.extend([self.id2label_map[y] for y in preds])

            pickle.dump(list(zip(texts,soft_labels,hard_labels)), open("soft_labels_fold_{}.pickle".format(fold), "wb"))

