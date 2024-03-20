import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torchmetrics.classification import MulticlassF1Score
from sklearn.metrics import confusion_matrix
import wandb
import numpy as np

from .model import Model, BertModel, BertWithPostionOnlyModel
from ddi_kt_2024.utils import save_model
from ddi_kt_2024.bc5_eval.bc5 import evaluate_bc5

class BaseTrainer:
    def __init__(self):
        pass
    def convert_label_to_2d(self, batch_label):
        i = 0
        for label in batch_label:
            i += 1
            tmp = torch.zeros((5)).to(self.device)
            tmp[label] = 1.
            
            if i == 1:
                to_return = tmp.unsqueeze(0)
            else:
                to_return = torch.vstack((to_return, tmp))
        
        return to_return
                

    def train_one_epoch(self, training_loader):
        running_loss = 0.
        i = 0


        for batch_data, batch_label in training_loader:
            batch_data = batch_data.clone().detach().to(self.device)
            batch_label = batch_label.clone().detach().to(self.device)
            batch_label = self.convert_label_to_2d(batch_label)
            i += 1
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            
        self.train_loss.append(running_loss)
        return running_loss

    
    def validate(self, validation_loader, option):
        running_loss = 0.
        predictions = torch.tensor([]).to(self.device)
        labels = torch.tensor([]).to(self.device)
        
        with torch.no_grad():
            for batch_data, batch_label in validation_loader:
                batch_data = batch_data.clone().detach().to(self.device)
                batch_label = batch_label.clone().detach().to(self.device)
                outputs = self.model(batch_data)
                batch_label_for_loss = self.convert_label_to_2d(batch_label)
                loss = self.criterion(outputs, batch_label_for_loss)
                running_loss += loss.item()
                
                batch_prediction = torch.argmax(outputs, dim=1)
                predictions = torch.cat((predictions, batch_prediction))
                labels = torch.cat((labels, batch_label))
        
        labels = labels.squeeze()
        true_pred = []
        for i in range(len(labels)):
            if labels[i].cpu() == 1:
                true_pred.append(i)
                
        cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[0, 1, 2, 3, 4])
        _micro_f1 = self.micro_f1(cm)

        f = MulticlassF1Score(num_classes=5, average=None).to(self.device)(predictions, labels)
        
        if option == 'train':
            self.train_loss.append(running_loss)
            self.train_f.append(f)
        elif option == 'val':
            self.val_loss.append(running_loss)
            self.val_f.append(f)
            self.val_micro_f1.append(_micro_f1)
    
    def micro_f1(self, cm):
        tp = cm[1][1] + cm[2][2] + cm[3][3] + cm[4][4]
        fp = np.sum(cm[:,1]) + np.sum(cm[:,2]) + np.sum(cm[:,3]) + np.sum(cm[:,4]) - tp
        fn = np.sum(cm[1,:]) + np.sum(cm[3,:]) + np.sum(cm[3,:]) + np.sum(cm[4,:]) - tp
        micro_f1 = tp / (tp + 1/2*(fp + fn))
        return micro_f1
            
    def plot_confusion_matrix(self, validation_loader):
        predictions = torch.tensor([]).to(self.device)
        labels = torch.tensor([]).to(self.device)
        
        with torch.no_grad():
            for batch_data, batch_label in validation_loader:
                batch_data = batch_data.clone().detach().to(self.device)
                batch_label = batch_label.clone().detach().to(self.device)
                outputs = self.model(batch_data)
                
                batch_prediction = torch.argmax(outputs, dim=1)
                predictions = torch.cat((predictions, batch_prediction))
                labels = torch.cat((labels, batch_label))
        
        label = labels.squeeze()
        cm = confusion_matrix(label.cpu().numpy(), predictions.cpu().numpy(), labels=[0, 1, 2, 3, 4])
        return cm
    
    def train(self, training_loader, validation_loader, num_epochs):
        loss = list()

        for epoch in tqdm(range(num_epochs)):
            running_loss = self.train_one_epoch(training_loader)
            
            self.validate(validation_loader, 'val')
            if self.wandb_available:
                self.log_wandb()
            # breakpoint()

            # Save model
            if self.val_micro_f1[-1] == max(self.val_micro_f1):
                save_model(f"checkpoints/{self.config.training_session_name}", f"epoch{epoch}loss{self.val_loss[-1]}val_micro_f1{self.val_micro_f1[-1]}.pt", self.config, self.model, self.wandb_available)
        if self.wandb_available:
            wandb.finish()

    def eval_bc5(self, pred, cand):
        dct, lst = self.convert_pred_to_lst(pred, cand)
        return_tuple = evaluate_bc5(lst)
        self.p.append(return_tuple[0][0])
        self.r.append(return_tuple[0][1])
        self.f.append(return_tuple[0][2])
        self.intra_p.append(return_tuple[1][0])
        self.intra_r.append(return_tuple[1][1])
        self.intra_f.append(return_tuple[1][2])
        self.inter_p.append(return_tuple[2][0])
        self.inter_r.append(return_tuple[2][1])
        self.inter_f.append(return_tuple[2][2])
        return return_tuple

    def convert_pred_to_lst(self, pred, cand):
        dct = {}
        for i in range(len(pred)):
            if pred[i] == 1:
                if cand[i]['e1']['@id'] == '-1' or cand[i]['e2']['@id'] == '-1':
                    continue
                elif len(cand[i]['e1']['@id'].split('|')) > 1:
                    tmp = cand[i]['e1']['@id'].split('|')
                    for ent in tmp:
                        idx = cand[i]['id'].split('_')[0]
                        if idx in dct.keys():
                            if f"{ent}_{cand[i]['e2']['@id']}" not in dct[idx]:
                                dct[idx].append(f"{ent}_{cand[i]['e2']['@id']}")
                        else:
                            dct[idx] = [f"{ent}_{cand[i]['e2']['@id']}"]
                elif len(cand[i]['e2']['@id'].split('|')) > 1:
                    tmp = cand[i]['e2']['@id'].split('|')
                    for ent in tmp:
                        idx = cand[i]['id'].split('_')[0]
                        if idx in dct.keys():
                            if f"{cand[i]['e1']['@id']}_{ent}" not in dct[idx]:
                                dct[idx].append(f"{cand[i]['e1']['@id']}_{ent}")
                        else:
                            dct[idx] = [f"{cand[i]['e1']['@id']}_{ent}"]
                else:
                    idx = cand[i]['id'].split('_')[0]
                    if idx in dct.keys():
                        if f"{cand[i]['e1']['@id']}_{cand[i]['e2']['@id']}" not in dct[idx]:
                            dct[idx].append(f"{cand[i]['e1']['@id']}_{cand[i]['e2']['@id']}")
                    else:
                        dct[idx] = [f"{cand[i]['e1']['@id']}_{cand[i]['e2']['@id']}"]

        lst = []
        for k, v in dct.items():
            for _ in v:
                lst.append((k, _, "CID"))

        return dct, lst

    def enable_parallel(self):
        self.model= torch.nn.DataParallel(self.model)
        if self.config is not None:
            self.model.to(self.config.device)

    def log_wandb(self):
        wandb.log(
                    {
                        "train_loss": self.train_loss[-1],
                        "val_loss": self.val_loss[-1],
                        "val_micro_f1": self.val_micro_f1[-1],
                        "val_f_false": self.val_f[-1][0],
                        "val_f_advise": self.val_f[-1][1],
                        "val_f_effect": self.val_f[-1][2],
                        "val_f_mechanism": self.val_f[-1][3],
                        "val_f_int": self.val_f[-1][4],
                        "max_val_micro_f1": max(self.val_micro_f1)
                    }
                )
        
class Trainer(BaseTrainer):
    def __init__(self,
                 we,
                 dropout_rate: float = 0.5,
                 word_embedding_size: int = 200,
                 tag_number: int = 51,
                 tag_embedding_size: int = 50,
                 position_number: int = 4,
                 position_embedding_size: int = 50,
                 direction_number: int = 3,
                 direction_embedding_size: int = 50,
                 edge_number: int = 46,
                 edge_embedding_size: int = 200,
                 token_embedding_size: int = 500,
                 dep_embedding_size: int = 500,
                 conv1_out_channels: int = 256,
                 conv2_out_channels: int = 256,
                 conv3_out_channels: int = 256,
                 conv1_length: int = 1,
                 conv2_length: int = 2,
                 conv3_length: int = 3,
                 target_class: int = 5,
                 w_false: float = 21580 / 17759,
                 w_advice: float = 21580 / 826,
                 w_effect: float = 21580 / 1687,
                 w_mechanism: float = 21580 / 1319,
                 w_int: float = 21580 / 189,
                 lr: float = 0.0001,
                 weight_decay: float = 1e-4,
                 device='cpu',
                 wandb_available=False):
        
        self.model = Model(we, 
                           dropout_rate,
                           word_embedding_size,
                           tag_number,
                           tag_embedding_size,
                           position_number,
                           position_embedding_size,
                           direction_number,
                           direction_embedding_size,
                           edge_number,
                           edge_embedding_size,
                           token_embedding_size,
                           dep_embedding_size,
                           conv1_out_channels,
                           conv2_out_channels,
                           conv3_out_channels,
                           conv1_length,
                           conv2_length,
                           conv3_length,
                           target_class).to(device)
        # weight = torch.tensor([w_false, w_advice, w_effect, w_mechanism, w_int]).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.train_loss = list()
        self.train_f = list()
        self.val_loss = list()
        self.val_f = list()
        self.val_micro_f1 = list()
        self.wandb_available = wandb_available
    
class BertTrainer(BaseTrainer):
    def __init__(self,
            dropout_rate: float = 0.5,
            word_embedding_size: int = 200,
            tag_number: int = 51,
            tag_embedding_size: int = 50,
            position_number: int = 4,
            position_embedding_size: int = 50,
            direction_number: int = 3,
            direction_embedding_size: int = 50,
            edge_number: int = 46,
            edge_embedding_size: int = 200,
            token_embedding_size: int = 500,
            dep_embedding_size: int = 500,
            conv1_out_channels: int = 256,
            conv2_out_channels: int = 256,
            conv3_out_channels: int = 256,
            conv1_length: int = 1,
            conv2_length: int = 2,
            conv3_length: int = 3,
            target_class: int = 5,
            w_false: float = 21580 / 17759,
            w_advice: float = 21580 / 826,
            w_effect: float = 21580 / 1687,
            w_mechanism: float = 21580 / 1319,
            w_int: float = 21580 / 189,
            lr: float = 0.0001,
            weight_decay: float = 1e-4,
            device='cpu',
            wandb_available=False
            ):
        self.model = BertModel(dropout_rate,
                           word_embedding_size,
                           tag_number,
                           tag_embedding_size,
                           position_number,
                           position_embedding_size,
                           direction_number,
                           direction_embedding_size,
                           edge_number,
                           edge_embedding_size,
                           token_embedding_size,
                           dep_embedding_size,
                           conv1_out_channels,
                           conv2_out_channels,
                           conv3_out_channels,
                           conv1_length,
                           conv2_length,
                           conv3_length,
                           target_class).to(device)
        # weight = torch.tensor([w_false, w_advice, w_effect, w_mechanism, w_int]).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.train_loss = list()
        self.train_f = list()
        self.val_loss = list()
        self.val_f = list()
        self.val_micro_f1 = list()
        self.wandb_available = wandb_available

class BertWithPostionOnlyTrainer(BaseTrainer):
    def __init__(self,
            dropout_rate: float = 0.5,
            word_embedding_size: int = 768,
            position_number: int = 4,
            position_embedding_size: int = 128,
            position_embedding_type: str = "normal",
            tag_number: int = 51,
            tag_embedding_size: int = 64,
            token_embedding_size : int = 256,
            conv1_out_channels: int = 256,
            conv2_out_channels: int = 256,
            conv3_out_channels: int = 256,
            conv1_length: int = 1,
            conv2_length: int = 2,
            conv3_length: int = 3,
            target_class: int = 5,
            w_false: float = 21580 / 17759,
            w_advice: float = 21580 / 826,
            w_effect: float = 21580 / 1687,
            w_mechanism: float = 21580 / 1319,
            w_int: float = 21580 / 189,
            lr: float = 0.0001,
            weight_decay: float = 1e-4,
            device='cpu',
            wandb_available=False
            ):
        self.model = BertWithPostionOnlyModel(
            dropout_rate,
            word_embedding_size,
            position_number,
            position_embedding_size,
            position_embedding_type,
            tag_number,
            tag_embedding_size,
            token_embedding_size ,
            conv1_out_channels,
            conv2_out_channels,
            conv3_out_channels,
            conv1_length,
            conv2_length,
            conv3_length,
            target_class,
        ).to(device)
        # weight = torch.tensor([w_false, w_advice, w_effect, w_mechanism, w_int]).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.train_loss = list()
        self.train_f = list()
        self.val_loss = list()
        self.val_f = list()
        self.val_micro_f1 = list()
        self.wandb_available = wandb_available

class BC5_Trainer(BaseTrainer):
    def __init__(self,
                test_cand,
                dropout_rate: float = 0.5,
                word_embedding_size: int = 200,
                tag_number: int = 51,
                tag_embedding_size: int = 50,
                position_number: int = 4,
                position_embedding_size: int = 50,
                position_embedding_type: str = 'linear',
                token_embedding_size = 256,
                conv1_out_channels: int = 256,
                conv2_out_channels: int = 256,
                conv3_out_channels: int = 256,
                conv1_length: int = 1,
                conv2_length: int = 2,
                conv3_length: int = 3,
                target_class: int = 2,
                hidden_channels: int = 512,
                lr: float = 0.0001,
                weight_decay: float = 1e-4,
                text_model: str = 'bert',
                modal: str = 'bc5_word',
                activation_function: str = 'relu',
                text_model_option: str = 'cnn',
                log: bool = True,
                device: str = 'cpu',
                wandb_available = False,
                **kwargs):
        self.device = device
        self.wandb_available = wandb_available
        self.model = BertWithPostionOnlyModel(
            dropout_rate,
            word_embedding_size,
            position_number,
            position_embedding_size,
            position_embedding_type,
            tag_number,
            tag_embedding_size,
            token_embedding_size,
            conv1_out_channels,
            conv2_out_channels,
            conv3_out_channels,
            conv1_length,
            conv2_length,
            conv3_length,
            target_class,
            dataset_name = "BC5").to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
        self.test_cand = test_cand
        self.train_loss = list()
        self.val_loss = list()
        self.p = list()
        self.r = list()
        self.f = list()
        self.intra_p = list()
        self.intra_r = list()
        self.intra_f = list()
        self.inter_p = list()
        self.inter_r = list()
        self.inter_f = list()
        self.log = log

    def validate(self, validation_loader, option):
        running_loss = 0.
        predictions = torch.tensor([]).to(self.device)
        labels = torch.tensor([]).to(self.device)
        
        with torch.no_grad():
            for batch_data, batch_label in validation_loader:
                batch_data = batch_data.clone().detach().to(self.device)
                batch_label = batch_label.clone().detach().to(self.device)
                outputs = self.model(batch_data)
                batch_label_for_loss = self.convert_label_to_2d(batch_label)
                loss = self.criterion(outputs, batch_label_for_loss)
                running_loss += loss.item()
                
                batch_prediction = torch.argmax(outputs, dim=1)
                predictions = torch.cat((predictions, batch_prediction))
                labels = torch.cat((labels, batch_label))
        
        labels = labels.squeeze()
        true_pred = []
        for i in range(len(labels)):
            if labels[i].cpu() == 1:
                true_pred.append(i)
                
        cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[0, 1, 2, 3, 4])
        
        self.eval_bc5(predictions.cpu().numpy(), self.test_cand)

        if option == 'train':
            self.train_loss.append(running_loss)
        elif option == 'val':
            self.val_loss.append(running_loss)

    def log_wandb(self):
        wandb.log(
            {
                "train_loss": self.train_loss[-1],
                "val_loss": self.val_loss[-1],
                "precision": self.p[-1],
                "recall": self.r[-1],
                "f1": self.f[-1],
                "intra_precision": self.intra_p[-1],
                "intra_recall": self.intra_r[-1],
                "intra_f1": self.intra_f[-1],
                "inter_precision": self.inter_p[-1],
                "inter_recall": self.inter_r[-1],
                "inter_f1": self.inter_f[-1]
            }
        )