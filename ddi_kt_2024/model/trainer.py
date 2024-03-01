import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torchmetrics.classification import MulticlassF1Score
from sklearn.metrics import confusion_matrix
import wandb
import numpy as np

from .model import Model
from ddi_kt_2024.utils import save_model

class Trainer:
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
        weight = torch.tensor([w_false, w_advice, w_effect, w_mechanism, w_int]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.train_loss = list()
        self.train_f = list()
        self.val_loss = list()
        self.val_f = list()
        self.val_micro_f1 = list()
        self.wandb_available = wandb_available
        
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
                    }
                )

            # Save model
            if self.val_micro_f1[-1] == max(self.val_micro_f1):
                save_model(f"checkpoints/{self.config.training_session_name}", f"epoch{epoch}loss{self.val_loss[-1]}val_micro_f1{self.val_micro_f1[-1]}.pt", self.config, self.model, self.wandb_available)
        if self.wandb_available:
            wandb.finish()