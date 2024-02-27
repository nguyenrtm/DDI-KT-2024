import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassF1Score
import numpy as np
import wandb

from src.mol.gcn import GCN

class Trainer:
    def __init__(self,
                 num_node_features: int,
                 hidden_channels: int,
                 w_false: float = 16626 / 14463,
                 w_advice: float = 16626 / 487,
                 w_effect: float = 16626 / 625,
                 w_mechanism: float = 16626 / 927,
                 w_int: float = 16626 / 124,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 device: str = 'cpu'
                 ):
        weight = torch.tensor([w_false, w_advice, w_effect, w_mechanism, w_int]).to(device)
        self.model = GCN(num_node_features=num_node_features, hidden_channels=hidden_channels, device=device).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight)
        self.device = device
        self.train_loss = list()
        self.train_f = list()
        self.val_loss = list()
        self.val_f = list()
        self.val_micro_f1 = list()
        
    def convert_label_to_2d(self, label):
        tmp = torch.zeros((5)).to(self.device)
        tmp[label] = 1.
        
        return tmp.unsqueeze(dim=0)
    
    def train_one_epoch(self, dataset_train):
        running_loss = 0.
        i = 0

        for data in dataset_train:
            mol1 = data[0][0].to(self.device)
            mol2 = data[0][1].to(self.device)
            label = data[1]
            label = self.convert_label_to_2d(label)
            
            i += 1
            out = self.model(mol1, mol2)
            self.optimizer.zero_grad()
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
        self.train_loss.append(running_loss)
        return running_loss
    
    def validate(self, dataset_test, option):
        running_loss = 0.
        predictions = torch.tensor([]).to(self.device)
        labels = torch.tensor([]).to(self.device)
        
        with torch.no_grad():
            for data in dataset_test:
                mol1 = data[0][0].to(self.device)
                mol2 = data[0][1].to(self.device)
                label = torch.tensor([data[1]]).to(self.device)

                out = self.model(mol1, mol2)
                label_for_loss = self.convert_label_to_2d(label)
                loss = self.criterion(out, label_for_loss)
                running_loss += loss.item()
                prediction = torch.argmax(out, dim=1)

                predictions = torch.cat((predictions, prediction))
                labels = torch.cat((labels, label))
        
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
        
    def train(self, dataset_train, dataset_test, num_epochs):
        loss = list()

        for epoch in tqdm(range(num_epochs)):
            running_loss = self.train_one_epoch(dataset_train)
            
            self.validate(dataset_train, 'train')
            self.validate(dataset_test, 'val')
            
            wandb.log(
                {
                    "train_loss": self.train_loss[-1],
                    "train_f": self.train_f[-1],
                    "val_loss": self.val_loss[-1],
                    "val_micro_f1": self.val_micro_f1[-1],
                    "val_f_false": self.val_f[-1][0],
                    "val_f_advise": self.val_f[-1][1],
                    "val_f_effect": self.val_f[-1][2],
                    "val_f_mechanism": self.val_f[-1][3],
                    "val_f_int": self.val_f[-1][4],
                }
            )
            
        wandb.finish()