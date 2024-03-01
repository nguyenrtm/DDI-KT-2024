import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassF1Score
import numpy as np
import wandb

from src.multimodal.model import MultimodalModel

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
                 num_node_features: int = 4, 
                 hidden_channels: int = 256,
                 w_false: float = 21580 / 17759,
                 w_advice: float = 21580 / 826,
                 w_effect: float = 21580 / 1687,
                 w_mechanism: float = 21580 / 1319,
                 w_int: float = 21580 / 189,
                 lr: float = 0.0001,
                 weight_decay: float = 1e-4,
                 device: str = 'cpu'
                 ):
        weight = torch.tensor([w_false, w_advice, w_effect, w_mechanism, w_int]).to(device)
        
        self.model = MultimodalModel(we=we,
                                    dropout_rate=dropout_rate,
                                    word_embedding_size=word_embedding_size,
                                    tag_number=tag_number,
                                    tag_embedding_size=tag_embedding_size,
                                    position_number=position_number,
                                    position_embedding_size=position_embedding_size,
                                    direction_number=direction_number,
                                    direction_embedding_size=direction_embedding_size,
                                    edge_number=edge_number,
                                    edge_embedding_size=edge_embedding_size,
                                    token_embedding_size=token_embedding_size,
                                    dep_embedding_size=dep_embedding_size,
                                    conv1_out_channels=conv1_out_channels,
                                    conv2_out_channels=conv2_out_channels,
                                    conv3_out_channels=conv3_out_channels,
                                    conv1_length=conv1_length,
                                    conv2_length=conv2_length,
                                    conv3_length=conv3_length,
                                    target_class=target_class,
                                    num_node_features=num_node_features,
                                    hidden_channels=hidden_channels,
                                    device=device)
                                     
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

        for data in tqdm(dataset_train):
            text = data[0][0].clone().detach().to(self.device)
            if data[0][1][0]:
                mol1 = data[0][1][0].to(self.device)
            else:
                mol1 = None
            if data[0][1][1]:
                mol2 = data[0][1][1].to(self.device)
            else:
                mol2 = None
            label = data[1]
            label = self.convert_label_to_2d(label)
            
            i += 1
            out = self.model(text, mol1, mol2)
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
            for data in tqdm(dataset_test):
                text = data[0][0].clone().detach().to(self.device)
                if data[0][1][0]:
                    mol1 = data[0][1][0].to(self.device)
                else:
                    mol1 = None
                if data[0][1][1]:
                    mol2 = data[0][1][1].to(self.device)
                else:
                    mol2 = None
                label = torch.tensor([data[1]]).to(self.device)

                out = self.model(text, mol1, mol2)

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
        for epoch in tqdm(range(num_epochs)):
            running_loss = self.train_one_epoch(dataset_train)
            self.train_loss.append(running_loss)

            self.validate(dataset_test, 'val')
            print(f'Epoch: {epoch}, Train Loss: {self.train_loss[-1]}, Val Loss: {self.val_loss[-1]}, Val Micro F1: {self.val_micro_f1[-1]}')
            
    def log(self):
        f_false, f_adv, f_eff, f_mech, f_int = list(), list(), list(), list(), list()
        for x in self.val_f:
            f_false.append(x[0])
            f_adv.append(x[1])
            f_eff.append(x[2])
            f_mech.append(x[3])
            f_int.append(x[4])
            
        wandb.log(
            {
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "val_micro_f1": self.val_micro_f1,
                "val_f_false": f_false,
                "val_f_adv": f_adv,
                "val_f_eff": f_eff,
                "val_f_mech": f_mech,
                "val_f_int": f_int
            }
        )

        wandb.finish()