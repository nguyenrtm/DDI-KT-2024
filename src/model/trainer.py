import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torchmetrics.classification import F1Score, Precision, Recall, MulticlassF1Score

from model.model import Model

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
                 lr: float = 0.0001,
                 weight_decay: float = 1e-4,
                 device='cpu'):
        
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
                
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.train_loss = list()
        self.train_f = list()
        self.train_f_micro = list()
        self.train_p_micro = list()
        self.train_r_micro = list()
        self.train_f_macro = list()
        self.train_p_macro = list()
        self.train_r_macro = list()
        self.val_loss = list()
        self.val_f = list()
        self.val_f_micro = list()
        self.val_p_micro = list()
        self.val_r_micro = list()
        self.val_f_macro = list()
        self.val_p_macro = list()
        self.valn_r_macro = list()
        
    def convert_label_to_2d(self, batch_label):
        i = 0
        for label in batch_label:
            i += 1
            tmp = torch.zeros((5))
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
            batch_data = torch.tensor(batch_data).to(self.device)
            batch_label = torch.tensor(batch_label).to(self.device)
            batch_label = self.convert_label_to_2d(batch_label)
            i += 1
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss

    
    def validate(self, validation_loader, option):
        running_loss = 0.
        predictions = torch.tensor([]).to(self.device)
        labels = torch.tensor([]).to(self.device)
        
        with torch.no_grad():
            for batch_data, batch_label in validation_loader:
                batch_data = torch.tensor(batch_data).to(self.device)
                batch_label = torch.tensor(batch_label).to(self.device)
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

        f_micro = F1Score(task="multiclass", average='micro', num_classes=5).to(self.device)(predictions, labels)
        p_micro = Precision(task="multiclass", average='micro', num_classes=5).to(self.device)(predictions, labels)
        r_micro = Recall(task="multiclass", average='micro', num_classes=5).to(self.device)(predictions, labels)
        f_macro = F1Score(task="multiclass", average='macro', num_classes=5).to(self.device)(predictions, labels)
        p_macro = Precision(task="multiclass", average='macro', num_classes=5).to(self.device)(predictions, labels)
        r_macro = Recall(task="multiclass", average='macro', num_classes=5).to(self.device)(predictions, labels)
        f = MulticlassF1Score(num_classes=5, average=None).to(self.device)(predictions, labels)
        
        if option == 'train':
            self.train_loss.append(running_loss)
            self.train_f.append(f.item())
            self.train_f_micro.append(f_micro.item())
            self.train_p_micro.append(p_micro.item())
            self.train_r_micro.append(r_micro.item())
            self.train_f_macro.append(f_macro.item())
            self.train_p_macro.append(p_macro.item())
            self.train_r_macro.append(r_macro.item())
        elif option == 'val':
            self.val_loss.append(running_loss)
            self.val_f.append(f.item())
            self.val_f_micro.append(f_micro.item())
            self.val_p_micro.append(p_micro.item())
            self.val_r_micro.append(r_micro.item())
            self.val_f_macro.append(f_macro.item())
            self.val_p_macro.append(p_macro.item())
            self.val_r_macro.append(r_macro.item())
    
    def train(self, training_loader, validation_loader, num_epochs):
        loss = list()

        for epoch in tqdm(range(num_epochs)):
            running_loss = self.train_one_epoch(training_loader)
            loss.append(running_loss)
            
            pred_train = self.validate(training_loader, 'train')
            pred_val = self.validate(validation_loader, 'val')
        return pred_train, pred_val