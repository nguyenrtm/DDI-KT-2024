import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassF1Score
import numpy as np
import wandb

from ddi_kt_2024.multimodal.model import MultimodalModel

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
                 hidden_channels: int = 512,
                 w_false: float = 21580 / 17759,
                 w_advice: float = 21580 / 826,
                 w_effect: float = 21580 / 1687,
                 w_mechanism: float = 21580 / 1319,
                 w_int: float = 21580 / 189,
                 lr: float = 0.0001,
                 weight_decay: float = 1e-4,
                 text_model: str = 'bert',
                 modal: str = 'multimodal',
                 gnn_option: str = 'GATVCONV2',
                 num_layers_gnn: str = 3,
                 readout_option: str = 'global_max_pool',
                 activation_function: str = 'relu',
                 log: bool = True,
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
                                    text_model=text_model,
                                    modal=modal,
                                    gnn_option=gnn_option,
                                    num_layers_gnn=num_layers_gnn,
                                    readout_option=readout_option,
                                    activation_function=activation_function,
                                    device=device).to(device)
                                     
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight)
        self.device = device
        self.train_loss = list()
        self.train_f = list()
        self.val_loss = list()
        self.val_f = list()
        self.val_micro_f1 = list()
        self.confusion_matrix = list()
        self.log = log
        
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
    
    def train_one_epoch(self, 
                        train_loader_text, 
                        train_loader_mol1, 
                        train_loader_mol2,
                        train_loader_mol1_bert,
                        train_loader_mol2_bert):
        
        running_loss = 0.
        i = 0

        for ((a, batch_label), b, c, d, e) in zip(train_loader_text, 
                                            train_loader_mol1, 
                                            train_loader_mol2,
                                            train_loader_mol1_bert,
                                            train_loader_mol2_bert):
            
            text = a.clone().detach().to(self.device)
            mol1 = b.to(self.device)
            mol2 = c.to(self.device)
            mol1_bert = d.to(self.device)
            mol2_bert = e.to(self.device)
            batch_label = batch_label.clone().detach().to(self.device)

            batch_label = self.convert_label_to_2d(batch_label)
            
            i += 1

            out = self.model(text, mol1, mol2, mol1_bert, mol2_bert)
            self.optimizer.zero_grad()
            loss = self.criterion(out, batch_label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
        self.train_loss.append(running_loss)
        return running_loss
    
    def validate(self, 
                 val_loader_text, 
                 val_loader_mol1, 
                 val_loader_mol2, 
                 val_loader_mol1_bert, 
                 val_loader_mol2_bert, 
                 option):
        running_loss = 0.
        predictions = torch.tensor([]).to(self.device)
        labels = torch.tensor([]).to(self.device)
        
        with torch.no_grad():
            for ((a, batch_label), b, c, d, e) in zip(val_loader_text, 
                                                val_loader_mol1, 
                                                val_loader_mol2,
                                                val_loader_mol1_bert, 
                                                val_loader_mol2_bert):
                text = a.clone().detach().to(self.device)
                mol1 = b.to(self.device)
                mol2 = c.to(self.device)
                mol1_bert = d.to(self.device)
                mol2_bert = e.to(self.device)

                batch_label = batch_label.clone().detach().to(self.device)

                out = self.model(text, mol1, mol2, mol1_bert, mol2_bert)

                batch_label_for_loss = self.convert_label_to_2d(batch_label)
                loss = self.criterion(out, batch_label_for_loss)
                running_loss += loss.item()

                batch_prediction = torch.argmax(out, dim=1)

                predictions = torch.cat((predictions, batch_prediction))
                labels = torch.cat((labels, batch_label))
        
        labels = labels.squeeze()
        true_pred = []
        for i in range(len(labels)):
            if labels[i].cpu() == 1:
                true_pred.append(i)
                
        cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[0, 1, 2, 3, 4])
        _micro_f1 = self.micro_f1(cm)

        if self.log == True:
            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=labels.cpu().numpy(), preds=predictions.cpu().numpy(),
                            class_names=['false', 'advise', 'effect', 'mechanism', 'int'])})

        f = MulticlassF1Score(num_classes=5, average=None).to(self.device)(predictions, labels)
        
        if option == 'train':
            self.train_loss.append(running_loss)
            self.train_f.append(f)
        elif option == 'val':
            self.confusion_matrix.append(cm)
            self.val_loss.append(running_loss)
            self.val_f.append(f)
            self.val_micro_f1.append(_micro_f1)
    
    def micro_f1(self, cm):
        tp = cm[1][1] + cm[2][2] + cm[3][3] + cm[4][4]
        fp = np.sum(cm[:,1]) + np.sum(cm[:,2]) + np.sum(cm[:,3]) + np.sum(cm[:,4]) - tp
        fn = np.sum(cm[1,:]) + np.sum(cm[3,:]) + np.sum(cm[3,:]) + np.sum(cm[4,:]) - tp
        micro_f1 = tp / (tp + 1/2*(fp + fn))
        return micro_f1
        
    def train(self, train_loader_text, train_loader_mol1, train_loader_mol2, train_loader_mol1_bert, train_loader_mol2_bert,
                    val_loader_text, val_loader_mol1, val_loader_mol2, val_loader_mol1_bert, val_loader_mol2_bert, num_epochs):
        for epoch in tqdm(range(num_epochs), desc='Training...'):
            print(f"Epoch {epoch + 1} training...")
            running_loss = self.train_one_epoch(train_loader_text, 
                                                train_loader_mol1, 
                                                train_loader_mol2,
                                                train_loader_mol1_bert, 
                                                train_loader_mol2_bert)
            
            self.train_loss.append(running_loss)

            self.validate(val_loader_text, 
                          val_loader_mol1, 
                          val_loader_mol2, 
                          val_loader_mol1_bert, 
                          val_loader_mol2_bert,
                          'val')
            
            if self.log == True:
                self.log_wandb()
            
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
            }
        )