import torch
import torch.nn as nn
# from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import transformers
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoImageProcessor, ResNetForImageClassification

class Image_Only_Model(nn.Module):
    def __init__(self):
        super(Image_Only_Model, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 75 * 75, 512)
        self.fc2 = nn.Linear(512, 5)  # 5 output classes
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x1, x2):
        # Forward pass for first image
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, 64 * 75 * 75)  # Flatten
        
        # Forward pass for second image
        x2 = self.pool(F.relu(self.conv1(x2)))
        x2 = self.pool(F.relu(self.conv2(x2)))
        x2 = self.pool(F.relu(self.conv3(x2)))
        x2 = x2.view(-1, 64 * 75 * 75)  # Flatten
        
        # Concatenate the features from two images
        x = torch.cat((x1, x2), dim=1)
        
        # Fully connected layers with dropout
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x))
        return x

class Image_PreTrained_Model(nn.Module):
    def __init__(self, 
                 model_name="microsoft/resnet-50", 
                 dropout_prob=0.3,
                 output_dim=5):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetForImageClassification.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.classifier.out_features, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,x1,x2):
        inputs_1 = self.processor(x1, return_tensors="pt")
        inputs_2 = self.processor(x2, return_tensors="pt")
        with torch.no_grad():
            logits_1 = self.model(**inputs_1).logits
            logits_2 = self.model(**inputs_2).logits
        x = torch.cat((logits_1, logits_2), dim=-1)
        x = self.dropout(x)
        return F.softmax(self.classifier(x))