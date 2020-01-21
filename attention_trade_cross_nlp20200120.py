# -*- coding: utf-8 -*-


import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import RandomSampler
from torch.nn import functional as F
import pandas as pd

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

trade_feature_size=415
text_feature_size=768
mid_space_size=1024
margin=0.1

training_size=409600*9

import math
class GELU(torch.nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))



def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix



def AllPositivePairSelector(trade_features, bert_embeddings, labels,batch_size):

    for i in range(batch_size):
        for j in range(batch_size):
            if labels[i]!=labels[j]:
                try:
                    anchor=torch.cat((anchor,trade_features[i].unsqueeze(0)),dim=0)
                    positive_pairs=torch.cat((positive_pairs,bert_embeddings[i].unsqueeze(0)),dim=0)
                    negative_pairs=torch.cat((negative_pairs,bert_embeddings[j].unsqueeze(0)),dim=0)
                except:
                    anchor=trade_features[i].unsqueeze(0)
                    positive_pairs=bert_embeddings[i].unsqueeze(0)
                    negative_pairs=bert_embeddings[j].unsqueeze(0)
    
    return anchor, positive_pairs, negative_pairs


def HardNegativePairSelector(trade_features, bert_embeddings, labels,batch_size):
   
    distance_matrix = pdist(bert_embeddings)
    #print(distance_matrix.shape)
    for i in range(batch_size):
        for j in range(batch_size):
            if labels[i]!=labels[j]:
                try:
                    anchor=torch.cat((anchor,trade_features[i].unsqueeze(0)),dim=0)
                    positive_pairs=torch.cat((positive_pairs,bert_embeddings[i].unsqueeze(0)),dim=0)
                    negative_pairs=torch.cat((negative_pairs,bert_embeddings[j].unsqueeze(0)),dim=0)
                    hard_rank=torch.cat((hard_rank,distance_matrix[i,j].unsqueeze(0)),dim=0)
                except:
                    anchor=trade_features[i].unsqueeze(0)
                    positive_pairs=bert_embeddings[i].unsqueeze(0)
                    negative_pairs=bert_embeddings[j].unsqueeze(0)
                    hard_rank=distance_matrix[i,j].unsqueeze(0)
    #print(hard_rank.shape)
    sorted_x, indices = torch.sort(hard_rank)
# =============================================================================
#     print(a)
#     print(indices.shape)
#     print(anchor.shape)
# =============================================================================
    
    return anchor[indices[:batch_size*10]], positive_pairs[indices[:batch_size*10]], negative_pairs[indices[:batch_size*10]]


def HardNegativePairSelectorScale(trade_features, bert_embeddings, labels,batch_size,num_of_scale):
   
    distance_matrix = pdist(bert_embeddings)
    #print(distance_matrix.shape)
    for i in range(batch_size):
        for j in range(batch_size):
            if labels[i]!=labels[j]:
                try:
                    anchor=torch.cat((anchor,trade_features[i].unsqueeze(0)),dim=0)
                    positive_pairs=torch.cat((positive_pairs,bert_embeddings[i].unsqueeze(0)),dim=0)
                    negative_pairs=torch.cat((negative_pairs,bert_embeddings[j].unsqueeze(0)),dim=0)
                    hard_rank=torch.cat((hard_rank,distance_matrix[i,j].unsqueeze(0)),dim=0)
                except:
                    anchor=trade_features[i].unsqueeze(0)
                    positive_pairs=bert_embeddings[i].unsqueeze(0)
                    negative_pairs=bert_embeddings[j].unsqueeze(0)
                    hard_rank=distance_matrix[i,j].unsqueeze(0)
    #print(hard_rank.shape)
    sorted_x, indices = torch.sort(hard_rank)
# =============================================================================
#     print(a)
#     print(indices.shape)
#     print(anchor.shape)
# =============================================================================
    
    return anchor[indices[:batch_size*num_of_scale]], positive_pairs[indices[:batch_size*num_of_scale]], negative_pairs[indices[:batch_size*num_of_scale]]


class Double_attention_active(torch.nn.Module):
    expansion=4

    def __init__(self):
        super(Double_attention_active,self).__init__()
        self.bn0=torch.nn.BatchNorm1d(text_feature_size)
        self.conv1=torch.nn.Conv1d(1,1,1)
        self.fc1=torch.nn.Linear(text_feature_size, mid_space_size)
        self.bn1=torch.nn.BatchNorm1d(mid_space_size)
        #self.conv2=nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=stride, padding=padding, bias=False)
        #self.bn2=nn.BatchNorm1d(planes)
        #self.conv3=nn.Conv3d(planes, planes*4, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False)
        #self.bn3=nn.BatchNorm1d(planes*4)
        self.conv2=torch.nn.Conv1d(1,1,1)
        self.fc2=torch.nn.Linear(text_feature_size, mid_space_size)
        self.bn2=torch.nn.BatchNorm1d(mid_space_size)
        self.conv3=torch.nn.Conv1d(1,1,1)
        self.fc3=torch.nn.Linear(trade_feature_size, mid_space_size)
        self.conv4=torch.nn.Conv1d(1,1,1)
        self.bn3=torch.nn.BatchNorm1d(mid_space_size)
        self.fc4=torch.nn.Linear(mid_space_size, mid_space_size)
        self.fc_out1=torch.nn.Linear(mid_space_size, mid_space_size)
        self.fc_out1_second=torch.nn.Linear(mid_space_size, mid_space_size)
        self.bn4=torch.nn.BatchNorm1d(mid_space_size)
        
        self.bn00=torch.nn.BatchNorm1d(trade_feature_size)
        self.conv5=torch.nn.Conv1d(1,1,1)
        self.fc5=torch.nn.Linear(trade_feature_size, mid_space_size)
        self.bn5=torch.nn.BatchNorm1d(mid_space_size)
        #self.conv2=nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=stride, padding=padding, bias=False)
        #self.bn2=nn.BatchNorm1d(planes)
        #self.conv3=nn.Conv3d(planes, planes*4, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False)
        #self.bn3=nn.BatchNorm1d(planes*4)
        self.conv6=torch.nn.Conv1d(1,1,1)
        self.fc6=torch.nn.Linear(trade_feature_size, mid_space_size)
        self.bn6=torch.nn.BatchNorm1d(mid_space_size)
        self.conv7=torch.nn.Conv1d(1,1,1)
        self.fc7=torch.nn.Linear(text_feature_size, mid_space_size)
        self.bn7=torch.nn.BatchNorm1d(mid_space_size)
        self.conv8=torch.nn.Conv1d(1,1,1)
        self.fc8=torch.nn.Linear(mid_space_size, mid_space_size)
        self.fc_out2=torch.nn.Linear(mid_space_size, mid_space_size)
        self.fc_out2_second=torch.nn.Linear(mid_space_size, mid_space_size)
        self.bn8=torch.nn.BatchNorm1d(mid_space_size)
        self.active=GELU()
        #self.relu=torch.nn.functional.leaky_relu(inplace=True)
        #self.downsample=downsample
        #self.stride=stride
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        torch.nn.init.kaiming_normal_(self.fc6.weight)
        torch.nn.init.kaiming_normal_(self.fc7.weight)
        torch.nn.init.kaiming_normal_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.conv6.weight)
        torch.nn.init.kaiming_normal_(self.conv7.weight)
        torch.nn.init.kaiming_normal_(self.conv8.weight)
        torch.nn.init.kaiming_normal_(self.fc_out2.weight)
        torch.nn.init.kaiming_normal_(self.fc_out1.weight)
        torch.nn.init.kaiming_normal_(self.fc_out2_second.weight)
        torch.nn.init.kaiming_normal_(self.fc_out1_second.weight)
# =============================================================================
#         torch.nn.init.xavier_uniform_(self.fc1.bias)
#         torch.nn.init.kaiming_uniform_(self.fc2.bias)
#         torch.nn.init.kaiming_uniform_(self.fc3.bias)
#         torch.nn.init.kaiming_uniform_(self.fc4.bias)
#         torch.nn.init.kaiming_uniform_(self.fc5.bias)
#         torch.nn.init.kaiming_uniform_(self.fc6.bias)
#         torch.nn.init.kaiming_uniform_(self.fc7.bias)
#         torch.nn.init.kaiming_uniform_(self.fc8.bias)
# =============================================================================
        
        
    def forward(self,y,x,z,P):
        y=self.bn00(y).unsqueeze(1)
        x=self.bn0(x).unsqueeze(1)
        z=self.bn0(z).unsqueeze(1)
        
        out1=self.conv1(x)
        out1=self.fc1(out1.squeeze(1))
        #out1=self.bn1(out1)
        out1=self.active(out1)
        out2=self.conv2(x)
        out2=self.fc2(out2.squeeze(1))
        #out2=self.bn2(out2)
        out2=self.active(out2)
        out3=self.conv3(y)
        out3=self.fc3(out3.squeeze(1))
        #out3=self.bn3(out3)
        out3=self.active(out3)
        
        
        f = out2*out3
        f=self.active(f)
       # qn = torch.norm(f).detach()
        #f=f.shape[0]*f/qn
        f1 = F.softmax(f, dim=-1)
        f2 = out1*f1
        f2=self.conv4(f2.unsqueeze(1))
        f3 = self.fc4(f2.squeeze(1))
        f3 = self.fc_out1(f3)
        f3 = self.fc_out1_second(f3)
        
        
        
        z=self.conv1(z)
        out1_negative=self.fc1(z.squeeze(1))
       # out1_negative=self.bn1(out1_negative)
        out1_negative=self.active(out1_negative)
        z=self.conv2(z)
        out2_negative=self.fc2(z.squeeze(1))
        #out2_negative=self.bn2(out2_negative)
        out2_negative=self.active(out2_negative)
        #out3_negative=self.fc3(y)
        #out3_negative=self.bn3(out3_negative)
        #out3_negative=self.active(out3_negative)
        
        f_negative = out2_negative*out3
        f_negative=self.active(f_negative)
        #qnn = torch.norm(f_negative).detach()
        #f_negative=f_negative.shape[0]*f_negative/qnn
        f1_negative = F.softmax(f_negative, dim=-1)
        f2_negative = out1_negative*f1_negative
        f2_negative=self.conv4(f2_negative.unsqueeze(1))
        f3_negative = self.fc4(f2_negative.squeeze(1))
        f3_negative = self.fc_out1(f3_negative)
        f3_negative = self.fc_out1_second(f3_negative)
        
        out4=self.conv5(y)
        out4=self.fc5(out4.squeeze(1))
        #out4=self.bn5(out4)
        out4=self.active(out4)
        out5=self.conv6(y)
        out5=self.fc6(out5.squeeze(1))
        #out5=self.bn6(out5)
        out5=self.active(out5)
        out6=self.conv7(x)
        out6=self.fc7(out6.squeeze(1))
        #out6=self.bn7(out6)
        out6=self.active(out6)
        
        
        ff = out5*out6
        ff=self.active(ff)
        #qn = torch.norm(ff).detach()
        #ff=ff.shape[0]*ff/qn
        ff1 = F.softmax(ff, dim=-1)
        ff2 = out4*ff1
        ff2=self.conv8(ff2.unsqueeze(1))
        ff3 = self.fc8(ff2.squeeze(1))
        ff3 = self.fc_out2(ff3)
        ff3 = self.fc_out2_second(ff3)
        
        return F.normalize(ff3,p=2,dim=1),F.normalize(f3,p=2,dim=1),F.normalize(f3_negative,p=2,dim=1)



class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path,header=None)
        #self.labels = pd.get_dummies(self.data['emotion']).as_matrix()
        #self.height = 48
        #self.width = 48
        self.transform = transform
        #self.df=self.data.get_chunk(102400)
        #self.count=0
        
    def __getitem__(self, index):
        # This method should return only 1 sample and label 
        # (according to "index"), not the whole dataset
        # So probably something like this for you:
        #pixel_sequence = self.data['pixels'][index]
        #face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        #face = np.asarray(face).reshape(self.width, self.height)
        #face = cv2.resize(face.astype('uint8'), (self.width, self.height))
        #label = self.labels[index]
        #print(index)
        #if  idx==0:
        #    self.df=self.data.get_chunk(102400)
           
        #temp_df=self.data.get_chunk(1)
        #if self.train:
        trade_features=self.data.iloc[index:index+1, :415].values[0]
        pos_t=self.data.iloc[index:index+1, 415:415+768].values[0]
        #neg_t=self.df.iloc[idx:idx+1, 415+768:415+768*2].values[0]
        pos_label=self.data.iloc[index:index+1, 415+768:415+768+1].values[0]
        #fsp_key=self.data.iloc[index:index+1, 415+768+1:415+768+2].values[0]
        #neg_label=self.df.iloc[idx:idx+1, 415+768*2+11:415+768*2+11*2].values[0]
        #img, target = , self.data.iloc[index:index+1, :1].values[0].tolist()[0]
        #else:
        #    img, target = self.data.iloc[index:index + 1, :].values.reshape(28, 28).astype(np.uint8), \
        #                  -1
        

        return trade_features,pos_t,pos_label#,fsp_key

    def __len__(self):
        return len(self.data)


class TripletLoss(torch.nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

model=Double_attention_active()
model.to(device)
loss_fn = TripletLoss(margin)
lr = 1e-2
from torch.optim import lr_scheduler
import torch.optim as optim
optimizer=torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
#optimizer=torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99),weight_decay=0.0001,amsgrad=True)
#scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
#optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
#optimizer=torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0.01, momentum=0.9, centered=False)
scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=3, threshold=0.0001, threshold_mode='rel', cooldown=2, min_lr=0.000001, eps=1e-08)

#my_path='E:\\pytorch_bigtest1.csv'
my_path='./pytorch_bigtest_20200110v21.csv'
dataset = CustomDatasetFromCSV(my_path)
training_batch_size = 64
validation_batch_size=100
validation_split = .1
shuffle_dataset = True
random_seed= 672

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
sampler = RandomSampler(dataset)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=training_batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=validation_batch_size,
                                                sampler=valid_sampler)


#train_loader_no_validation=torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=sampler,num_workers=4)
# Usage Example:
num_epochs = 3
# =============================================================================
# for epoch in range(num_epochs):
#     # Train:   
#     for batch_index, (trade_features,pos_t,neg_t,pos_label,neg_label) in enumerate(train_loader):
#         print('For index '+str(batch_index)+': We are here!')
# =============================================================================


criterion = torch.nn.TripletMarginLoss(margin)
criterion_valid01 = torch.nn.TripletMarginLoss(0.1)
criterion_valid02 = torch.nn.TripletMarginLoss(0.2)
criterion_valid05 = torch.nn.TripletMarginLoss(0.5)
losses = []
reg_loss=[]
log_loss=[]
log_acc01=[]
log_acc02=[]
log_acc05=[]
total_loss = 0
log_interval =2
import pickle
with open('./test.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
                tt1, tt2, tt3= pickle.load(f)
tt1=np.array(tt1)
tt4=tt1[:,1:]
tt1=np.array(tt4)
tt2=np.array(tt2)
tt3=np.array(tt3)
validation_iter=iter(validation_loader)
val_loss_low=0
import random
import matplotlib.pyplot as plt
for epoch in range(num_epochs):
    if epoch>=1:
        margin=0.2
        random_seed=random.randint(1, 1000) 
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        sampler = RandomSampler(dataset)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=training_batch_size, 
                                                   sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=validation_batch_size,
                                                sampler=valid_sampler)
    for batch_index, (x1,x2,y1) in enumerate(train_loader):
        
        trade_features,pos_t,neg_t=HardNegativePairSelectorScale(x1.to(device),x2.to(device),y1.to(device),y1.shape[0],8)
    
        optimizer.zero_grad()

        anchor,positive,negative = model(trade_features.float(),pos_t.float(),neg_t.float(),0.1)
        #print(torch.norm(anchor,dim=1)[0])
        #print(torch.norm(positive,dim=1)[0])
        #print(torch.norm(negative,dim=1)[0])
        #if type(outputs) not in (tuple, list):
        #    outputs = (outputs,)
    
        #loss_inputs = outputs
        #if target is not None:
        #    target = (target,)
        #    loss_inputs += target
        
        #distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        #distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        #losses_output = -F.relu(distance_positive - distance_negative + margin)
        loss=criterion(anchor,positive,negative)
        regularization_loss=0
        for param in model.parameters():
            regularization_loss+=torch.sum(param.pow(2))
        
        #loss.mean() 
        #if size_average else losses.sum()
    
        #loss_outputs = loss_fn(*loss_inputs)
        #loss = losses_output #if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        reg_loss.append(0.01*regularization_loss.item())
        #total_loss += loss.item()
        #ll=loss+F.relu(regularization_loss-50000)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())  
        #for metric in metrics:
        #    metric(outputs, target, loss_outputs)
        
        #acc=sum(loss.cpu().detach().numpy()>=0.1)
        if batch_index % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                batch_index * len(trade_features[0]), len(train_loader.dataset),
                100. * batch_index / len(train_loader), np.mean(losses))
            message += '\t Regularization Loss: {:.6f}'.format(
               np.mean(reg_loss))
            #for metric in metrics:
            with torch.no_grad():
        #for i in range(int(np.shape(tt1)[0]/batch_size)):
                correct=0
                total=0
                try:
                    x1_test,x2_test,y1_test=next(validation_iter)
                except:
                    validation_iter=iter(validation_loader)
                    x1_test,x2_test,y1_test=next(validation_iter)
                    #trade_features_test=tt1
                    #pos_txt_test=tt2
                    #neg_txt_test=tt3
                trade_features_test,pos_t_test,neg_t_test=HardNegativePairSelector(x1_test.to(device),x2_test.to(device),y1_test.to(device),y1_test.shape[0])
                anchor_test,positive_test,negative_test =  model(trade_features_test.float(),pos_t_test.float(),neg_t_test.float(),0)
# =============================================================================
#                 distance_positive = (anchor_test - positive_test).pow(2).sum(1)/((anchor_test.pow(2).sum(1)*positive_test.pow(2).sum(1)).pow(.5))  # .pow(.5)
#                 distance_negative = (anchor_test - negative_test).pow(2).sum(1)/((anchor_test.pow(2).sum(1)*negative_test.pow(2).sum(1)).pow(.5))  # .pow(.5)
#                 losses_output = distance_positive - distance_negative + 0.5
# =============================================================================
                val_loss01=criterion_valid01(anchor_test,positive_test,negative_test)
                val_loss02=criterion_valid02(anchor_test,positive_test,negative_test)
                val_loss05=criterion_valid05(anchor_test,positive_test,negative_test)
                message += '\t{}: {:.6f}'.format('Val Loss',  val_loss01.item())
                print(message)
                log_loss.append(np.mean(losses))
                log_acc01.append(val_loss01.item())
                log_acc02.append(val_loss02.item())
                log_acc05.append(val_loss05.item())
# =============================================================================
#                 elif val_loss.item()<0.01:
#                     val_loss_low+=1
#                     if val_loss_low>=5:
#                         torch.save(model, 'model_margin_'+str(int(margin*10))+'.pkl')
#                         margin+=0.1
#                         if margin>1:
#                             break;
#                         criterion = torch.nn.TripletMarginLoss(margin)
#                         print('The margin is set to '+str(margin))
#                         val_loss_low=0
#                 else:
#                     val_loss_low=0
# =============================================================================
            losses = []
            
            #if val_loss
            #reg_loss=[]
    #total_loss /= (batch_index + 1)