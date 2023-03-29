from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification, AutoImageProcessor, ResNetForImageClassification
from torch.utils.data import Dataset,TensorDataset, DataLoader
import requests
# from optimum.intel.openvino import OVModelForImageClassification
from tqdm import tqdm
import random
import shutil
import os 
import sys
sys.path.append('../')
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
# from datasets import load_dataset
from task2vec_embed import dataset
from autoencoder import AE
from glob import glob

def load_np(folder_dir,dataset_name,model_name):
    my_x = np.load(os.path.join(folder_dir,dataset_name,model_name+'_feature.npy'))
    my_y = np.load(os.path.join(folder_dir,dataset_name,model_name+'_label.npy'))
    return torch.from_numpy(my_x),torch.from_numpy(my_y)

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets #torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            # x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)



def train(dataloader,input_shape,output_shape,epochs=5,device='cuda'):
    model = AE(input_shape=input_shape,output_shape=output_shape).to('cuda')
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # mean-squared error loss
    criterion = nn.MSELoss()
    # train auto-encoder
    for epoch in range(epochs):
        # auto_encoder,loss = train(auto_encoder,last_hidden_layer,optimizer,criterion,input_shape)
        loss = 0
        for batch_features, _ in tqdm(dataloader):
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            num_features = batch_features.shape[0]
            batch_features = batch_features.view(-1, input_shape).to(device)
            # print(f'batch_features.shape: {batch_features.shape}')
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
                    
            # compute reconstructions
            outputs = model(batch_features)
            # print(f'outputs.shape: {outputs.shape}')
            # compute training reconstruction loss
            train_loss =  criterion(outputs, batch_features)
            # train_loss.requires_grad = True
            # print(f'train_loss: {train_loss}')
            # compute accumulated gradients
            train_loss.backward()
                    
            # perform parameter update based on current gradients
            optimizer.step()
                    
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
                
            # compute the epoch training loss
            loss = loss / num_features

            # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    return model

def save_feature(model,dataloader,model_name,dataset_name):
    features = ''
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            # print(f'len of a batch: {len(y)}')
            feature = model(x.cuda(),train=False)
            if features == '':
                print(f'x.shape:{x.shape}')
                features = torch.zeros(1,feature.shape[1]).to('cuda')
            features = torch.cat((features,feature),0)
    features = features.cpu().detach().numpy()
    features = features[1:]
    save_dir = os.path.join('./feature', dataset_name)
    np.save(os.path.join(save_dir, model_name + '.npy'), features)
    print(f'model feature saved in {model_name}.npy')

def main():
    DIM_EMD = 1024
    folder_dir = './feature'
    df = pd.read_csv('models.csv')
    dataset_names = os.listdir(folder_dir)
    for dataset_name in dataset_names:
        print(f'dataset_name: {dataset_name}')
        # npy_files = glob(os.path.join(folder_dir,dataset_name))
        # load feature and labels from .npy file
        df_dataset = df.loc[df['dataset']==dataset_name]
        # print(df_dataset)
        for i, row in df_dataset.iterrows():
            model_name = row['model'].replace('/','_')
            try:
                data, targets = load_np(folder_dir,dataset_name,model_name)
            except Exception as e: 
                print(e)
                continue
            print(f"input shape: {data[0].shape[0]}, {row['dimension']}")
            if row['dimension'] == DIM_EMD:
                src=os.path.join(folder_dir,dataset_name,model_name+'_feature.npy')
                dst=os.path.join(folder_dir,dataset_name,model_name+'.npy')
                shutil.copy(src,dst)
                print(f'dimension is the same. Feature is saved in {model_name}.npy')
                continue
            # retrieve dataloader
            ds = MyDataset(data, targets)
            dataloader = DataLoader(
                            ds,
                            batch_size=64, # may need to reduce this depending on your GPU 
                            num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True
                        )
            encoder = train(dataloader,data[0].shape[0],output_shape=DIM_EMD,epochs=5,device='cuda')
            torch.save(encoder,f'./weights/{model_name}_auto_encoder.pth')
            save_feature(encoder,dataloader,model_name,dataset_name)

if __name__ == "__main__":
    main()