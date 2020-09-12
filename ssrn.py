import sys

import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import random
from torch.utils.data import Dataset, DataLoader

class Net(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(24, classes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(classes, classes),
            nn.ReLU(inplace=True),
            nn.Linear(classes, classes),
        )

    def forward(self, x):
        batch_size = x.size()[0]
        # spectral
        x = nn.Conv3d(1, 24, kernel_size=(7,1,1), stride=(2,1,1))(x)
        x = nn.BatchNorm3d(24)(x)
        x = nn.ReLU(inplace=True)(x)
        depth = np.shape(x)[2]
        x = nn.Conv3d(24, 128, kernel_size=(depth,1,1), stride=(1,1,1))(x)
        x = nn.BatchNorm3d(128)(x)
        x = nn.ReLU(inplace=True)(x)
        x = x.reshape(batch_size, 1, 128, 7, 7)
        # spatial
        x = nn.Conv3d(1, 24, kernel_size=(128,3,3), stride=(1,1,1))(x)
        x = nn.BatchNorm3d(24)(x)
        x = nn.ReLU(inplace=True)(x)
        x = x.reshape(batch_size, 1, 24, 5, 5)
        # avgpool and full-connection
        x = nn.AvgPool3d(kernel_size=(1,5,5))(x)
        x = x.view(batch_size, 24)
        x = self.classifier(x)
        return x


def train(net, trainloader, device):
    # net.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # train
    for epoch in range(1):
        print('epoch %d...' % (epoch+1))
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            # inputs = inputs.to(device=device)
            # labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def load_data(src_fn):
    data = scio.loadmat(src_fn)
    for key in data.keys():
        if key[:2] != '__':
            return data[key]
    return None


def main():
    dataset = load_data('./dataset/Indian_pines_corrected.mat')
    dataset = dataset.astype(float)
    dataset = np.swapaxes(dataset, 0, 2) # x,y,x --> z,y,z
    dataset = np.swapaxes(dataset, 1, 2) # z,y,x --> z,x,y
    label = load_data('./dataset/Indian_pines_gt.mat')
    label = label.astype(int)
    classes = np.max(label)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(dataset.shape, label.shape)
    dst_fn = 'res_ssrn.tif'
    # dataset split
    pad_size = 3
    batch_size_train = 4
    batch_size_test = 4
    dataset = np.pad(dataset, ((0,0),(pad_size,pad_size),(pad_size,pad_size)))
    label = np.pad(label, ((pad_size,pad_size),(pad_size,pad_size)))
    dsize = dataset.shape
    valid_index = []
    for i in range(pad_size, dsize[1]-pad_size):
        for j in range(pad_size, dsize[2]-pad_size):
            if label[i,j] != 0:
                valid_index.append([i,j])
    random.shuffle(valid_index)
    train_size = int(len(valid_index) * 0.1)
    train_index = valid_index[0:train_size]
    test_index = valid_index[train_size:]
    net = Net(classes)
    train_data = []
    train_label = []
    for item in train_index:
        t_data = dataset[
            :, item[0]-pad_size:item[0]+pad_size+1,
            item[1]-pad_size:item[1]+pad_size+1]
        t_data = t_data.reshape(1, dsize[0], pad_size*2+1, pad_size*2+1)
        t_data = torch.from_numpy(t_data)
        t_data = t_data.type(torch.FloatTensor)
        t_label = torch.tensor(label[item[0], item[1]]-1)
        train_data.append(t_data)
        train_label.append(t_label.type(torch.LongTensor))
    trainloader = torch.utils.data.DataLoader(
        MyDataset(train_data,train_label), batch_size=batch_size_train, shuffle=True)
    train(net, trainloader, device)
    
    # a = torch.rand(1,1,200,7,7)
    # net = Net(2)
    # y = net.forward(a)
    # print(y)

if __name__ == '__main__':
    main()
