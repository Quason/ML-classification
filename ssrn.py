import sys

import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import random
from torch.utils.data import Dataset, DataLoader

class NetSS(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.spectral_block = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(7,1,1), stride=(2,1,1)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.spectral_block2 = nn.Sequential(
            nn.Conv3d(24, 24, kernel_size=(7,1,1), stride=(1,1,1), padding=(3,0,0)),
            nn.BatchNorm3d(24),
        )
        depth = 97
        self.reshape_block = nn.Sequential(
            nn.Conv3d(24, 128, kernel_size=(depth,1,1), stride=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.spatial_block = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(128,3,3), stride=(1,1,1)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.spatial_block2 = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(24,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(24),
        )
        self.pool1 = nn.AvgPool3d(kernel_size=(1,5,5))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(24, classes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(classes, classes),
            nn.ReLU(inplace=True),
            nn.Linear(classes, classes),
        )
        self.relu = nn.ReLU(inplace=True)

    def _res_block_spectral(self, x):
        y = self.spectral_block2(x)
        y = self.relu(y)
        y = self.spectral_block2(y)
        y += x
        y = self.relu(y)
        return y

    def _res_block_spatial(self, x):
        y = self.spatial_block2(x)
        y = y.reshape(y.shape[0], y.shape[2], y.shape[1], y.shape[3], y.shape[4])
        y = self.relu(y)
        y = self.spatial_block2(y)
        y = y.reshape(y.shape[0], y.shape[2], y.shape[1], y.shape[3], y.shape[4])
        y += x
        y = self.relu(y)
        return y

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.spectral_block(x) # spectral (x -> 97*7*7, 24)
        x = self._res_block_spectral(x) # residual block (97*7*7, 24)
        x = self.reshape_block(x) # reshape (1*7*7, 128)
        x = x.reshape(batch_size, 1, 128, 7, 7) # (128*7*7)
        x = self.spatial_block(x)  # spatial (1*5*5, 24)
        x = x.reshape(batch_size, 1, 24, 5, 5) # (24,5,5)
        x = self._res_block_spatial(x) # (24,5,5)
        # avgpool and full-connection
        x = self.pool1(x)
        x = x.view(batch_size, 24)
        x = self.classifier(x)
        return x


class NetSS2(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.cnov3d_block1 = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(7,3,3), stride=(2,1,1), padding=(3,1,1)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.cnov3d_block2 = nn.Sequential(
            nn.Conv3d(24, 24, kernel_size=(7,3,3), stride=(1,1,1), padding=(3,1,1)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.AvgPool3d(kernel_size=(2,2,2))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(24*25, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.cnov3d_block1(x) # (x -> 100*7*7)
        x = self.pool1(x) # (50*3*3)
        x = self.cnov3d_block2(x) # (50*3*3)
        x = self.pool1(x) # (25*1*1)
        x = x.view(batch_size, 24*25)
        x = self.classifier(x)
        return x


class Net1D(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.features = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=23), # (178, 20)
            nn.BatchNorm1d(20),
            nn.MaxPool1d(kernel_size=5), # (35, 20)
            nn.Tanh(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(20*35, 100),
            nn.Tanh(),
            nn.Linear(100, classes),
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.features(x)
        x = x.view(batch_size, 20*35)
        x = self.classifier(x)
        return x


class NetBP(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        
        self.classifier = nn.Sequential(
            nn.Linear(200, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, classes),
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, 200)
        x = self.classifier(x)
        return x


def train(net, trainloader):
    print('train...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(30):
        print('epoch %d...' % (epoch+1))
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %d] loss: %.3f' % (epoch+1, i+1, running_loss/100))
                running_loss = 0.0
    return net


def test(net, testloader):
    print('test...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    running_loss = 0.0
    predict_true = 0.0
    predict_sum = 0.0
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        predict_true += int(torch.sum(predicted==labels))
        predict_sum += len(labels)
    print('accuracy: %.2f' % (predict_true/predict_sum))


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


def myLoader(dataset, label, train_perc):
    pad_size = 3
    batch_size_train = 4
    batch_size_test = 4
    dataset = np.pad(dataset, ((0,0),(pad_size,pad_size),(pad_size,pad_size)))
    label = np.pad(label, ((pad_size,pad_size),(pad_size,pad_size)))
    dsize = dataset.shape
    valid_index = []
    for i in range(pad_size, dsize[1]-pad_size):
        for j in range(pad_size, dsize[2]-pad_size):
            valid_index.append([i,j])
    random.shuffle(valid_index)
    # train dataset
    train_size = int(len(valid_index) * train_perc)
    train_index = valid_index[0:train_size]
    train_data = []
    train_label = []
    for item in train_index:
        t_data = dataset[
            :, item[0]-pad_size:item[0]+pad_size+1,
            item[1]-pad_size:item[1]+pad_size+1]
        t_data = t_data.reshape(1, dsize[0], pad_size*2+1, pad_size*2+1)
        t_data = torch.from_numpy(t_data)
        t_data = t_data.type(torch.FloatTensor)
        t_label = torch.tensor(label[item[0], item[1]])
        train_data.append(t_data)
        train_label.append(t_label.type(torch.LongTensor))
    trainloader = torch.utils.data.DataLoader(
        MyDataset(train_data,train_label), batch_size=batch_size_train, shuffle=True)
    # test dataset
    test_index = valid_index[train_size:]
    test_data = []
    test_label = []
    for item in test_index:
        t_data = dataset[
            :, item[0]-pad_size:item[0]+pad_size+1,
            item[1]-pad_size:item[1]+pad_size+1]
        t_data = t_data.reshape(1, dsize[0], pad_size*2+1, pad_size*2+1)
        t_data = torch.from_numpy(t_data)
        t_data = t_data.type(torch.FloatTensor)
        t_label = torch.tensor(label[item[0], item[1]])
        test_data.append(t_data)
        test_label.append(t_label.type(torch.LongTensor))
    testloader = torch.utils.data.DataLoader(
        MyDataset(test_data,test_label), batch_size=batch_size_test, shuffle=True)
    return trainloader, testloader


def myLoader1d(dataset, label, train_perc):
    batch_size_train = 4
    batch_size_test = 4
    dsize = dataset.shape
    valid_index = []
    for i in range(dsize[1]):
        for j in range(dsize[2]):
            valid_index.append([i,j])
    random.shuffle(valid_index)
    # train dataset
    train_size = int(len(valid_index) * train_perc)
    train_index = valid_index[0:train_size]
    train_data = []
    train_label = []
    for item in train_index:
        t_data = dataset[:, item[0], item[1]]
        t_data = t_data.reshape(1, dsize[0])
        t_data = torch.from_numpy(t_data)
        t_data = t_data.type(torch.FloatTensor)
        t_label = torch.tensor(label[item[0], item[1]])
        train_data.append(t_data)
        train_label.append(t_label.type(torch.LongTensor))
    trainloader = torch.utils.data.DataLoader(
        MyDataset(train_data,train_label), batch_size=batch_size_train, shuffle=True)
    # test dataset
    test_index = valid_index[train_size:]
    test_data = []
    test_label = []
    for item in test_index:
        t_data = dataset[:, item[0], item[1]]
        t_data = t_data.reshape(1, dsize[0])
        t_data = torch.from_numpy(t_data)
        t_data = t_data.type(torch.FloatTensor)
        t_label = torch.tensor(label[item[0], item[1]])
        test_data.append(t_data)
        test_label.append(t_label.type(torch.LongTensor))
    testloader = torch.utils.data.DataLoader(
        MyDataset(test_data,test_label), batch_size=batch_size_test, shuffle=True)
    return trainloader, testloader


def main():
    dataset = load_data('./dataset/Indian_pines_corrected.mat')
    dataset = dataset.astype(float)
    dataset = np.swapaxes(dataset, 0, 2) # x,y,x --> z,y,z
    dataset = np.swapaxes(dataset, 1, 2) # z,y,x --> z,x,y
    label = load_data('./dataset/Indian_pines_gt.mat')
    label = label.astype(int)
    classes = np.max(label) + 1
    dst_fn = 'res_ssrn.tif'
    net = Net1D(classes)
    # train_loader, test_loader = myLoader(dataset, label, 0.5) # data split
    train_loader, test_loader = myLoader1d(dataset, label, 0.2)
    print('train samples:%d, test samples:%d' % (len(train_loader),len(test_loader)))
    net = train(net, train_loader) # train
    test(net, test_loader)


if __name__ == '__main__':
    main()
