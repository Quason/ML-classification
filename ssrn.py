import sys

import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import random
from torch.utils.data import Dataset, DataLoader

# RF: train:test=2:8 accuracy=0.76
# Baseline: epoch=30 train:test=2:8 accuracy=0.51
# Conv1D: epoch=30 train:test=2:8 accuracy=0.7
# Net3DByHamida: epoch=30 train:test=2:8 accuracy=0.78

class Baseline(nn.Module):
    ''' baseline network: BP
    '''
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        
        self.classifier = nn.Sequential(
            nn.Linear(200, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, classes),
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, 200)
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


class Net3DByHamida(nn.Module):
    def __init__(self, classes, input_channels, patch_size, dilation=1):
        super().__init__()
        self.classes = classes
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 20, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(20),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(20, 35, kernel_size=(3,3,3), stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(35),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(35, 35, (3,1,1), dilation=dilation, stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(35),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(35, 35, (2,1,1), dilation=dilation, stride=(2,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(35),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, classes)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x


def train(net, trainloader):
    print('train...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(30):
        running_loss_sum = 0
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
        print('loss: %.3f' % (running_loss/len(trainloader)))
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


def myLoader3d(dataset, label, train_perc, patch_size=5):
    pad_size = int((patch_size-1)/2)
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
    net = Net3DByHamida(classes, input_channels=200, patch_size=5)
    train_loader, test_loader = myLoader3d(dataset, label, 0.2) # data split
    # train_loader, test_loader = myLoader1d(dataset, label, 0.2)
    print('train samples:%d, test samples:%d' % (len(train_loader),len(test_loader)))
    net = train(net, train_loader) # train
    test(net, test_loader)


if __name__ == '__main__':
    main()
