import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# read data
def read_data(num_class):
    # train data
    train_data_path = 'data/train.csv'
    train_raw_data = pd.read_csv(train_data_path, dtype=np.float32)
    train_data = train_raw_data.loc[:, train_raw_data.columns!="label"].values / 255.
    train_label = train_raw_data.label.values
    x_train, x_validation, y_train, y_validation = train_test_split(train_data, train_label, test_size=0.1, random_state=9527)

    # reshape
    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_validation = x_validation.reshape(len(x_validation), 1, 28, 28)

    # to Tensor
    x_train = torch.from_numpy(x_train)
    x_validation = torch.from_numpy(x_validation)
    y_train = torch.from_numpy(y_train)
    y_validation = torch.from_numpy(y_validation)

    #test data
    test_data_path = 'data/test.csv'
    test_raw_data = pd.read_csv(test_data_path, dtype=np.float32)
    test_data = test_raw_data.values / 255.
    x_test = torch.from_numpy(test_data.reshape(len(test_data), 1, 28, 28))

    return x_train, x_validation, x_test, y_train, y_validation

# model
class model(nn.Module):
    def __init__(self, output_dims):
        super(model, self).__init__()
        self.output_dims = output_dims
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=self.output_dims)
    def forward(self, input):
        features = self.feature(input)
        features = features.view(features.size(0), -1)
        output = self.fc1(features)
        output = self.fc2(output)
        return output

if __name__ == '__main__':
    batch_size = 600
    num_epochs = 20
    lr = 0.1
    num_class = 10

    # read data
    x_train, x_validation, x_test, y_train, y_validation = read_data(num_class)

    # dataset
    train = TensorDataset(x_train, y_train.long())
    val = TensorDataset(x_validation, y_validation.long())

    # data loader
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
    val_loader = DataLoader(val, batch_size = batch_size, shuffle = False)

    num_train = len(train)
    num_val = len(val)
    num_test = len(x_test)
    print(num_train, num_val, num_test)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model(num_class).to(device)
    print(net)

    '''
    plt.imshow(x_train[10].reshape(28, 28))
    plt.axis("off")
    plt.title(str(y_train[10]))
    plt.savefig('graph.png')
    plt.show()
    '''

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss = 0.
        train_accuracy = 0.
        val_accuracy = 0.
        for index, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            predict = net(train_x)
            loss = criterion(predict, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            #print(index, loss)
            train_accuracy += (torch.argmax(predict, 1).int()==train_y.int()).sum().item() * 1.0 / num_train


        for index, (val_x, val_y) in enumerate(val_loader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            predict = net(val_x)
            val_accuracy += (torch.argmax(predict, 1).int()==val_y.int()).sum().item() * 1.0 / num_val

        print('epoch: %d, train loss: %.4f, train accuracy: %.4f, val accuracy: %.4f' % (epoch, train_loss, train_accuracy, val_accuracy))

    test_y = []
    num_iter = (num_test + batch_size - 1)//batch_size
    for iterator in range(num_iter):
        start = iterator * batch_size
        end = start + batch_size
        if end > num_test:
            end = num_test
        data = x_test[start:end,:,:,:]
        test_x = torch.from_numpy(np.array(data)).to(device)
        predict = net(test_x)
        predict = torch.argmax(predict, 1).int()
        test_y.extend(predict.tolist())

    print(len(test_y))
    print('save to submission.csv')
    submission = pd.DataFrame(data={'ImageId': (np.arange(len(test_y)) + 1), 'Label': test_y})
    submission.to_csv('submission.csv', index=False)
    submission.tail()
