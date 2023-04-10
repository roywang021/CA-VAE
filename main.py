import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from resnet import ResNet18
import os
import cv2
from dataloader import MNIST_Dataset, CIFAR10_Dataset, SVHN_Dataset, CIFARAdd10_Dataset, CIFARAdd50_Dataset, CIFARAddN_Dataset
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--seed_sampler', type=str, default='777 1234 2731 3925 5432', help='random seed for dataset sampler')
parser.add_argument('--dataset', type=str, default="SVHN", help='The dataset going to use')
args = parser.parse_args()

def control_seed(args):
    # seed
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True



EPOCH = 135
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.01
control_seed(args)
seed_sampler = int(args.seed_sampler.split(' ')[0])

load_dataset = SVHN_Dataset()
train_dataset, val_dataset, test_dataset = load_dataset.sampler(seed_sampler, args)
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)



net = ResNet18().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85
    print("Start Training, Resnet-18!")
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):

                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()


                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('test_acc：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total

                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()

                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc

                        # save model
                        states = {}
                        states['epoch'] = epoch
                        states['model'] = net.state_dict()

                        states['acc'] = acc
                        torch.save(states, os.path.join(args.outf, 'model.pkl'))

            print("Training Finished, best_acc=%f" % acc)



