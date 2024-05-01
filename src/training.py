import torch
import torch.nn as nn
import torch.optim
import os
from lib import resnet, dataset
from torch.utils.data import DataLoader


def set_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=1, help='frequency to save checkpoints')
    return parser.parse_args()


def _set_up_for_training(args):
    # set up device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model:
    model = resnet.resnet18().to(device)
    # set up optimizer:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    # set up loss function:
    criterion = nn.CrossEntropyLoss()
    # set up data loader:
    train_ds, val_ds = dataset.get(200)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return model, optimizer, criterion, train_loader, val_loader, device


def train(args):
    # set up for training:
    model, optimizer, criterion, train_loader, val_loader, device = _set_up_for_training(args)

    # train the model:
    for epoch in range(args.num_epochs):
        # train:
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # validate:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            'Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch + 1, args.num_epochs, loss.item(), correct / total))
        # save the model:
        if (epoch + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}.pth'.format(epoch + 1)))

    # save the final model:
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final.pth'))


# run all the things:
if __name__ == '__main__':
    # set up arguments:
    args = set_args()
    # train
    train(args)
