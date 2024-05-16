import torch
import torch.optim
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train(train_loader, test_loader, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 40
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        train = tqdm(train_loader)

        model.train()
        for cnt, (data, label) in enumerate(train, 1):
            outputs = model(data)
            loss = criterion(outputs, label)
            _, predict_label = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (predict_label == label).sum()
            train.set_description(f'train Epoch {epoch}')
            train.set_postfix({'loss': float(train_loss) / cnt, 'acc': float(train_acc) / cnt})

        model.eval()
        test = tqdm(test_loader)
        test_acc = 0
        for cnt, (data, label) in enumerate(test, 1):
            outputs = model(data)
            _, predict_label = torch.max(outputs, 1)
            test_acc += (predict_label == label).sum()
            test.set_description(f'test Epoch {epoch}')
            test.set_postfix({'acc': float(test_acc) / cnt})


def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
