import argparse
import os
import time
from matplotlib import pyplot as plt
import pickle

import numpy as np
import torch
import torch.nn as nn
from data_loader import DataSetWrapper
from resnet import resnet18, resnet50
import torchvision.models as models

from module import NT_XentLoss


def plot_training(total_accuracy, name):
    x_axis = np.arange(len(total_accuracy))
    plt.figure()
    plt.plot(x_axis, total_accuracy, 'b')
    plt.legend(['acc'])
    plt.savefig(os.path.join('./result', f'{name}.png'))

def train(epoch, model, optimizer, criterion, criterion2, train_loader, teacher_model, projection_head, device, beta = 1):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    total = 0
    correct = 0
    for x, z in train_loader:
        x = x.to(device)
        z = z.to(device)
        output, representation = model(x)

        optimizer.zero_grad()
        loss = criterion(output, z)

        if teacher_model != None: # Distillation
            representation = projection_head(representation)
            teacher_output = teacher_model(x).squeeze()
            loss2 = beta * criterion2(representation, teacher_output)
            loss += loss2
        loss.backward()
        optimizer.step()

        total += z.shape[0]
        _, output = output.max(dim=1)
        correct += (output == z).sum()
        total_loss += loss.item()

    acc = float(correct) / total
    print(f'[TRAIN] epoch: {epoch}, loss: {total_loss}, acc: {acc}, time: {time.time() - start_time}')

def validate(epoch, model, criterion, optimizer, valid_loader, 
            best_acc, name, device):
    model.eval()
    total_loss = 0.0
    start_time = time.time()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, z in valid_loader:
            x = x.to(device)
            z = z.to(device)
            output, _ = model(x)

            loss = criterion(output, z)

            total += z.shape[0]
            _, output = output.max(dim=1)
            correct += (output == z).sum()
            total_loss += loss.item()

    acc = float(correct) / total
    print(f'[VALID] epoch: {epoch}, loss: {total_loss}, acc: {acc}, time: {time.time() - start_time}')
    if acc > best_acc:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optmizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'acc': acc,
        }, os.path.join('./model', f'{name}.pt'))
        print('Model saved')

    return acc

def test(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0.0
    start_time = time.time()
    total = 0
    correct = 0

    with torch.no_grad():
        for x, z in test_loader:
            x = x.to(device)
            z = z.to(device)
            output, _ = model(x)

            loss = criterion(output, z)

            total += z.shape[0]
            _, output = output.max(dim=1)
            correct += (output == z).sum()
            total_loss += loss.item()

    acc = float(correct) / total
    print(f'[TEST] loss: {total_loss}, acc: {acc}, time: {time.time() - start_time}')

def main(args):
    if not os.path.isdir('./model'):
        os.mkdir('./model')
    if not os.path.isdir('./result'):
        os.mkdir('./result')
    if not os.path.isdir('./data'):
        os.mkdir('./data')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_worker = args.num_worker
    option = args.option
    epochs = args.epochs
    batch_size = args.batch_size
    T = args.termperature

    print(f'Using device: {device}')

    dataset = DataSetWrapper(batch_size, num_worker, 0.1)
    if option == 'teacher':
        model = resnet50().to(device)
    elif option == 'student':
        model = resnet18().to(device)
    elif option == 'distill':
        model = resnet18().to(device)

    criterion2 = None
    projection_head = None
    teacher_model = None
    model_parameters = model.parameters()
    if args.option == 'distill':
        checkpoint = torch.load(os.path.join(args.teacher_model), map_location=device)
        teacher_model = resnet50().to(device)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        teacher_model = nn.Sequential(*list(teacher_model.children())[:-1])
        for p in teacher_model.parameters():
            p.requires_grad = False
        criterion2 = NT_XentLoss(batch_size, T, device)
        projection_head = nn.Linear(512, 2048).to(device)
        model_parameters = list(model.parameters()) + list(projection_head.parameters())

    optimizer = torch.optim.SGD(model_parameters, lr=0.05, weight_decay=5e-4,
                                momentum=0.9)
    e = 180
    milestones = []
    while e < epochs:
        milestones.append(e)
        e += 30

    epoch = 0
    best_acc = 0.0
    total_accuracy = []
    if args.continue_training:
        checkpoint = torch.load(os.path.join(args.prev_model), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optmizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        for idx in range(len(milestones)):
            milestones[idx] -= epoch
        with open("acc.txt", "rb") as fp:
            total_accuracy = pickle.load(fp)
        print(f"Load model epoch: {epoch}")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    if not args.test:
        train_loader, valid_loader = dataset.get_train_data_loaders()

        while epoch < epochs:
            train(epoch, model, optimizer, criterion, criterion2, train_loader, teacher_model, projection_head, device, beta = args.beta)
            accuracy = validate(epoch, model, criterion, optimizer, valid_loader, best_acc, option + f"_beta{args.beta}", device)
            scheduler.step()

            total_accuracy.append(accuracy)

            if accuracy > best_acc:
                best_acc = accuracy
                with open("acc.txt", "wb") as fp:
                    pickle.dump(total_accuracy, fp)
            epoch += 1
        
        plot_training(total_accuracy, option)
    else:
        test_loader = dataset.get_test_loaders()
        checkpoint = torch.load(os.path.join(args.prev_model), map_location=device)
        epoch = checkpoint['epoch']+1
        print(f"Load model epoch: {epoch}")
        model.load_state_dict(checkpoint['model_state_dict'])
        test(model, criterion, test_loader, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "CRD")

    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--termperature', type=float, default=1)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--option', type=str, 
        choices=['teacher', 'student', 'distill'], default='teacher'
    )
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--prev_model', type=str, default="")
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--teacher_model', type=str, default='./model/teacher.pt')
    
    args = parser.parse_args()
    main(args)