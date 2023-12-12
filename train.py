import os
import torch
import argparse
import timm
import numpy as np
import utils
import time
import torch.nn.functional as F

import warnings
warnings.filterwarnings(action='ignore')
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'vit_tiny_patch16_224', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--jigsawnum', '-j', default = 3, type=int)


    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu

    model_name = args.net
    dataset_path = config['id_dataset']
    save_path = config['save_path'] + args.save_path
    num_classes = int(config['num_classes'])
    class_range = list(range(0, num_classes))
    jigsaw_num = args.jigsawnum

    batch_size = int(int(config['batch_size'])) #/2)
    max_epoch = int(config['epoch'])
    lrde = [10]

    print(model_name, dataset_path.split('/')[-2], batch_size, class_range, jigsaw_num)
    
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    if 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar_jigsaw(args.data, dataset_path, batch_size, jigsaw_num = jigsaw_num)
    elif 'tinyimagenet' in args.data:
        train_loader, valid_loader = utils.get_tinyimagenet_jigsaw(dataset_path, batch_size, size=224)
    elif 'mnist' in args.data:
        train_loader, valid_loader = utils.get_mnist_jigsaw(dataset_path, batch_size, jigsaw_num = jigsaw_num)
    elif 'imagenet' in args.data:
        train_loader, valid_loader = utils.get_imagenet_jigsaw(dataset_path, batch_size)

    print(args.net)
    model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    print(utils.validation_accuracy(model, valid_loader, device))
    
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum=0.9, weight_decay = 5e-4)
    if 'mnist' in args.data:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.0003, momentum=0.9, weight_decay = 1e-4)
    if 'imagenet' in args.data:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum=0.9, weight_decay = 0)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)
    print(train_loader.dataset[0][0].shape)
    start_time = time.time()
    for epoch in range(15):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, inputs_jigsaw, targets) in enumerate(train_loader):
            inputs, inputs_jigsaw, targets = inputs.to(device), inputs_jigsaw.to(device), targets.to(device)
            optimizer.zero_grad()

            input_ = torch.cat((inputs, inputs_jigsaw), 0)

            # compute output
            output_ = model(input_)

            outputs = output_[:len(targets)]
            outputs_corr = output_[len(targets):]

            loss_in = criterion(outputs, targets)

            loss_out = torch.norm(F.relu(outputs_corr), dim=1).mean()
            loss = loss_in + loss_out

            loss.backward()            
            optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model, valid_loader, device)
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
if __name__ =='__main__':
    train()