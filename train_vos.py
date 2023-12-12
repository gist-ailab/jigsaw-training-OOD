import os
import torch
import torch.nn.functional as F
import argparse
import timm
import numpy as np
import time

import utils

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'vit_tiny_patch16_224', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--wd', '-w', type=float)
    parser.add_argument('--sample_number', default = 1000, type=int)
    parser.add_argument('--sample_from', type=int, default=10000)
    parser.add_argument('--select', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=40)
    parser.add_argument('--loss_weight', type=float, default=0.1)
    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu

    model_name = args.net
    dataset_path = config['id_dataset']
    save_path = config['save_path'] + args.save_path
    num_classes = int(config['num_classes'])
    class_range = list(range(0, num_classes))

    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    wd = 5e-04
    lrde = [10]

    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    if 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar(args.data, dataset_path, batch_size)
    elif 'tinyimagenet' in args.data:
        train_loader, valid_loader = utils.get_tinyimagenet(dataset_path, batch_size, size=224)
    elif 'mnist' in args.data:
        train_loader, valid_loader = utils.get_mnist_train(dataset_path, batch_size)
    elif 'imagenet' in args.data:
        train_loader, valid_loader = utils.get_imagenet(dataset_path, batch_size)


    model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)
    model.eval()
    model.to(device)
    feat_size = 192
    if 'imagenet' in args.data:
        feat_size = 384
    criterion = torch.nn.CrossEntropyLoss()

    print(utils.validation_accuracy(model, valid_loader, device))

    weight_energy = torch.nn.Linear(num_classes, 1).to(device)
    torch.nn.init.uniform_(weight_energy.weight)
    data_dict = torch.zeros(num_classes, args.sample_number, feat_size).to(device)
    number_dict = {}
    for i in range(num_classes):
        number_dict[i] = 0
    eye_matrix = torch.eye(feat_size, device=device)
    logistic_regression = torch.nn.Linear(1, 2)
    logistic_regression = logistic_regression.to(device)

    optimizer = torch.optim.SGD(list(model.parameters()) + list(weight_energy.parameters()) + \
        list(logistic_regression.parameters()), lr = 0.003, momentum=0.9, weight_decay = wd, nesterov=True)
    if 'mnist' in args.data:
        optimizer = torch.optim.SGD(list(model.parameters()) + list(weight_energy.parameters()) + \
            list(logistic_regression.parameters()), lr = 0.0003, momentum=0.9, weight_decay = 1e-4, nesterov=True)
    if 'imagenet' in args.data:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum=0.9, weight_decay = 0)

    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)   
    print(train_loader.dataset[0][0].shape)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            max_epoch * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / 0.1))

    def log_sum_exp(value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        import math
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(
                F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)
        
    start_time = time.time()
    for epoch in range(15):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Forward
            features = model.forward_features(inputs)
            outputs = model.head(features)
            
            # Energy regularization
            sum_temp = 0
            for index in range(num_classes):
                sum_temp += number_dict[index]
            lr_reg_loss = torch.zeros(1).cuda()[0]
            if sum_temp == num_classes * args.sample_number and epoch < args.start_epoch:
                # maintaining an ID data queue for each class.
                target_numpy = targets.cpu().data.numpy()
                for index in range(len(targets)):
                    dict_key = target_numpy[index]
                    data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                        features[index].detach().view(1, -1)), 0)
            elif sum_temp == num_classes * args.sample_number and epoch >= args.start_epoch:
                target_numpy = targets.cpu().data.numpy()
                for index in range(len(targets)):
                    dict_key = target_numpy[index]
                    data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                        features[index].detach().view(1, -1)), 0)
                # the covariance finder needs the data to be centered.
                for index in range(num_classes):
                    if index == 0:
                        X = data_dict[index] - data_dict[index].mean(0)
                        mean_embed_id = data_dict[index].mean(0).view(1, -1)
                    else:
                        X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                        mean_embed_id = torch.cat((mean_embed_id,
                                                data_dict[index].mean(0).view(1, -1)), 0)

                ## add the variance.
                temp_precision = torch.mm(X.t(), X) / len(X)
                temp_precision += 0.0001 * eye_matrix


                for index in range(num_classes):
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
                    negative_samples = new_dis.rsample((args.sample_from,))
                    prob_density = new_dis.log_prob(negative_samples)
                    # breakpoint()
                    # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                    # keep the data in the low density area.
                    cur_samples, index_prob = torch.topk(- prob_density, args.select)
                    if index == 0:
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                if len(ood_samples) != 0:
                    # add some gaussian noise
                    # ood_samples = self.noise(ood_samples)
                    # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                    energy_score_for_fg = log_sum_exp(outputs, 1)
                    predictions_ood = model.head(ood_samples)
                    # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                    energy_score_for_bg = log_sum_exp(predictions_ood, 1)

                    input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                    labels_for_lr = torch.cat((torch.ones(len(features)).cuda(),
                                            torch.zeros(len(ood_samples)).cuda()), -1)

                    criterion = torch.nn.CrossEntropyLoss()
                    output1 = logistic_regression(input_for_lr.view(-1, 1))
                    lr_reg_loss = criterion(output1, labels_for_lr.long())

                    # if epoch % 5 == 0:
                    #     print(lr_reg_loss)
            else:
                target_numpy = targets.cpu().data.numpy()
                for index in range(len(targets)):
                    dict_key = target_numpy[index]
                    if number_dict[dict_key] < args.sample_number:
                        data_dict[dict_key][number_dict[dict_key]] = features[index].detach()
                        number_dict[dict_key] += 1

            # backward

            optimizer.zero_grad()
            loss = F.cross_entropy(outputs, targets)
            # breakpoint()
            loss += args.loss_weight * lr_reg_loss
            loss.backward()       
            optimizer.step()
            scheduler.step()

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
        # scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
if __name__ =='__main__':
    train()