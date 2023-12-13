import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse

from utils import *
# from evaluation import *


def calculate_norm(model, loader, device, thr):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)
            out = model(x)            
            # norm = torch.max(out, dim=1).values
            
            norm = torch.norm(F.relu(out-thr), p=2, dim=1)

            predictions.append(norm)
    predictions = torch.cat(predictions).to(device)
    return predictions

def OOD_results(preds_id, model, loader, device, method, thr, file):  
    #image_norm(loader)  
    if 'norm' in method:
        preds_ood = calculate_norm(model, loader, device, thr).cpu()

    print(torch.mean(preds_ood), torch.mean(preds_id))
    fpr, auroc = show_performance(preds_id, preds_ood, method, file=file)
    return fpr, auroc

def norm_thr(model, train_loader, device):
    model.eval()

    norms = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >1000:
                break
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            mask_ = torch.ones_like(outputs)
            max_logit = outputs.max(1).indices.unsqueeze(1)
            mask_ = mask_.scatter(dim=1, index=max_logit, src =torch.zeros_like(outputs))

            # outputs = outputs * mask_

            # norms.append((outputs).max(1).values)
            # norms.append(outputs.max(1).values)
            sorted_val = torch.sort(outputs, dim=1).values
            # print(sorted_val[0].shape, sorted_val[0])
            norms.append(sorted_val[:, outputs.size(1)-2])

    norms = torch.cat(norms, dim=0)
    print(norms.mean())
    return norms.mean()

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'vit_tiny_patch16_224', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', type=str, default = '0')
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--method' ,'-m', default = 'norm', type=str)

    args = parser.parse_args()

    config = read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    save_path = config['save_path'] + args.save_path
    
    num_classes = int(config['num_classes']) 
    if 'cnc' in args.method:
        num_classes += 1

    if 'cifar' in args.data:
        train_loader, valid_loader = get_cifar(args.data, dataset_path, batch_size)
    elif 'mnist' in args.data:
        train_loader, valid_loader = get_mnist_train(dataset_path, batch_size)
    elif 'imagenet' in args.data:
        train_loader, valid_loader = get_imagenet(dataset_path, batch_size)

    model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)

    state_dict = (torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict'])    

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    thr = norm_thr(model, train_loader, device)


    calculate_score = calculate_norm

    f = open(save_path+'/{}_result.txt'.format(args.method), 'w')
    valid_accuracy = validation_accuracy(model, valid_loader, device)



    print('In-distribution accuracy: ', valid_accuracy)
        
    f.write('Accuracy for ValidationSet: {}\n'.format(str(valid_accuracy)))
    #MSP
    #image_norm(valid_loader)  
    mean_fpr, mean_auroc = [], []
    preds_in = calculate_score(model, valid_loader, device, thr).cpu()
    if 'cifar' in args.data:
        fpr, auroc = OOD_results(preds_in, model, get_svhn('./ood-set/svhn', batch_size), device, args.method+'-SVHN', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);
        fpr, auroc =OOD_results(preds_in, model, get_textures('./ood-set/textures/images'), device, args.method+'-TEXTURES', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);
        fpr, auroc =OOD_results(preds_in, model, get_lsun('./ood-set/LSUN'), device, args.method+'-LSUN', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);
        
        fpr, auroc =OOD_results(preds_in, model, get_lsun('./ood-set/LSUN_resize'), device, args.method+'-LSUN-resize', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);

        fpr, auroc =OOD_results(preds_in, model, get_lsun('./ood-set/iSUN'), device, args.method+'-iSUN', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);

        fpr, auroc =OOD_results(preds_in, model, get_places('./ood-set/places'), device, args.method+'-Places365', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);

        cifar = 'cifar100' if args.data == 'cifar10' else 'cifar10'
        fpr, auroc =OOD_results(preds_in, model, get_cifar(cifar, './{}'.format(cifar), batch_size)[1], device, args.method+'-{}'.format(cifar), thr, f)
    
    if 'mnist' in args.data:
        fpr, auroc = OOD_results(preds_in, model, get_fnist('./ood-set/fmnist'), device, args.method+'-FMNIST', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);
        fpr, auroc =OOD_results(preds_in, model, get_knist('./ood-set/kmnist'), device, args.method+'-KMNIST', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);

    if 'imagenet' in args.data:  
        fpr, auroc = OOD_results(preds_in, model, get_ood_folder('./ood-set/OOD_for_ImageNet/iNaturalist', batch_size), device, args.method+'-iNaturalist', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);
        fpr, auroc = OOD_results(preds_in, model, get_ood_folder('./ood-set/OOD_for_ImageNet/SUN', batch_size), device, args.method+'-SUN', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);
        fpr, auroc = OOD_results(preds_in, model, get_ood_folder('./ood-set/OOD_for_ImageNet/Places', batch_size), device, args.method+'-PLACES', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);
        fpr, auroc = OOD_results(preds_in, model, get_ood_folder('./ood-set/OOD_for_ImageNet/dtd/images', batch_size), device, args.method+'-Textures', thr, f)
        mean_fpr.append(fpr); mean_auroc.append(auroc);
    f.close()

    print(torch.mean(torch.tensor(mean_fpr)).item())
    print(torch.mean(torch.tensor(mean_auroc)).item())



if __name__ =='__main__':
    eval()