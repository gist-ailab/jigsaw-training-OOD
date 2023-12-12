import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
import logging

corrupt_name = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'snow', 'fog', 'zoom_blur',
                     'contrast', 'elastic_transform', 'brightness',
                    'speckle_noise', 'gaussian_blur', 'saturate', 'frost']


# corrupt_name = [ "frost", "contrast", "saturate"]

# logging.info("using corruptions", corrupt_name)

from imagecorruptions import corrupt

def get_corrupt_data(image):
    # normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # tiny-imagenet
    # normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]) # for cifar10/100
    mean=[0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std) # for cifar10/100

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # for imagenet
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    image_corrupt = []
    for i in image:    
        i = i * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        image_ = i.permute(1,2,0)
        image_ = np.array(image_*255).astype("uint8")

        idx = np.random.choice(len(corrupt_name));severity_index = np.random.choice(5)
        image_  = corrupt(image_, severity = severity_index + 1, corruption_name = corrupt_name[idx])
        image_ = transform(image_)
        image_corrupt.append(image_)
    return torch.stack(image_corrupt, 0)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def collate_cutmix(data):
    image, label = zip(*data)
    label = torch.tensor(label)
    image = torch.stack(image, 0)
    lam = np.random.beta(1,1)
    rand_index = torch.randperm(image.size()[0])
    label_perm = label[rand_index]
    com_ind = (label==label_perm)  
    label = label.cpu().detach().numpy() ; label_perm = label_perm.cpu().detach().numpy()
    indices = [i for i, x in enumerate(com_ind) if x]
    if indices :
        for ind, i in enumerate(indices):
            val = label_perm[i]
            ind_list = np.where(label == val)
            match_ind = np.setdiff1d(ind_list[0],indices)
            
            if len(match_ind) : label_perm[i] = label_perm[match_ind[0]]

            else : 
                
                for j in np.arange(image.size()[0]) :
                    if not j in ind_list[0]:
                        if not label_perm[j] == label_perm[i]:
                            label_perm[i] = label_perm[j]
                            break
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
    return image.to(dtype = torch.float), torch.tensor(label)

def collate_cnc(data):
    image, label = collate_cutmix(data)
    image = get_corrupt_data(image)
    return image.to(dtype = torch.float), label

def collate_corr(data):
    image, label = zip(*data)
    image = torch.stack(image, 0)
    image = get_corrupt_data(image)
    return image.to(dtype = torch.float), torch.tensor(label)

aug_dict = {
    "cutmix" : collate_cutmix,
    "vanilla" : torch.utils.data.dataloader.default_collate,
    "corr" : collate_corr,
    "cnc" : collate_cnc
}