# Jigsaw training for improving OOD
Official Implementation of the **"Exploring the Use of Jigsaw Puzzles in Out-of-Distribution Detection (Submitted to Computer Vision and Image Understanding)"** by Yeonguk Yu, Sungho Shin, Minhwan Ko, and Kyoobin Lee. 

In this study, we e introduce a novel training method to improve the OOD detection performance with jigsaw puzzles, where the model is trained to produce low logit norm for given jigsaw puzzles. We conduct comprehensive experiments and show that our method consistently improves the OOD detection performance of the model and outperforms previous baselines.

![concept.png](/figure/figure_intro.png)


---
# Updates & TODO Lists
- [ ] pretrained checkpoints
- [x] Environment settings and Train & Evaluation Readme
- [ ] Presentation video

# Getting Started
## Environment Setup
   This code is tested under Window10 and Python 3.7.7 environment, and the code requires following packages to be installed:
    
   - [Pytorch](https://pytorch.org/): Tested under 1.11.0 version of Pytorch-GPU.
   - [torchvision](https://pytorch.org/vision/stable/index.html): which will be installed along Pytorch. Tested under 0.6.0 version.
   - [timm](https://github.com/rwightman/pytorch-image-models): Tested under 0.4.12 version.
   - [scipy](https://www.scipy.org/): Tested under 1.4.1 version.
   - [scikit-learn](https://scikit-learn.org/stable/): Tested under 0.22.1 version.

## Dataset Preparation
   Some public datasets are required to be downloaded for running evaluation. Required dataset can be downloaded in following links as in https://github.com/wetliu/energy_ood:    
   - [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
   - [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
   - [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
   - [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)

### Config file need to be changed for your path to download. For example,
~~~
# conf/cifar10.json
{
    "epoch" : "100",
    "id_dataset" : "./cifar10",   # Your path to Cifar10
    "batch_size" : 128,
    "save_path" : "./cifar10/",   # Your path to checkpoint
    "num_classes" : 10
}
~~~
Also, you need to change the path of the OOD dataset in "eval.py" to conduct a OOD benchmark. 
~~~
OOD_results(preds_in, model, get_svhn('/SSDe/yyg/data/svhn', batch_size), device, args.method+'-SVHN', f)
OOD_results(preds_in, model, get_ood('/SSDe/yyg/data/ood-set/textures/images'), device, args.method+'-TEXTURES', f)
OOD_results(preds_in, model, get_ood('/SSDe/yyg/data/ood-set/LSUN'), device, args.method+'-LSUN-crop', f)
OOD_results(preds_in, model, get_ood('/SSDe/yyg/data/ood-set/LSUN_resize'), device, args.method+'-LSUN-resize', f)
OOD_results(preds_in, model, get_ood('/SSDe/yyg/data/ood-set/iSUN'), device, args.method+'-iSUN', f)
OOD_results(preds_in, model, get_places('/SSDd/yyg/data/places256'), device, args.method+'-Places365', f)
~~~

---
## How to Run
### To train a model by our setting (i.e., ours) with ViT tiny variant
~~~
python train_vit_jigsaw_norm.py -d 'data_name' -g 'gpu-num' -n vit_tiny_patch16_224 -s 'save_name'
~~~
for example,
~~~
python train_vit_jigsaw_norm.py -d cifar10 -g 0 -n vit_tiny_patch16_224 -s baseline
~~~


### To detect OOD using norm of logit
~~~
python eval_vit.py -n vit_tiny_patch16_224 -d 'data_name' -g 'gpu_num' -s 'save_name' -m norm
~~~
for example, 
~~~
python eval_vit.py -n vit_tiny_patch16_224 -d cifar10 -g 0 -s baseline -m norm
~~~
Also, you can try MSP method
~~~
python eval_vit.py -n vit_tiny_patch16_224 -d 'data_name' -g 'gpu_num' -s 'save_name' -m msp
~~~
    
# License
The source code of this repository is released only for academic use. See the [license](LICENSE) file for details.

# Acknowledgement
This work was partially supported by Institute of Information \& communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00951, Development of Uncertainty-Aware Agents Learning by Asking Questions) and by ICT R\&D program of MSIT/IITP[2020-0-00857, Development of Cloud Robot Intelligence Augmentation, Sharing and Framework Technology to Integrate and Enhance the Intelligence of Multiple Robots].

