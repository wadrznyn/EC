#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

import sys

sys.path.append(r'/home/weiweili/notebook/QIANLI')

from utils24 import MyDataSet

from main_model24 import HiFuse_Small as create_model

from utils24 import read_train_data, read_val_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


# In[2]:


def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"using {device} device.")



    print(args)

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')



    tb_writer = SummaryWriter()



    train_images_path, train_images_label = read_train_data(args.train_data_path)

    val_images_path, val_images_label = read_val_data(args.val_data_path)
    
    


    img_size = 56

    data_transform = {

        "train": transforms.Compose([transforms.Resize(56),
                                     transforms.CenterCrop(img_size),

                                     transforms.RandomHorizontalFlip(),

                                     transforms.ToTensor(),

                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]), 
        
        "val": transforms.Compose([transforms.Resize(56),

                                   transforms.CenterCrop(img_size),

                                   transforms.ToTensor(),

                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}



    train_dataset = MyDataSet(images_path=train_images_path,

                              images_class=train_images_label,

                              transform=data_transform["train"])



    val_dataset = MyDataSet(images_path=val_images_path,

                            images_class=val_images_label,

                            transform=data_transform["val"])



    batch_size = args.batch_size

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,

                                               batch_size=batch_size,

                                               shuffle=False,

                                               pin_memory=True,

                                               num_workers=nw,

                                               collate_fn=train_dataset.collate_fn)




    val_loader = torch.utils.data.DataLoader(val_dataset,

                                             batch_size=batch_size,

                                             shuffle=False,

                                             pin_memory=True,

                                             num_workers=nw,

                                             collate_fn=val_dataset.collate_fn)


    model = create_model(num_classes=args.num_classes).to(device)
    '''model = create_model(num_classes=args.num_classes)
    if torch.cuda.device_count() > 1:
            # 将模型封装到nn.DataParallel中以支持多GPU并行训练
            model = nn.DataParallel(model)
    model.to(device)'''
   

    '''方法一：model._modules.items()
    返回网络中所有模块(该模块包含子模块)的iterators

    for name, module in model._modules.items():
        print(name)
        print(module)
    
    方法二：model.children()
    返回model中所有直接子模块的一个iterator
    for module in model.children():
    print(module)'''
    
    if args.RESUME == False:

        if args.weights != "":

            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)

            weights_dict = torch.load(args.weights, map_location=device)#['state_dict']



            # Delete the weight of the relevant category
            '''for k in list(weights_dict.keys()):

                if "conv_norm" in k:

                    del weights_dict[k]
                    
            for k in list(weights_dict.keys()):

                if "conv_head" in k:

                    del weights_dict[k]
            for k in list(weights_dict.keys()):
                
                if k=="head.weight":
                    
                    del weights_dict[k]
            for k in list(weights_dict.keys()):
                
                if k=="head.bias":
                    
                    del weights_dict[k]
            for k in list(weights_dict.keys()):
                
                if k=="norm.weight":
                    
                    del weights_dict[k]
            for k in list(weights_dict.keys()):
                
                if k=="norm.bias":
                    
                    del weights_dict[k]
            for k in list(weights_dict.keys()):

                if "downsample_layers.0" in k:

                    del weights_dict[k]
            for k in list(weights_dict.keys()):

                if "patch_embed." in k:

                    del weights_dict[k]
            for k in list(weights_dict.keys()):

                if "stages.0." in k:

                    del weights_dict[k] 
            for k in list(weights_dict.keys()):

                if "layers1." in k:

                    del weights_dict[k]
            for k in list(weights_dict.keys()):
                
                if "fu1" in k:
                    
                    del weights_dict[k]'''
            

            for k in list(weights_dict.keys()):

                if "stages.3." in k:

                    del weights_dict[k]
            for k in list(weights_dict.keys()):

                if "conv_norm" in k:

                    del weights_dict[k]
                    
            for k in list(weights_dict.keys()):

                if "conv_head" in k:

                    del weights_dict[k]
                    
            for k in list(weights_dict.keys()):

                if "layers4." in k:

                    del weights_dict[k]    
            
            for k in list(weights_dict.keys()):
                
                if "fu4" in k:
                    
                    del weights_dict[k]
            for k in list(weights_dict.keys()):
                
                if k=="head.weight":
                    
                    del weights_dict[k]
            for k in list(weights_dict.keys()):
                
                if k=="head.bias":
                    
                    del weights_dict[k]
            for k in list(weights_dict.keys()):
                
                if k=="norm.weight":
                    
                    del weights_dict[k]
            for k in list(weights_dict.keys()):
                
                if k=="norm.bias":
                    
                    del weights_dict[k]
            for k in list(weights_dict.keys()):

                if "downsample_layers.0" in k:

                    del weights_dict[k]
            for k in list(weights_dict.keys()):

                if "patch_embed." in k:

                    del weights_dict[k]
            '''for k in list(weights_dict.keys()):

                if "downsample_layers.3" in k:

                    del weights_dict[k] 
            for k in list(weights_dict.keys()):

                if "stages.2." in k:

                    del weights_dict[k] 
            for k in list(weights_dict.keys()):

                if "layers3." in k:

                    del weights_dict[k]
            for k in list(weights_dict.keys()):
                
                if "fu3" in k:
                    
                    del weights_dict[k]'''
                          

            model.load_state_dict(weights_dict, strict=False)
    
    
    if args.freeze_layers:

        for name, para in model.named_parameters():

            # All weights except head are frozen

            if "head" not in name:

                para.requires_grad_(False)

            else:

                print("training {}".format(name))
    
    #params= get_params_groups(model, weight_decay=args.wd)
    '''for name, value in model.named_parameters():
        if "conv_head" in name :
            value.requires_grad = True
        elif "conv_norm" in name:
            value.requires_grad = True
        elif "norm.weight" == name:
            value.requires_grad = True
        elif "norm.bias" == name:
            value.requires_grad = True
        elif "head.weight" == name:
            value.requires_grad = True
        elif "head.bias" == name:
            value.requires_grad = True
        elif "downsample_layers.0" in name:
            value.requires_grad = True
        elif "patch_embed." in name:
            value.requires_grad = True
        elif "stages.0." in name:
            value.requires_grad = True
        elif "layers1." in name:
            value.requires_grad = True
        elif "fu1" in name:
            value.requires_grad = True
        else:
            value.requires_grad = False'''
    
    for name, value in model.named_parameters():
        if "conv_head" in name :
            value.requires_grad = True
        elif "conv_norm" in name:
            value.requires_grad = True
        elif "norm.weight" == name:
            value.requires_grad = True
        elif "norm.bias" == name:
            value.requires_grad = True
        elif "head.weight" == name:
            value.requires_grad = True
        elif "head.bias" == name:
            value.requires_grad = True
        elif "stages.3." in name:
            value.requires_grad = True
        elif "layers4." in name:
            value.requires_grad = True
        elif "fu4" in name:
            value.requires_grad = True
        elif "downsample_layers.0" in name:
            value.requires_grad = True
        elif "patch_embed." in name:
            value.requires_grad = True
        else:
            value.requires_grad = False 
    
    params = filter(lambda p: p.requires_grad, model.parameters())

    
    
    # optimizer
    
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,

                                       warmup=True, warmup_epochs=1)



    best_acc = 0.

    start_epoch = 0



    if args.RESUME:

        path_checkpoint = "./model_weight/checkpoint/ckpt_best_100.pth"

        print("model continue train")

        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])

        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch']

        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])



    for epoch in range(start_epoch + 1, args.epochs + 1):

        with open('./nroot.txt', 'a')as f:
            print(epoch,file=f) 
            for nti in train_images_path:
                print(nti, end='\n',file=f)
            for nvi in val_images_path:
                print(nvi, end='\n ',file=f)

        # train

        train_loss, train_acc = train_one_epoch(model=model,

                                                optimizer=optimizer,

                                                data_loader=train_loader,

                                                device=device,

                                                epoch=epoch,

                                                lr_scheduler=lr_scheduler)



        # validate

        val_loss, val_acc = evaluate(model=model,

                                     data_loader=val_loader,

                                     device=device,

                                     epoch=epoch)

        

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]

        tb_writer.add_scalar(tags[0], train_loss, epoch)

        tb_writer.add_scalar(tags[1], train_acc, epoch)

        tb_writer.add_scalar(tags[2], val_loss, epoch)

        tb_writer.add_scalar(tags[3], val_acc, epoch)

        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)


        if best_acc < val_acc:

            if not os.path.isdir("./model_weight"):

                os.mkdir("./model_weight")

            torch.save(model.state_dict(), "./model_weight/best_model.pth")

            print("Saved epoch{} as new best model".format(epoch))

            best_acc = val_acc



        if epoch % 10 == 0:

            print('epoch:', epoch)

            print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

            checkpoint = {

                "net": model.state_dict(),

                'optimizer': optimizer.state_dict(),

                "epoch": epoch,

                'lr_schedule': lr_scheduler.state_dict()

            }

            if not os.path.isdir("./model_weight/checkpoint"):

                os.mkdir("./model_weight/checkpoint")

            torch.save(checkpoint, './model_weight/checkpoint/ckpt_best_%s.pth' % (str(epoch)))



        #add loss, acc and lr into tensorboard

        print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 3)))



    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameters: %.2fM" % (total/1e6))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', type=int, default=3)

    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--wd', type=float, default=1e-2)

    parser.add_argument('--RESUME', type=bool, default=False)



    parser.add_argument('--train_data_path', type=str,  
                        default="/home/weiweili/notebook/QIANLI/c4/train")

    parser.add_argument('--val_data_path', type=str,    
                        default="/home/weiweili/notebook/QIANLI/c4/val")



    parser.add_argument('--weights', type=str, 
                        
                        default= '/home/weiweili/notebook/QIANLI/covidct/first/model_weight/best_model.pth',

                        help='initial weights path')


    parser.add_argument('--freeze-layers', type=bool, default=False)

    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')



    opt = parser.parse_args()



    main(opt)


# In[ ]:











    

