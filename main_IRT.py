from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import scipy.io
import torch
import torch.nn as nn
import numpy as np
from glob import glob
import scipy.misc as sic
import subprocess
import models.network as net
import argparse
import random
from vgg import VGG19

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='Test', type=str, help="Name of model")
parser.add_argument("--save_freq", default=5, type=int, help="save frequency of epochs")
parser.add_argument("--use_gpu", default=1, type=int, help="use gpu or not")
parser.add_argument("--with_IRT", default=0, type=int, help="use IRT or not")
parser.add_argument("--IRT_initialization", default=0, type=int, help="use initialization for IRT or not")
parser.add_argument("--max_epoch", default=25, type=int, help="The max number of epochs for training")
parser.add_argument("--input", default='./demo/colorization/goat_input', type=str, help="dir of input video")
parser.add_argument("--processed", default='./demo/colorization/goat_processed', type=str, help="dir of processed video")
parser.add_argument("--output", default='None', type=str, help="dir of output video")

# set random seed
seed = 2020
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# process arguments
ARGS = parser.parse_args()
print(ARGS)
save_freq = ARGS.save_freq
input_folder = ARGS.input
processed_folder = ARGS.processed
with_IRT = ARGS.with_IRT
maxepoch = ARGS.max_epoch + 1
model=  ARGS.model
task = "/{}_IRT{}_initial{}".format(model, with_IRT, ARGS.IRT_initialization) #Colorization, HDR, StyleTransfer, Dehazing

# set gpu
if ARGS.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) 
        for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))    
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

device = torch.device("cuda:0" if ARGS.use_gpu else "cpu")

# define loss function 
def compute_error(real,fake):
    # return tf.reduce_mean(tf.abs(fake-real))
    return torch.mean(torch.abs(fake-real))

def Lp_loss(x, y):
    vgg_real = VGG_19(normalize_batch(x))
    vgg_fake = VGG_19(normalize_batch(y))
    p0 = compute_error(normalize_batch(x), normalize_batch(y))
    
    content_loss_list = []
    content_loss_list.append(p0)
    feat_layers = {'conv1_2' : 1.0/2.6, 'conv2_2' : 1.0/4.8, 'conv3_2': 1.0/3.7, 'conv4_2':1.0/5.6, 'conv5_2':10.0/1.5}

    for layer, w in feat_layers.items():
        pi = compute_error(vgg_real[layer], vgg_fake[layer])
        content_loss_list.append(w * pi)
    
    content_loss = torch.sum(torch.stack(content_loss_list))

    return content_loss

loss_L2 = torch.nn.MSELoss()
loss_L1 = torch.nn.L1Loss()


# Define model .
out_channels = 6 if with_IRT else 3
net = net.UNet(in_channels=3, out_channels=out_channels, init_features=32)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000,8000], gamma=0.5)

VGG_19 = VGG19(requires_grad=False).to(device)

# prepare data 
input_folders = [input_folder]
processed_folders = [processed_folder]

def prepare_paired_input(task, id, input_names, processed_names, is_train=0):
    net_in = np.float32(scipy.misc.imread(input_names[id]))/255.0
    if len(net_in.shape) == 2:
        net_in = np.tile(net_in[:,:,np.newaxis], [1,1,3])
    net_gt = np.float32(scipy.misc.imread(processed_names[id]))/255.0
    org_h,org_w = net_in.shape[:2]   
    h = org_h // 32 * 32
    w = org_w // 32 * 32
    print(net_in.shape, net_gt.shape)
    return net_in[np.newaxis, :h, :w, :], net_gt[np.newaxis, :h, :w, :]

# some functions 
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):        
            # nn.init.kaiming_normal_(module.weight)
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return (batch - mean) / std


# start to train 
for folder_idx, input_folder in enumerate(input_folders): 
    # -----------load data------------- 
    input_names = sorted(glob(input_folders[folder_idx] + "/*"))
    processed_names = sorted(glob(processed_folders[folder_idx] + "/*"))
    if ARGS.output == "None":
        output_folder = "./result/{}".format(task + '/' + input_folder.split("/")[-2] + '/' + input_folder.split("/")[-1]) 
    else:
        output_folder = ARGS.output + "/" + task + '/' + input_folder.split("/")[-1] 
    print(output_folder, input_folders[folder_idx], processed_folders[folder_idx] )
    
    num_of_sample = min(len(input_names), len(processed_names))
    data_in_memory = [None] * num_of_sample                                                 #Speedup
    for id in range(min(len(input_names), len(processed_names))):                           #Speedup
        net_in,net_gt = prepare_paired_input(task, id, input_names, processed_names)        #Speedup
        net_in = torch.from_numpy(net_in).permute(0,3,1,2).float().to(device)
        net_gt = torch.from_numpy(net_gt).permute(0,3,1,2).float().to(device)
        data_in_memory[id] = [net_in,net_gt]                                                #Speedup

    # model re-initialization 
    initialize_weights(net)
    
    step = 0
    for epoch in range(1,maxepoch):
        # -----------start to train-------------
        print("Processing epoch {}".format(epoch))
        frame_id = 0
        if os.path.isdir("{}/{:04d}".format(output_folder, epoch)):
            continue
        else:
            os.makedirs("{}/{:04d}".format(output_folder, epoch))
        if not os.path.isdir("{}/training".format(output_folder)):
            os.makedirs("{}/training".format(output_folder))

        print(len(input_names), len(processed_names))
        for id in range(num_of_sample): 
            if with_IRT:      
                if epoch < 6 and ARGS.IRT_initialization:
                    net_in,net_gt = data_in_memory[0]      #Option: 
                    prediction = net(net_in)
                    
                    crt_loss = loss_L1(prediction[:,:3,:,:], net_gt) + 0.9*loss_L1(prediction[:,3:,:,:], net_gt)

                else:
                    net_in,net_gt = data_in_memory[id]
                    prediction = net(net_in)
                    
                    prediction_main = prediction[:,:3,:,:]
                    prediction_minor = prediction[:,3:,:,:]
                    diff_map_main,_ = torch.max(torch.abs(prediction_main - net_gt) / (net_in+1e-1), dim=1, keepdim=True)
                    diff_map_minor,_ = torch.max(torch.abs(prediction_minor - net_gt) / (net_in+1e-1), dim=1, keepdim=True)
                    confidence_map = torch.lt(diff_map_main, diff_map_minor).repeat(1,3,1,1).float()
                    crt_loss = loss_L1(prediction_main*confidence_map, net_gt*confidence_map) \
                            + loss_L1(prediction_minor*(1-confidence_map), net_gt*(1-confidence_map))
            else:
                net_in,net_gt = data_in_memory[id] 
                prediction = net(net_in)
                crt_loss = Lp_loss(prediction, net_gt)

            optimizer.zero_grad()
            crt_loss.backward()
            optimizer.step()

            frame_id+=1
            step+=1
            if step % 10 == 0:
                print("Image iter: {} {} {} || Loss: {:.4f} ".format(epoch, frame_id, step, crt_loss))
            if step % 100 == 0 :
                net_in = net_in.permute(0,2,3,1).cpu().numpy()
                net_gt = net_gt.permute(0,2,3,1).cpu().numpy()
                prediction = prediction.detach().permute(0,2,3,1).cpu().numpy()
                if with_IRT:
                    prediction = prediction[...,:3]
                sic.imsave("{}/training/step{:06d}_{:06d}.jpg".format(output_folder, step, id), 
                           np.uint8(np.concatenate([net_in[0], prediction[0], net_gt[0]], axis=1).clip(0,1) * 255.0))      

        # # -----------save intermidiate results-------------
        if epoch % save_freq == 0:
            for id in range(num_of_sample):
                st=time.time()
                net_in,net_gt = data_in_memory[id]
                print("Test: {}-{} \r".format(id, num_of_sample))

                with torch.no_grad():
                    prediction = net(net_in)                
                net_in = net_in.permute(0,2,3,1).cpu().numpy()
                net_gt = net_gt.permute(0,2,3,1).cpu().numpy()
                prediction = prediction.detach().permute(0,2,3,1).cpu().numpy()
                
                if with_IRT:
                    prediction_main = prediction[...,:3]
                    prediction_minor = prediction[...,3:]
                    diff_map_main = np.amax(np.absolute(prediction_main - net_gt) / (net_in+1e-1), axis=3, keepdims=True)
                    diff_map_minor = np.amax(np.absolute(prediction_minor - net_gt) / (net_in+1e-1), axis=3, keepdims=True)                    
                    confidence_map = np.tile(np.less(diff_map_main, diff_map_minor), (1,1,1,3)).astype('float32')

                    sic.imsave("{}/{:04d}/predictions_{:05d}.jpg".format(output_folder, epoch, id),
                        np.uint8(np.concatenate([net_in[0,:,:,:3],prediction_main[0], prediction_minor[0],net_gt[0], confidence_map[0]], axis=1).clip(0,1) * 255.0))
                    sic.imsave("{}/{:04d}/out_main_{:05d}.jpg".format(output_folder, epoch, id),np.uint8(prediction_main[0].clip(0,1) * 255.0))               
                    sic.imsave("{}/{:04d}/out_minor_{:05d}.jpg".format(output_folder, epoch, id),np.uint8(prediction_minor[0].clip(0,1) * 255.0))   

                else:

                    sic.imsave("{}/{:04d}/predictions_{:05d}.jpg".format(output_folder, epoch, id), 
                        np.uint8(np.concatenate([net_in[0,:,:,:3], prediction[0], net_gt[0]],axis=1).clip(0,1) * 255.0))
                    sic.imsave("{}/{:04d}/out_main_{:05d}.jpg".format(output_folder, epoch, id), 
                        np.uint8(prediction[0].clip(0,1) * 255.0))

