import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from resnet import ResNet18
import os
import cv2

def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())
    print("hook")
def farward_hook(module, input, output):
    fmap_block.append(output)

def comp_class_vec(ouput_vec, index=None):
    """

    :param ouput_vec: tensor
    :param index: int
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)
    return class_vec

def gen_cam(feature_map, grads):
    """

    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #shape (C,)

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)#relu
    cam = cv2.resize(cam, (32, 32))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam

def show_cam_on_image(img, mask,cnt, out_dir='./cam/'):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    print('heatmap',heatmap.shape)
    print('img',img.shape)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir,str(cnt)+ "cam.jpg")
    print(path_cam_img)
    path_raw_img = os.path.join(out_dir,str(cnt)+"raw.jpg")
    print(path_raw_img)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))
    
def show_cam_on_image_groundtruth(img, mask,cnt, out_dir='./cam/'):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    print('heatmap',heatmap.shape)
    print('img',img.shape)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir,str(cnt)+ "cam_g.jpg")
    print(path_cam_img)
    path_raw_img = os.path.join(out_dir,str(cnt)+"raw.jpg")
    print(path_raw_img)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
inv_normalize = torchvision.transforms.Normalize(
        mean=(-2.4290,-2.4183,-2.2214),
        std= (4.9432, 5.0150, 4.9751))

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    #net = ResNet18().to(device)
    net=ResNet18()
    states = torch.load(os.path.join('./model/', 'model.pkl'))
    net.load_state_dict(states['model'])

    fmap_block = list()
    grad_block = list()
    net.identity.register_forward_hook(farward_hook)
    net.identity.register_backward_hook(backward_hook)

    # grad_CAM
    cnt=0
    for data in testloader:
        images, labels = data
        # forward
        output = net(images[0].unsqueeze(0))
        idx = np.argmax(output.cpu().data.numpy())
        print("predict: {}".format(classes[idx]))
        print("groundtruth: {}".format(classes[labels[0]]))
        # backward
        net.zero_grad()
        class_loss = comp_class_vec(output)
        class_loss.backward()

        grads_val = grad_block[cnt].cpu().data.numpy().squeeze()
        fmap = fmap_block[cnt].cpu().data.numpy().squeeze()
        cam = gen_cam(fmap, grads_val)

        img_show = np.float32(cv2.resize(inv_normalize(images[0]).permute(1,2,0).numpy(), (32, 32))) 
        show_cam_on_image(img_show, cam,cnt)
        cnt+=1
    cnt=0
    for data in testloader:
        images, labels = data
        # forward
        output = net(images[0].unsqueeze(0))
        idx = np.argmax(output.cpu().data.numpy())
        print("predict: {}".format(classes[idx]))
        print("groundtruth: {}".format(classes[labels[0]]))
        # backward
        net.zero_grad()
        class_loss = comp_class_vec(output,labels[0])
        class_loss.backward()

        grads_val = grad_block[cnt+100].cpu().data.numpy().squeeze()
        fmap = fmap_block[cnt+100].cpu().data.numpy().squeeze()
        cam = gen_cam(fmap, grads_val)

        img_show = np.float32(cv2.resize(inv_normalize(images[0]).permute(1,2,0).numpy(), (32, 32))) 
        show_cam_on_image_groundtruth(img_show, cam,cnt)
        cnt+=1
    print(len(grad_block))
        
