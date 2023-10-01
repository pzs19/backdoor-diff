import torch
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from models.resnet import ResNet18
import numpy as np
import os
import os.path as osp
from torchvision.utils import save_image
from collections import OrderedDict
import argparse
import json

from utils import progress_bar
from models.resnet import ResNet18

import sys
sys.path.append('../data/CIFAR10')
from cifar_badnet import BadNetCIFAR10
from cifar_blend import BlendCIFAR10
from cifar_trojan import TrojanCIFAR10

def test(dataloader, net, net_p):
    net.eval()
    net_p.eval()
    correct = 0
    correct_p = 0
    inconsistent = 0
    total = 0
    pred_c = []
    pred_p = []
    is_eq = [] 
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            outputs_p = net_p(inputs)
            _, predicted_p = outputs_p.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            correct_p += predicted_p.eq(targets).sum().item()
            inconsistent += (~predicted.eq(predicted_p)).sum().item()
            pred_c.append(predicted.detach().cpu().numpy())
            pred_p.append(predicted_p.detach().cpu().numpy())
            progress_bar(batch_idx, len(dataloader), 'Clean Acc: %.3f%% | Poison Acc %.3f%% | Inconsistency %.3f%%'
                         % (100.*correct/total, 100.*correct_p/total, 100.*inconsistent/total))
    pred_c = np.concatenate(pred_c)
    pred_p = np.concatenate(pred_p)
    return 100.*correct/total, 100.*correct_p/total, 100.*inconsistent/total, pred_c, pred_p

parser = argparse.ArgumentParser(description='Cognitive Distillation CIFAR10 Eval')
parser.add_argument('--data_dir', default="../data2/CIFAR10", type=str)
parser.add_argument('--sample_dir', default="../edm/samples", type=str)
parser.add_argument("--model_dir", default="./checkpoint", type=str)
args = parser.parse_args()
device = torch.device("cuda")

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
clean_set = torchvision.datasets.CIFAR10(root=osp.join(args.data_dir, 'cifar10'), train=True, transform=transform_test)
clean_loader = torch.utils.data.DataLoader(clean_set, batch_size=1000, shuffle=False, num_workers=8)

# Clean Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == torch.device("cuda"):
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.eval()
# Load checkpoint.
print('==> Resuming from checkpoint..')
checkpoint = torch.load("./checkpoint/ckpt01.pth")
net.load_state_dict(checkpoint['net'])

# Poison Model
net_p = ResNet18()
net_p = net_p.to(device)
if device == torch.device("cuda"):
    net_p = torch.nn.DataParallel(net_p)
    cudnn.benchmark = True
net_p.eval()

info = {}
if os.path.exists(os.path.join(args.sample_dir, 'clf_pred.json')):
    with open(os.path.join(args.sample_dir, 'clf_pred.json'), 'r') as f:
        info = json.load(f)
pred_cs = {}
if os.path.exists(os.path.join(args.sample_dir, 'pred_cs.npz')):
    tmp = np.load(os.path.join(args.sample_dir, 'pred_cs.npz'), allow_pickle=True)
    for key in tmp.keys():
        pred_cs[key] = tmp[key]
pred_ps = {}
if os.path.exists(os.path.join(args.sample_dir, 'pred_ps.npz')):
    tmp = np.load(os.path.join(args.sample_dir, 'pred_ps.npz'), allow_pickle=True)
    for key in tmp.keys():
        pred_ps[key] = tmp[key]

# for trigger in ['badnet', 'trojan', 'blend']:
#     for tgt in [0, 4]:
#         for pr in [0.01, 0.05, 0.1]:
#             if trigger == 'badnet':
#                 ps = 5
#                 Dataset = BadNetCIFAR10
#                 data_path = osp.join(args.data_dir, f"{trigger}_ps{ps}_pr{pr}_tgt{tgt}.npz")
#                 model_path = osp.join(args.model_dir, f'ckpt01_poison{trigger}_ps{ps}_pr0.01_tgt{tgt}.pth')
#             elif trigger == 'trojan':
#                 Dataset = TrojanCIFAR10
#                 data_path = osp.join(args.data_dir, f"{trigger}_pr{pr}_tgt{tgt}.npz")
#                 model_path = osp.join(args.model_dir, f'ckpt01_poison{trigger}_pr0.01_tgt{tgt}.pth')
#             elif trigger == "blend":
#                 Dataset = BlendCIFAR10
#                 data_path = osp.join(args.data_dir, f"{trigger}_pr{pr}_tgt{tgt}.npz")
#                 model_path = osp.join(args.model_dir, f'ckpt01_poison{trigger}_pr0.01_tgt{tgt}.pth')
#             else:
#                 raise NotImplementedError()
#             if not osp.exists(data_path):
#                 print(f"{data_path} do not exist, continue.")
#                 continue
#             if not osp.exists(model_path):
#                 print(f"{model_path} do not exist, continue.")
#                 continue
#             if data_path in info:
#                 print(f'{data_path} already processed')
#                 continue
#             print(f'processing {data_path}')
            
#             dataset = Dataset(
#                 root=osp.join(args.data_dir, 'cifar10'), train=True, transform=transform_test, 
#                 data_path=data_path, target_label=tgt, select_label=tgt, patch_size=ps, poison_rate=pr)
#             # dataset.data = dataset.data[dataset.targets == tgt]
#             # dataset.targets = dataset.targets[dataset.targets == tgt]
#             dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=8)
#             checkpoint = torch.load(model_path)
#             net_p.load_state_dict(checkpoint['net'])

#             acc, acc_p, inconst, pred_c, pred_p = test(dataloader, net, net_p)
#             print(data_path)
#             print(acc, acc_p, inconst)
#             info[data_path] = (acc, acc_p, inconst)
#             pred_cs[data_path] = pred_c
#             pred_ps[data_path] = pred_p
#             np.savez(os.path.join(args.sample_dir, "pred_cs.npz"), **pred_cs)
#             np.savez(os.path.join(args.sample_dir, "pred_ps.npz"), **pred_ps)
#             with open(os.path.join(args.sample_dir, 'clf_pred.json'), 'w') as f:
#                 json.dump(info, f, indent=2, sort_keys=True)

for trigger in ['badnet', 'trojan', 'blend', 'clean']:
    for tgt in [0, 4]:
        for epoch in ['050176', '100352', '150528', '200000']:
            for pr in [0.01, 0.05, 0.1]:
                if trigger == 'badnet':
                    ps = 5
                    Dataset = BadNetCIFAR10
                    data_path = osp.join(args.sample_dir, f"{trigger}_ps{ps}_pr{pr}_tgt{tgt}_epoch{epoch}/class{tgt}/sample10000.npz")
                    model_path = osp.join(args.model_dir, f'ckpt01_poison{trigger}_ps{ps}_pr0.01_tgt{tgt}.pth')
                elif trigger == 'trojan':
                    Dataset = TrojanCIFAR10
                    data_path = osp.join(args.sample_dir, f"{trigger}_pr{pr}_tgt{tgt}_epoch{epoch}/class{tgt}/sample10000.npz")
                    model_path = osp.join(args.model_dir, f'ckpt01_poison{trigger}_pr0.01_tgt{tgt}.pth')
                elif trigger == 'blend':
                    Dataset = BlendCIFAR10
                    data_path = osp.join(args.sample_dir, f"{trigger}_pr{pr}_tgt{tgt}_epoch{epoch}/class{tgt}/sample10000.npz")
                    model_path = osp.join(args.model_dir, f'ckpt01_poison{trigger}_pr0.01_tgt{tgt}.pth')
                elif trigger == "clean":
                    Dataset = BadNetCIFAR10
                    data_path = osp.join(args.sample_dir, f"{trigger}_tgt{tgt}_epoch{epoch}/class{tgt}/sample10000.npz")
                    model_path = osp.join(args.model_dir, f'ckpt01_poisonbadnet_ps5_pr0.01_tgt{tgt}.pth')
                else:
                    raise NotImplementedError()
                if not osp.exists(data_path):
                    print(f"{data_path} do not exist, continue.")
                    continue
                if not osp.exists(model_path):
                    print(f"{model_path} do not exist, continue.")
                    continue
                if data_path in info:
                    print(f'{data_path} already processed')
                    continue
                print(f'processing {data_path}')

                dataset = Dataset(root=osp.join(args.data_dir, 'cifar10'), train=True, transform=transform_test, 
                        data_path=data_path, target_label=tgt, select_label=tgt)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=8)
                checkpoint = torch.load(model_path)
                net_p.load_state_dict(checkpoint['net'])

                acc, acc_p, inconst, pred_c, pred_p = test(dataloader, net, net_p)
                print(data_path)
                print(acc, acc_p, inconst)
                info[data_path] = (acc, acc_p, inconst)
                pred_cs[data_path] = pred_c
                pred_ps[data_path] = pred_p
                np.savez(os.path.join(args.sample_dir, "pred_cs.npz"), **pred_cs)
                np.savez(os.path.join(args.sample_dir, "pred_ps.npz"), **pred_ps)
                with open(os.path.join(args.sample_dir, 'clf_pred.json'), 'w') as f:
                    json.dump(info, f, indent=2, sort_keys=True)