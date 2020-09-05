import numpy as np
import torch
import argparse
import os
from dataset import wsalDataset
from torch.utils.data import DataLoader
from models.stpn import STPN
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from utils.classification import calc_classification_mAP
from utils.detection import calc_detection_mAP
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

RANDOM_SEED = 7

def l1_penalty(var):
    return torch.abs(var).sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def returnTCAM(feature, fc):
    cam = feature.reshape((-1,1024)).dot(fc.reshape((1024,-1)))
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def adjust_learning_rate(optimizer, epoch):
    lr = 0.0001 * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_loader():
    dataset = wsalDataset(datapath = args.datapath,
                          mode = args.mode,
                          len_snippet = args.len_snippet,
                          stream = args.stream)

    shuffle = False
    if args.mode == "train":
        shuffle = True
    return DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=args.batch_size
    ), dataset

def train_step(loader, optimizer, criterion):
    total_loss = 0.0
    num_iterations = 0

    for i, (feats, labels, num_clips, _) in enumerate(loader):
        num_iterations+=1
        optimizer.zero_grad()
        feats = feats.to(device)
        labels = labels.to(device)
        output, attention = model(feats)
        cls_loss = criterion(output, labels)
        sparse_loss = l1_penalty(attention)/5000
        loss = cls_loss + sparse_loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    total_loss = total_loss/num_iterations
    return total_loss

def train(model, loader, optimizer):
    model.train()
    criterion = nn.BCELoss()
    best_loss = 10e8

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        loss = train_step(loader, optimizer, criterion)
        if loss<best_loss:
            best_loss = loss
            torch.save(model.state_dict(),f"./checkpoints/{args.stream}_model.pkl")
        print(f"Epoch: {epoch}\tLoss:{round(loss,5)} \tLearning Rate:{optimizer.param_groups[0]['lr']}\n\n")


def test(model, loader, dataset, store_cam=False, f="./t_cam.pth"):
    model.eval()
    model_path = f"./checkpoints/{args.stream}_model.pkl"
    if os.path.isfile(model_path):
        print("loading checkpoint file")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        print("loaded checkpoint")

    fc = np.squeeze(list(model.parameters())[-2].data.cpu().numpy())
    list_cas = []
    list_cls_scores = []
    list_labels = []
    for i, (feats, labels, num_clips, vname) in enumerate(loader):
        feats = feats.to(device)
        with torch.no_grad():
            output, attention = model(feats)
        feats = feats.data.cpu().squeeze().numpy()
        labels = labels.squeeze().numpy()
        output = output.data.cpu().squeeze().numpy()
        attention = attention.data.cpu().squeeze().numpy() # (T,)
        tcam = returnTCAM(feats,fc) # (T,20)
        
        weighted_tcam = (tcam.T * attention).T
        mask = np.ones(feats.shape[0])
        mask[:num_clips]=1
        weighted_tcam = (weighted_tcam.T * mask).T
        list_cas.append(weighted_tcam)
        list_cls_scores.append(output)
        list_labels.append(labels)

    cmap = calc_classification_mAP(np.array(list_cls_scores), np.array(list_labels))
    print(f"CmAP: {cmap}\n")
    list_iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_dmap = calc_detection_mAP(list_cas,np.array(list_cls_scores), list_iou, dataset)
    print("\t".join(list(map(lambda x:str(x),list_iou))))
    print("\n")
    print("\t".join(list(map(lambda x:str(round(x,3)),list_dmap))))
    # print("\t".join(list_dmap))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath',
                        default="./data/THUMOS14",
                        help="data path")
    parser.add_argument('--mode',
                        default="train",
                        help="mode")
    parser.add_argument("--len_snippet",
                        default=200,
                        type=int,
                        help="max number of clips in a video to consider")
    parser.add_argument("--batch_size",
                        default=10,
                        type=int,
                        help="Training Batch Size")
    parser.add_argument("--stream",
                        default="rgb",
                        help="Data Stream")
    parser.add_argument("--lr",
                        default=10e-4,
                        help="Learning rate")
    parser.add_argument("--epochs",
                        default=120,
                        type=int,
                        help="Learning rate")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in used : {device}, Number of GPU's : {torch.cuda.device_count()}")

    loader, dataset = get_loader()
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    model = STPN(20)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    if args.mode == "train":
        train(model,loader, optimizer)
    else:
        test(model,loader, dataset)