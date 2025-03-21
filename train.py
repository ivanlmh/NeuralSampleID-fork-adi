import os
import numpy as np
import argparse
import torch
import gc
import sys
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler
import torchaudio
torchaudio.set_audio_backend("soundfile")


from util import *
from simclr.ntxent import ntxent_loss, SoftCrossEntropy
from simclr.simclr import SimCLR   
from modules.transformations import GPUTransformSampleID
from modules.data import NeuralSampleIDDataset
# from encoder.graph_encoder import GraphEncoder
from encoder.dgl.graph_encoder import GraphEncoderDGL
from encoder.resnet_ibn import ResNetIBN
from eval import eval_faiss
# from test_fp import create_fp_db, create_dummy_db

# Directories
root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")
parent_dir = os.path.abspath(os.path.join(root, os.pardir))
sys.path.append(parent_dir)
nan_counter = 0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


parser = argparse.ArgumentParser(description='Grafprint Training')
parser.add_argument('--config', default='config/grafp.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--train_dir', default=None, type=str, metavar='PATH',
                    help='path to training data')
parser.add_argument('--val_dir', default=None, type=str, metavar='PATH',
                    help='path to validation data')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--ckp', default='test', type=str,
                    help='checkpoint_name')
parser.add_argument('--encoder', default='grafp', type=str)
parser.add_argument('--n_dummy_db', default=None, type=int)
parser.add_argument('--n_query_db', default=None, type=int)
parser.add_argument('--k', default=3, type=int)



def mixco(model, xis, xjs, zis, zjs, cfg):
    # This code is adapted from https://github.com/Lee-Gihun/MixCo-Mixup-Contrast

    crit = SoftCrossEntropy()
    B = xis.shape[0]
    assert B % 2 == 0
    sid = int(B / 2)
    loss = 0

    for x, z in zip([xis, xjs], [zjs, zis]):
        x_1, x_2 = x[:sid], x[sid:]

        # Each input gets a different lambda
        lam = torch.from_numpy(np.random.uniform(0, 1, size=(sid, 1, 1))).float().to(x.device)
        spec_mix = lam * x_1 + (1 - lam) * x_2

        _, _, _, z_mix = model(spec_mix, spec_mix)
        z_mix = F.normalize(z_mix, dim=1)

        # Create labels with equal weighting regardless of lambda
        lbls_mix = torch.cat((torch.eye(sid), torch.eye(sid)), dim=1).to(x.device)
        logits_mix = torch.mm(z_mix, z.transpose(0, 1))
        logits_mix /= cfg['tau_mix']
        loss += crit(logits_mix, lbls_mix) / 2

    return loss



def train(cfg, train_loader, model, optimizer, scaler, ir_idx, noise_idx, augment=None):
    model.train()
    loss_epoch = 0
    global nan_counter

    for idx, (x_i, x_j) in enumerate(train_loader):

        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        with torch.no_grad():
            x_i, x_j = augment(x_i, x_j)

        _, _, z_i, z_j = model(x_i, x_j)

        simclr_loss = ntxent_loss(z_i, z_j, cfg)


        if torch.isnan(simclr_loss):
            print(f"NaN detected in loss at step {idx}, skipping batch")
            nan_counter = save_nan_batch(x_i, x_j, save_dir="nan_batches", counter=nan_counter)
            continue

        if cfg['beta'] > 0.0:       # Beta set to 0.0 as mixco support is not implemented
            mixco_loss = mixco(model, x_i, x_j, z_i, z_j, cfg)
        else:
            mixco_loss = torch.tensor(0.0)

        loss = simclr_loss
        assert not torch.isnan(loss), "Loss is NaN"

        scaler.scale(loss).backward()

        # Added gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(train_loader)}]\t SimCLR Loss: {simclr_loss.item()} \t MixCo Loss: {mixco_loss.item()}")

        loss_epoch += loss.item()

    return loss_epoch


def validate(epoch, query_loader, dummy_loader, augment, model, output_root_dir):
    # model.eval()
    # if epoch==1 or epoch % 10 == 0:
    #     create_dummy_db(dummy_loader, augment=augment, model=model, output_root_dir=output_root_dir, verbose=False)
    #     create_fp_db(query_loader, augment=augment, model=model, output_root_dir=output_root_dir, verbose=False)
    #     hit_rates = eval_faiss(emb_dir=output_root_dir, test_ids='all', index_type='l2', n_centroids=32, nogpu=True)
    #     print("-------Validation hit-rates-------")
    #     print(f'Top-1 exact hit rate = {hit_rates[0]}')
    #     print(f'Top-1 near hit rate = {hit_rates[1]}')
    # else:
    #     hit_rates = None
    # return hit_rates
    
    # Not implemented for now
    return None

def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    writer = SummaryWriter(f'runs/{args.ckp}')
    ir_dir = cfg['ir_dir']
    noise_dir = cfg['noise_dir']
    
    # Hyperparameters
    batch_size = cfg['bsz_train']
    learning_rate = cfg['lr']
    num_epochs = override(cfg['n_epochs'], args.epochs)
    model_name = args.ckp
    random_seed = args.seed
    shuffle_dataset = True

    print("Intializing augmentation pipeline...")
    noise_train_idx = load_augmentation_index(noise_dir, splits=0.8)["train"]
    ir_train_idx = load_augmentation_index(ir_dir, splits=0.8)["train"]
    noise_val_idx = load_augmentation_index(noise_dir, splits=0.8)["test"]
    ir_val_idx = load_augmentation_index(ir_dir, splits=0.8)["test"]
    gpu_augment = GPUTransformSampleID(cfg=cfg, ir_dir=ir_train_idx, noise_dir=noise_train_idx, train=True).to(device)
    cpu_augment = GPUTransformSampleID(cfg=cfg, ir_dir=ir_train_idx, noise_dir=noise_train_idx, cpu=True)
    val_augment = GPUTransformSampleID(cfg=cfg, ir_dir=ir_val_idx, noise_dir=noise_val_idx, train=False).to(device)

    print("Loading dataset...")
    train_dataset = NeuralSampleIDDataset(cfg=cfg, train=True, transform=cpu_augment)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)
    
    valid_dataset = NeuralSampleIDDataset(cfg=cfg, train=False)
    print("Creating validation dataloaders...")
    dataset_size = len(valid_dataset)
    indices = list(range(dataset_size))
    split1 = override(cfg['n_dummy'],args.n_dummy_db)
    split2 = override(cfg['n_query'],args.n_query_db)
    
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    dummy_indices, query_db_indices = indices[:split1], indices[split1: split1 + split2]

    dummy_db_sampler = SubsetRandomSampler(dummy_indices)
    query_db_sampler = SubsetRandomSampler(query_db_indices)

    dummy_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=1, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=dummy_db_sampler)
    
    query_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=1, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=query_db_sampler)
    

    print("Creating new model...")
    if args.encoder == 'resnet':
        # TODO: Add support for resnet encoder (deprecated)
        raise NotImplementedError
    elif args.encoder == 'grafp':
        model = SimCLR(cfg, encoder=GraphEncoderDGL(cfg=cfg, in_channels=cfg['n_filters'], k=args.k))
    elif args.encoder == 'resnet-ibn':
        model = SimCLR(cfg, encoder=ResNetIBN())
        
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # model = DataParallel(model).to(device)
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(device)
        
    print(count_parameters(model, args.encoder))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg['T_max'], eta_min = cfg['min_lr'])
    # scaler = GradScaler(enabled=True)
    scaler = DummyScaler()
       
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model, optimizer, scheduler, start_epoch, loss_log, hit_rate_log = load_ckp(args.resume, model, optimizer, scheduler)
            output_root_dir = create_fp_dir(resume=args.resume)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    else:
        start_epoch = 0
        loss_log = []
        hit_rate_log = []
        output_root_dir = create_fp_dir(ckp=args.ckp, epoch=1)


    print("Calculating initial loss ...")
    best_loss = float('inf')
    best_hr = 0
    # training

    for epoch in range(start_epoch+1, num_epochs+1):
        print("#######Epoch {}#######".format(epoch))
        loss_epoch = train(cfg, train_loader, model, optimizer, scaler, ir_train_idx, noise_train_idx, gpu_augment)
        writer.add_scalar("Loss/train", loss_epoch, epoch)
        loss_log.append(loss_epoch)
        output_root_dir = create_fp_dir(ckp=args.ckp, epoch=epoch)
        hit_rates = validate(epoch, query_loader, dummy_loader, val_augment, model, output_root_dir)
        # hit_rate_log.append(hit_rates[0] if hit_rates is not None else hit_rate_log[-1])
        if hit_rates is not None:
            writer.add_scalar("Exact Hit_rate (2 sec)", hit_rates[0][0], epoch)
            writer.add_scalar("Exact Hit_rate (4 sec)", hit_rates[0][1], epoch)
            writer.add_scalar("Near Hit_rate (2 sec)", hit_rates[1][0], epoch)

        checkpoint = {
            'epoch': epoch,
            'loss': loss_log,
            'valid_acc' : hit_rate_log,
            # 'hit_rate': hit_rates,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        save_ckp(checkpoint, model_name, model_folder, 'current')
        assert os.path.exists(f'checkpoint/model_{model_name}_current.pth'), "Checkpoint not saved"

        if loss_epoch < best_loss:
            best_loss = loss_epoch
            save_ckp(checkpoint, model_name, model_folder, 'best')

        if epoch % 10 == 0 or epoch in [105, 108, 113, 115, 118]:
            save_ckp(checkpoint, model_name, model_folder, epoch)

        if hit_rates is not None and hit_rates[0][0] > best_hr:
            best_hr = hit_rates[0][0]
            save_ckp(checkpoint, model_name, model_folder, epoch)
            
        scheduler.step()
    
  
        
if __name__ == '__main__':
    main()