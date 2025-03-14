import torch
import argparse
import numpy as np
import os
import random
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.model import R2GenModel
from modules.utils import parse_args, auto_resume_helper, load_checkpoint
from modules.logger import create_logger
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter



def sample():
    args, config = parse_args()
    print(args)
    tokenizer = Tokenizer(args)
    os.makedirs('ignore', exist_ok=True)
    #logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    logger = create_logger(output_dir='ignore',  name=f"{config.MODEL.NAME}")

    model = R2GenModel(args, tokenizer, logger, config)

    state_dict = torch.load('model_iu_xray.pth')['model']
    model.load_state_dict(state_dict,strict=False)
    print('Load state dict.')
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    device = 'cuda'
    
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    with torch.no_grad():
                val_gts, val_res = [], []
                for batch_idx, (images_id, images, reports_ids, reports_masks,labels) in enumerate(val_dataloader):
                    images, reports_ids, reports_masks,labels = images.to(device,non_blocking=True), \
                                                         reports_ids.to(device,non_blocking=True), \
                                                         reports_masks.to(device, non_blocking=True), \
                                                         labels.to(device, non_blocking = True)
                    total_attn = None
                    output, _, _ = model(images,labels=labels,mode='sample')

                    reports = tokenizer.decode_batch(output.cpu().numpy())
                    groung_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    for r,gt in zip(reports,groung_truths):
                        print('ImageID:',images_id)
                        print('Ground Truth: ',gt)
                        print('Report: ',r) 



