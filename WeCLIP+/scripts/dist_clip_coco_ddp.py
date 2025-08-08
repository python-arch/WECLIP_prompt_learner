import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import datetime
import logging
import os
import random
import wandb
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import coco
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from WeCLIP_Plus.model_attn_aff_coco import WeCLIP_Plus
from WeCLIP_Plus.dice_loss import DiceLoss


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/data1/zbf_data/Project2024/FCLIP_DINO/configs/coco_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")
parser.add_argument("--local_rank" , type=int, default=os.getenv("LOCAL_RANK", 0))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_ddp(local_rank):
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    if local_rank != 0:
        os.environ["WANDB_MODE"] = "offline"
def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def validate(model=None, data_loader=None, cfg=None):
    model.eval()
    num_classes = cfg.dataset.num_classes
    preds, gts = [], []
    seg_hist = np.zeros((num_classes, num_classes) , device='cuda')
    num=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            seg_clip, seg_dino = model(inputs, name, 'val')
            # as we do in training
            seg_logits = 0.5 * seg_clip + 0.5 * seg_dino

            resized_segs = F.interpolate(seg_logits, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            num += 1

            if num % 1000 == 0:
                seg_hist, _ = evaluate.scores(gts, preds, seg_hist, num_classes=num_classes)
                preds, gts = [], []

        seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist, num_classes=num_classes)
    model.train()
    return seg_score



def validate_ddp(model=None, data_loader=None, cfg=None):
    model.eval()
    rank = dist.get_rank()
    num_classes = cfg.dataset.num_classes
    seg_hist = torch.zeros((num_classes, num_classes), device='cuda')

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >=", disable=(rank != 0)):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                seg_clip, seg_dino = model(inputs, name, 'val')
                seg_logits = 0.5 * seg_clip + 0.5 * seg_dino

            resized_segs = F.interpolate(seg_logits, size=labels.shape[1:], mode='bilinear', align_corners=False)
            preds = torch.argmax(resized_segs, dim=1)

            mask = (labels >= 0) & (labels < num_classes)
            hist = torch.bincount(
                num_classes * labels[mask].to(torch.int) + preds[mask],
                minlength=num_classes**2
            ).reshape(num_classes, num_classes)
            seg_hist += hist


    dist.all_reduce(seg_hist, op=dist.ReduceOp.SUM)

    model.train()
    return seg_hist

def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask



def train(cfg, rank):
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)


    train_dataset = coco.CocoClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                       train_dataset, shuffle=True)

    val_dataset = coco.CocoSegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=cfg.train.samples_per_gpu,
                              pin_memory=True,
                              num_workers = 12,
                              drop_last=True,
                              persistent_workers=True,
                              prefetch_factor=4)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        shuffle=False
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=1,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        drop_last=False
    )


    model = WeCLIP_Plus(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        dino_model=cfg.dino_init.dino_model,
        dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
        decoder_layers=cfg.dino_init.decoder_layer,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        clip_flag=cfg.clip_init.clip_flag,
        ctx_len=cfg.clip_init.ctx_len,
        device='cuda'
    ).cuda(rank)
    logging.info('\nNetwork config: \n%s'%(model))
    param_groups = model.get_param_groups()
    model = DDP(model, device_ids=[rank], output_device=rank,
                find_unused_parameters=True)

    mask_size = int(cfg.dataset.crop_size // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    if rank == 0:
        wandb.init(
            project="WeCLIP-plus",
            name=f"run_{timestamp}",
            config=OmegaConf.to_container(cfg, resolve=True),  # log YAML
        )
        
    prompt_lr = cfg.optimizer.prompt_lr  

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {"params": param_groups[4], "lr": prompt_lr,"weight_decay": 0.0},
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()
    criterion_dice = DiceLoss().cuda(rank)
    
    best_val_miou = -1.0

    for n_iter in range(cfg.train.max_iters):
        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        segs_clip, segs_dino, cam, attn_pred, fg_text_features, bg_text_features = model(inputs.cuda(rank, non_blocking=True), img_name)

        pseudo_label = cam

        segs= 0.5*segs_clip+0.5*segs_dino
        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        segs_clip = F.interpolate(segs_clip, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        segs_dino = F.interpolate(segs_dino, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        pred_clip_max, pred_label_clip = torch.max(F.softmax(segs_clip, dim=1), dim=1)
        pred_dino_max, pred_label_dino = torch.max(F.softmax(segs_dino, dim=1), dim=1)
        pred_max, pred_label_seg = torch.max(F.softmax(segs, dim=1), dim=1)

        fts_cam = cam.clone()

        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        if n_iter > cfg.train.max_iters // 2:
            pseudo_label[pred_max>0.75] = pred_label_seg[pred_max>0.75]

        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_loss2 = criterion_dice(segs, pseudo_label.type(torch.long))

        seg_clip_loss2 = get_seg_loss(segs_clip, pred_label_dino.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_dino_loss2 = get_seg_loss(segs_dino, pred_label_clip.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        loss = 1 * seg_loss + 0.1 * attn_loss + 0.1 * seg_clip_loss2 + 0.1 * seg_dino_loss2 + 1 * seg_loss2
        text_reg_loss = 1.0 * (fg_text_features.pow(2).mean() + bg_text_features.pow(2).mean())
            
        loss = loss + text_reg_loss
        avg_meter.add({'seg_loss': seg_loss.item(), 'attn_loss': attn_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        if rank == 0 and n_iter % 100 ==0:
            fg_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.module.prompt_learner_fg.parameters())
            bg_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.module.prompt_learner_bg.parameters())
            print(f"Iter {n_iter}: FG prompt grad: {fg_has_grad}, BG prompt grad: {bg_has_grad}")
            print(f'text reg loss: {text_reg_loss}')
        optimizer.step()
        
        if (n_iter + 1) % cfg.train.log_iters == 0  and rank == 0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, attn_loss: %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('attn_loss'), seg_mAcc))
            wandb.log(
                {
                    "iter": n_iter + 1,
                    "train/seg_loss": seg_loss.item(),
                    "train/attn_loss": attn_loss.item(),
                    "train/mAcc": seg_mAcc,
                    "lr": cur_lr,
                    "gpu_mem_GB": torch.cuda.memory_allocated(rank) / 1e9,
                },
                step=n_iter + 1,
            )
        if (n_iter + 1) % cfg.train.eval_iters == 0:
            dist.barrier()
            total_seg_hist = validate_ddp(model, val_loader, cfg)

            if rank == 0:
                iou = torch.diag(total_seg_hist) / (total_seg_hist.sum(dim=1) + total_seg_hist.sum(dim=0) - torch.diag(total_seg_hist))
                valid_classes = total_seg_hist.sum(dim=1) > 0
                miou = torch.nanmean(iou[valid_classes]).item()
                logging.info(f"[VAL] iter {n_iter+1}: mIoU = {miou:.2f}%")
                wandb.log({"val/mIoU": miou}, step=n_iter + 1)

                if miou > best_val_miou:
                    best_val_miou = miou
                    ckpt_path = os.path.join(cfg.work_dir.ckpt_dir, "best_miou.pth")
                    torch.save(model.module.state_dict(), ckpt_path)
                    logging.info(f"New best mIoU {miou:.2f}% – checkpoint saved to {ckpt_path}")

            dist.barrier()

    if rank == 0:
        wandb.finish()

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size
    init_ddp(args.local_rank)

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg, rank=args.local_rank)
