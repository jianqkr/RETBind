import os
import yaml
import json
import torch
import argparse
import numpy as np
import torchvision.utils as vutils
import torch.distributed as dist
from pathlib import Path

from models.model import pretrain_masked_multimae_base
from models.criterion import MaskedMSELoss
from models.input_model import PatchedInputAdapter, OctInputAdapter
from models.output_model import SpatialOutputAdapter

from utils.task_balancing import NoWeightingStrategy, UncertaintyWeightingStrategy
from utils.optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils.datasets import build_multimae_pretraining_dataset
from utils.logger import MetricLogger
from utils.native_scaler import cosine_scheduler
from utils.checkpoint import save_model_pretrain
from utils import create_model


DOMAIN_CONF = {
    'uwf': {
        'input_adapter': lambda p, s: PatchedInputAdapter(3, patch_size_full=p, stride_level=1, image_size=s),
        'output_adapter': lambda p, s, c, d: SpatialOutputAdapter(3, patch_size_full=p, stride_level=1, image_size=s, context_tasks=c, depth=d, task='uwf'),
        'loss': lambda p: MaskedMSELoss(patch_size=p),
    },
    'oct': {
        'input_adapter': lambda p, s: OctInputAdapter(3, patch_size_full=p, image_size=s),
        'output_adapter': lambda p, s, c, d: SpatialOutputAdapter(3, patch_size_full=p, stride_level=1, image_size=s, context_tasks=c, depth=d, task='oct'),
        'loss': lambda p: MaskedMSELoss(patch_size=p),
    },
}

def init_distributed_mode(cfg):
    cfg.rank = int(os.environ['RANK'])
    cfg.world_size = int(os.environ['WORLD_SIZE'])
    cfg.local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(cfg.local_rank)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=cfg.world_size,
        rank=cfg.rank
    )

    dist.barrier()

    print(f"DDP initialized | rank: {cfg.rank}, local_rank: {cfg.local_rank}, world_size: {cfg.world_size}, device: cuda:{cfg.local_rank}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args_cmd, _ = parser.parse_known_args()

    with open(args_cmd.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Pull environment info from torchrun
    cfg['local_rank'] = int(os.environ['LOCAL_RANK'])
    cfg['rank'] = int(os.environ['RANK'])
    cfg['world_size'] = int(os.environ['WORLD_SIZE'])

    cfg['blr'] = float(cfg['blr'])               
    cfg['batch_size'] = int(cfg['batch_size'])  
    total_batch_size = cfg['batch_size'] * cfg['world_size'] 
    cfg['lr'] = cfg['blr'] * total_batch_size / 256
    
    cfg.setdefault('warmup_lr', 1e-6)
    cfg.setdefault('min_lr', 0.0)
    cfg.setdefault('warmup_epochs', 40)
    cfg.setdefault('warmup_steps', -1)
    cfg.setdefault('find_unused_params', True)
    cfg.setdefault('task_balancer', 'none')
    cfg.setdefault('balancer_lr_scale', 1.0)

    return argparse.Namespace(**cfg)


def main():
    cfg = get_args()
    cfg.all_domains = list(set(cfg.in_domains) | set(cfg.out_domains))

    init_distributed_mode(cfg)
    device = torch.device(f'cuda:{cfg.local_rank}')

    print(f"Rank info -> rank: {cfg.rank}, local_rank: {cfg.local_rank}, device: {device}")
    
    torch.manual_seed(cfg.seed + cfg.rank)
    np.random.seed(cfg.seed + cfg.rank)
    os.makedirs(cfg.output_dir, exist_ok=True)

    dataset = build_multimae_pretraining_dataset(cfg)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True
    )

    steps_per_epoch = len(dataloader)
    lr_schedule_values = cosine_scheduler(
        cfg.lr, cfg.min_lr, cfg.epochs, steps_per_epoch,
        warmup_epochs=cfg.warmup_epochs, warmup_steps=cfg.warmup_steps
    )
    wd_schedule_values = cosine_scheduler(
    cfg.weight_decay, cfg.weight_decay, cfg.epochs, steps_per_epoch,
    warmup_epochs=cfg.warmup_epochs, warmup_steps=cfg.warmup_steps
    )

    input_adapters = {k: DOMAIN_CONF[k]['input_adapter'](cfg.patch_size, cfg.input_size) for k in cfg.in_domains}
    output_adapters = {k: DOMAIN_CONF[k]['output_adapter'](cfg.patch_size, cfg.input_size, cfg.in_domains, cfg.decoder_depth) for k in cfg.out_domains}
    if cfg.extra_norm_pix_loss:
        output_adapters['norm_oct'] = SpatialOutputAdapter(
            num_channels=3,
            patch_size_full=cfg.patch_size,
            stride_level=1,
            image_size=cfg.input_size,
            dim_tokens=256,  
            depth=cfg.decoder_depth,
            context_tasks=cfg.in_domains,
            task='oct',
            use_task_queries=True
        )

    model = create_model(
        model_name=cfg.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        use_modality_tokens=True,
        use_masked_attention=True,
        num_global_tokens=cfg.num_global_tokens,
        drop_path_rate=getattr(cfg, 'drop_path', 0.0),
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[cfg.local_rank],
        output_device=cfg.local_rank,
        find_unused_parameters=cfg.find_unused_params
    )

    if cfg.task_balancer == 'uncertainty':
        all_loss_domains = cfg.out_domains + (['norm_oct'] if cfg.extra_norm_pix_loss else [])
        loss_balancer = UncertaintyWeightingStrategy(tasks=all_loss_domains)
    else:
        loss_balancer = NoWeightingStrategy()
    
    loss_balancer.to(device)
    loss_balancer_without_ddp = loss_balancer
    
    if cfg.task_balancer != 'none':
        loss_balancer = torch.nn.parallel.DistributedDataParallel(
            loss_balancer,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            find_unused_parameters=cfg.find_unused_params
        )
        loss_balancer_without_ddp = loss_balancer.module if isinstance(loss_balancer, torch.nn.parallel.DistributedDataParallel) else loss_balancer

    optimizer = create_optimizer(cfg, {
        'model': model.module,
        'balancer': loss_balancer_without_ddp
    })

    loss_scaler = NativeScaler()
    loss_fn = {k: DOMAIN_CONF[k]['loss'](cfg.patch_size) for k in cfg.out_domains}
    if cfg.extra_norm_pix_loss:
        loss_fn['norm_oct'] = MaskedMSELoss(patch_size=cfg.patch_size, norm_pix=True)
    
    log_path = os.path.join(cfg.output_dir, 'loss_log.json')
    loss_log = {}
    if cfg.rank == 0 and os.path.exists(log_path):
        with open(log_path, 'r') as f:
            loss_log = json.load(f)

    for epoch in range(cfg.epochs):
        model.train()
        sampler.set_epoch(epoch)
        metric_logger = MetricLogger()

        for step, batch in enumerate(metric_logger.log_every(dataloader, 10, f"[Epoch {epoch}]")):
            it = epoch * steps_per_epoch + step
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

            inputs = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            preds, masks = model(inputs,
                                mask_inputs=True,
                                num_encoded_tokens=cfg.num_encoded_tokens,
                                alphas=cfg.alphas,
                                sample_tasks_uniformly=cfg.sample_tasks_uniformly)

            task_losses = {}
            for task in preds:
                if task == 'norm_oct':
                    target = inputs['oct'] 
                else:
                    target = inputs[task]

                task_losses[task] = loss_fn[task](preds[task], target, mask=masks.get(task))

            weighted_losses = loss_balancer(task_losses)

            if not isinstance(weighted_losses, dict):
                weighted_losses = {task: weighted_losses[i] for i, task in enumerate(task_losses.keys())}
                
            loss = sum(weighted_losses.values())

            optimizer.zero_grad()
            grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters())
            loss_scale_value = loss_scaler.state_dict()["scale"]
            torch.cuda.synchronize()
            
            loss_value = sum(task_losses.values()).item()
            task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
            weighted_task_loss_values = {f'{task}_loss_weighted': l.item() for task, l in weighted_losses.items()}

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(grad_norm=grad_norm)
            metric_logger.update(**task_loss_values)
            metric_logger.update(**weighted_task_loss_values)

            min_lr, max_lr = float('inf'), float('-inf')
            weight_decay_value = None
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            if weight_decay_value is not None:
                metric_logger.update(weight_decay=weight_decay_value)
        
        if cfg.rank == 0:
            epoch_key = f"epoch_{epoch+1}"
            loss_log[epoch_key] = {k: v.global_avg for k, v in metric_logger.meters.items()}
            with open(log_path, 'w') as f:
                json.dump(loss_log, f, indent=4)

            if (epoch+1) % 100 == 0 or (epoch+1) % 50 == 0:
                save_model_pretrain(
                    cfg=cfg, model=model, model_without_ddp=model.module, optimizer=optimizer,
                    loss_scaler=loss_scaler, loss_balancer=loss_balancer_without_ddp, epoch=epoch)

                print(f"Saved checkpoint")
            
            print(f"Epoch {epoch+1} summary:")
            for k, v in metric_logger.meters.items():
                print(f"  {k}: {v.global_avg:.4f}")


if __name__ == '__main__':
    main()
