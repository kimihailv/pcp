import wandb
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from .models.diffae import DiffAE2
from .datasets import ShapeNetDataset, ModelNetDataset, PointCloudNormalize, RandomRotation
from torch.utils.data import DataLoader
from .utils.training_routines import RunningMetrics
from tqdm.auto import tqdm
from json import load
from argparse import ArgumentParser
from copy import deepcopy

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
# os.environ['WANDB_MODE'] = 'offline'


def init_process(rank, size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


def gather(tensor):
    result = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(result, tensor)
    result[dist.get_rank()] = tensor
    return torch.cat(result, dim=0)


def diffusion_loss(e, et):
    loss = (e - et).pow(2).mean()
    return loss, {'diff': loss}


def log_pointclouds(x, log_name):
    x = x.cpu().transpose(2, 1).numpy()

    for pc in x:
        wandb.log({log_name: wandb.Object3D(pc)})


class EMA:
    def __init__(self, model, beta):
        self.module = deepcopy(model)
        self.beta = beta

    @torch.no_grad()
    def update(self, model):
        for ema_p, m_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.beta).add_(m_p.data, alpha=1 - self.beta)


class Trainer:
    def __init__(self, rank, world_size, opts):
        config = self.load_config(opts.config_path)
        self.save_dir = config['save_dir']
        self.save_every = config['save_every']
        self.validate_every = config['validate_every']
        self.device = rank
        self.n_epochs = config['n_epochs']

        train_dataset = ShapeNetDataset(config['dataset_path'], ['val'], ['airplane'],
                                        transform=PointCloudNormalize(mode='shape_unit'))

        self.sampler = DistributedSampler(train_dataset,
                                          rank=self.device,
                                          shuffle=True,
                                          num_replicas=world_size)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=config['batch_size']['train'],
                                       shuffle=False,
                                       sampler=self.sampler,
                                       drop_last=True)

        self.val_loader = DataLoader(ShapeNetDataset(config['dataset_path'], ['val'], ['airplane'],
                                                     transform=PointCloudNormalize(mode='shape_unit')),
                                     batch_size=config['batch_size']['val'],
                                     shuffle=True)

        model = DiffAE2(len(self.train_loader) * self.n_epochs).to(self.device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(self.device)
        self.model = DistributedDataParallel(model, device_ids=[self.device])

        if self.device == 0:
            self.ema = EMA(self.model.module, 0.999)

        config['lr'] *= world_size
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config['lr'], weight_decay=0)

        num_iters = len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs * num_iters,
                                                                    eta_min=1e-4)

        if self.device == 0:
            os.environ['WANDB_API_KEY'] = config['wandb_api_key']
            wandb.init(project="diploma", entity="kimihailv", config=config)
            wandb.run.name = f"{config['run_name']}_{wandb.run.id}"
            wandb.watch(models=[self.model])

    def train_epoch(self, epoch):
        self.model.train()
        self.sampler.set_epoch(epoch - 1)
        bar = enumerate(self.train_loader)

        if self.device == 0:
            bar = tqdm(bar, desc='Train', total=len(self.train_loader))
            metrics = RunningMetrics()

        for step, (x, _) in bar:
            self.optimizer.zero_grad()
            x = x.to(self.device)
            e, (et, reg) = self.model(x)
            loss_value, loss = diffusion_loss(e, et)
            loss_value = loss_value + reg
            loss_value.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.device == 0:
                # loss['reg'] = reg
                self.ema.update(self.model.module)
                metrics.step(loss)
                report = metrics.report()
                report.update({'epoch': epoch})
                bar.set_postfix(report)

                loss = {f'train_batch_{k}': v.item() for k, v in loss.items()}
                loss.update({'epoch': epoch,
                             'step': step,
                             'lr': self.scheduler.get_last_lr()[0]})
                wandb.log(loss)

        if self.device == 0:
            report = metrics.report('train_epoch')
            report.update({'epoch': epoch})
            wandb.log(report)

    @torch.no_grad()
    def validate(self, epoch, x1, x2):
        # metrics = RunningMetrics()
        self.model.module.eval()

        '''for x, _ in tqdm(self.val_loader, desc='Validation'):
            x = x.to(self.device)
            x_rec = self.model.module.auto_encode(x, deterministic=False)
            diff = (x_rec - x).pow(2).mean()
            metrics.step({'diff_rec': diff})

        report = metrics.report('val')
        report.update({'epoch': epoch})
        wandb.log(report)'''

        x_rec = self.model.module.auto_encode(x1, deterministic=False, use_tqdm=True, steps=400)
        log_pointclouds(x_rec, f'rec_{epoch}')

        x_int = self.model.module.interpolate(x1, x2, 0.5, steps=400, use_tqdm=True)
        log_pointclouds(x_int, f'int_{epoch}')

    def train(self):
        if self.device == 0:
            int_batch1 = next(iter(self.val_loader))[:8][0].to(self.device)
            int_batch2 = int_batch1[torch.randperm(8)].to(self.device)
            log_pointclouds(int_batch1, 'int_batch1')
            log_pointclouds(int_batch2, 'int_batch2')

        for epoch in range(1, self.n_epochs + 1):
            self.train_epoch(epoch)

            if epoch % self.validate_every == 0:
                if self.device == 0:
                    self.validate(epoch, int_batch1, int_batch2)

            if epoch % self.save_every == 0:
                if self.device == 0:
                    state = {
                        'epoch': epoch,
                        'model': self.model.module.state_dict(),
                        'ema': self.ema.module.state_dict()
                    }
                    torch.save(state, f'{self.save_dir}/diffae_run_{wandb.run.id}_ckp_{epoch}.pt')


            dist.barrier()

        if self.device == 0:
            wandb.finish()

        dist.destroy_process_group()

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return load(f)


def worker(rank, world_size, opts):
    init_process(rank, world_size)
    Trainer(rank, world_size, opts).train()


if __name__ == '__main__':
    opts = ArgumentParser()
    opts.add_argument('--config_path',
                      action='store',
                      type=str,
                      help='path to config file')

    opts.add_argument('--gpus',
                      action='store',
                      type=int,
                      help='the number of gpus')
    opts = opts.parse_args()

    try:
        mp.spawn(worker,
                 args=(opts.gpus, opts),
                 nprocs=opts.gpus,
                 join=True)
    except KeyboardInterrupt:
        dist.destroy_process_group()
        wandb.finish()
