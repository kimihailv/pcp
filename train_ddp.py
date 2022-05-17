import wandb
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from .models import models_invertory, DGCNNSegBackbone, DGCNNClfBackbone, PointNet, PVCNN
from .datasets import (ShapeNetDataset,
                       DoubleDataset,
                       RandomRotation,
                       RandomJitter,
                       RandomScale,
                       PointCloudNormalize,
                       Compose)
from torch.utils.data import DataLoader
from .utils.training_routines import RunningMetrics, get_warmup_schedule
from tqdm.auto import tqdm
from json import load
from argparse import ArgumentParser
from .eval.classification import train_eval
from sklearn.metrics import accuracy_score, f1_score


# os.environ['WANDB_MODE'] = 'offline'
# torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def init_process(rank, size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


class Trainer:
    def __init__(self, rank, world_size, opts):
        config = self.load_config(opts.config_path)
        self.save_dir = config['save_dir']
        self.save_every = config['save_every']
        self.validate_every = config['validate_every']
        self.eval_clf_every = config['eval_clf_every']
        self.eval_clf_kwargs = config['eval_clf_kwargs']
        self.device = rank
        self.n_epochs = config['n_epochs']

        transform = Compose(
            PointCloudNormalize(mode='box'),
            RandomScale(low=0.8, high=1.2),
            RandomRotation(-25, 25, 'xyz'),
            RandomJitter(0.01, 0.05)
        )

        config['dataset']['train']['transform'] = transform
        config['dataset']['val']['transform'] = transform

        train_dataset = ShapeNetDataset(**config['dataset']['train'])

        train_dataset = DoubleDataset(train_dataset)

        self.sampler = DistributedSampler(train_dataset,
                                          rank=self.device,
                                          shuffle=True,
                                          num_replicas=world_size)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=config['batch_size']['train'],
                                       shuffle=False,
                                       sampler=self.sampler,
                                       drop_last=True)

        val_dataset = ShapeNetDataset(**config['dataset']['val'])

        val_dataset = DoubleDataset(val_dataset)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=config['batch_size']['val'],
                                     shuffle=True)

        self.framework = config['framework']['type']
        self.backbone_type = config['backbone_type']
        if config['backbone_type'] == 'clf':
            encoder = DGCNNClfBackbone()
        elif config['backbone_type'] == 'seg':
            encoder = DGCNNSegBackbone()
        elif config['backbone_type'] == 'pointnet':
            encoder = PointNet()
        elif config['backbone_type'] == 'pvcnn':
            encoder = PVCNN()

        constructor = models_invertory[self.framework]
        config['framework']['kwargs']['encoder'] = encoder

        if config['framework']['type'] == 'byol':
            config['framework']['kwargs']['n_steps'] = len(self.train_loader) * self.n_epochs

        model = constructor(**config['framework']['kwargs'])

        if config['sync_bn']:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(self.device)
        self.model = DistributedDataParallel(model, device_ids=[self.device])

        config['lr'] *= world_size
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=config['lr'], weight_decay=config['weight_decay'])

        num_iters = len(self.train_loader)
        self.scheduler = get_warmup_schedule(self.optimizer,
                                             num_warmup_steps=config['warmup_epochs'] * num_iters,
                                             num_training_steps=self.n_epochs * num_iters)

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

        total_step = 0
        for step, ((x1, labels), (x2, _)) in bar:
            self.optimizer.zero_grad()
            x1 = x1.to(self.device)
            labels = labels.to(self.device)
            x2 = x2.to(self.device)

            if self.framework in ('moco', 'supcon'):
                loss_value, loss_dict = self.model(x1, x2, labels, update_q=True)
            elif self.framework == 'byol':
                loss_value, loss_dict = self.model(x1, x2, labels, total_step)
            else:
                loss_value, loss_dict = self.model(x1, x2, labels)

            if self.backbone_type == 'pointnet':
                loss_value += 0.001 * self.model.module.encoder.reg

            loss_value.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_step += 1
            if self.device == 0:
                if len(loss_dict) > 1:
                    loss_dict['total_loss'] = loss_value

                metrics.step(loss_dict)
                report = metrics.report()
                report.update({'epoch': epoch})
                bar.set_postfix(report)

                loss = {f'train_batch_{k}': v.item() for k, v in loss_dict.items()}
                loss.update({'epoch': epoch,
                             'step': step,
                             'lr': self.scheduler.get_last_lr()[0]})
                wandb.log(loss)

        if self.device == 0:
            report = metrics.report('train_epoch')
            report.update({'epoch': epoch})
            wandb.log(report)

    @torch.no_grad()
    def validate(self, epoch):
        metrics = RunningMetrics()
        self.model.module.eval()
        bar = tqdm(enumerate(self.val_loader), desc='Val', total=len(self.val_loader))
        for step, ((x1, labels), (x2, _)) in bar:
            x1 = x1.to(self.device)
            labels = labels.to(self.device)
            x2 = x2.to(self.device)

            if self.framework in ('moco', 'supcon'):
                _, loss_dict = self.model.module(x1, x2, labels, update_q=True)
            else:
                _, loss_dict = self.model.module(x1, x2, labels)

            metrics.step(loss_dict)
            report = metrics.report()
            report.update({'epoch': epoch})
            bar.set_postfix(report)

        report = metrics.report('val')
        report.update({'epoch': epoch})
        wandb.log(report)

    @torch.no_grad()
    def eval_clf(self, epoch):
        y_test, y_pred = train_eval(self.model.module.encoder, **self.eval_clf_kwargs)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')

        wandb.log({
            'epoch': epoch,
            'acc': acc,
            'f1': f1
        })
        torch.cuda.empty_cache()

    def train(self):
        for epoch in range(1, self.n_epochs + 1):
            self.train_epoch(epoch)

            if epoch % self.validate_every == 0:
                if self.device == 0:
                    self.validate(epoch)

            if epoch % self.eval_clf_every == 0:
                if self.device == 0:
                    self.eval_clf(epoch)

            if epoch % self.save_every == 0:
                if self.device == 0:
                    state = {
                        'epoch': epoch,
                        'model': self.model.module.encoder.state_dict(),
                    }
                    torch.save(state, f'{self.save_dir}/{self.framework}_run_{wandb.run.id}_ckp_{epoch}.pt')

            dist.barrier()

        if self.device == 0:
            pass
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
