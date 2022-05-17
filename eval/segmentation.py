import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models import DGCNNSegBackbone, DGCNNSegmentation, PointNetSeg, PointNet
from ..datasets import ShapeNetPartDataset, PointCloudNormalize
from ..utils.training_routines import RunningMetrics
from collections import defaultdict

# from https://github.com/lulutang0608/Point-BERT/blob/master/segmentation/test_partseg.py
N_PARTS = 50
N_CLASSES = 16

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


@torch.no_grad()
def calc_metrics(model, loader, device):
    model.eval()
    test_metrics = {}
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(N_PARTS)]
    total_correct_class = [0 for _ in range(N_PARTS)]
    shape_ious = {cat: [] for cat in seg_classes.keys()}

    for batch_id, (points, label, target) in tqdm(enumerate(loader), total=len(loader)):
        cur_batch_size, _, NUM_POINT = points.size()
        points, label, target = points.to(device), label.long().to(device), target.long()

        seg_pred = model(points.to(device), F.one_hot(label, N_CLASSES).float().to(device)).transpose(2, 1)

        cur_pred_val = seg_pred.cpu().data.numpy()
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        target = target.data.numpy()

        for i in range(cur_batch_size):
            cat = seg_label_to_cat[target[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

        correct = np.sum(cur_pred_val == target)
        total_correct += correct
        total_seen += (cur_batch_size * NUM_POINT)

        for l in range(N_PARTS):
            total_seen_class[l] += np.sum(target == l)
            total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i, :]
            segl = target[i, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (
                        np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])

    mean_shape_ious = np.mean(list(shape_ious.values()))
    test_metrics['accuracy'] = total_correct / float(total_seen)
    test_metrics['class_avg_accuracy'] = np.mean(
        np.array(total_correct_class) / np.array(total_seen_class, dtype=float))

    for cat in sorted(shape_ious.keys()):
        print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))

    test_metrics['class_avg_iou'] = mean_shape_ious
    test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

    print('Accuracy is: %.5f' % test_metrics['accuracy'])
    print('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    print('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    print('Instance avg mIOU is: %.5f' % test_metrics['instance_avg_iou'])

    return test_metrics


def finetune(backbone_ckp_path, dataset_path, batch_size,
             lr, weight_decay, n_epochs, sample_frac, seed,
             validate_every, device, finetune_head, different_lr):
    #backbone = DGCNNSegBackbone()
    backbone = PointNet()
    if backbone_ckp_path is not None:
        backbone_state = torch.load(backbone_ckp_path, map_location='cpu')['model']
        backbone.load_state_dict(backbone_state)
        print('loaded ckp', backbone_ckp_path)
    model = PointNetSeg(backbone, N_PARTS, N_CLASSES).to(device)
    #model = DGCNNSegmentation(backbone, N_PARTS, N_CLASSES, head='mlp').to(device)

    if finetune_head:
        for p in model.backbone.parameters():
            p.requires_grad = False

    train_loader = DataLoader(ShapeNetPartDataset(dataset_path, 'train',
                                                  n_points=2048,
                                                  transform=PointCloudNormalize(mode='box'),
                                                  use_cache=False, sample_frac=sample_frac, seed=seed),
                              batch_size=batch_size, num_workers=3, shuffle=True)

    val_loader = DataLoader(ShapeNetPartDataset(dataset_path, 'val',
                                                n_points=2048,
                                                transform=PointCloudNormalize(mode='box')),
                            batch_size=batch_size, num_workers=3)

    test_loader = DataLoader(ShapeNetPartDataset(opts.dataset_path, 'test', n_points=-1,
                                                 transform=PointCloudNormalize(mode='box')))

    if different_lr and not finetune_head:
        optimizer = torch.optim.Adam([{'params': model.backbone.parameters(), 'lr': 1e-5, 'weight_decay': 0},
                                      {'params': model.head.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                                      ])
        # {'params': model.category_embedding.parameters(),
        #                                        'lr': lr, 'weight_decay': weight_decay}
    else:
        print(len(list(filter(lambda x: x.requires_grad, model.parameters()))))
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                     lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    loss = torch.nn.CrossEntropyLoss()

    if sample_frac == 1:
        calc_metrics(model, val_loader, device)

    for epoch in range(1, n_epochs + 1):
        bar = tqdm(train_loader)
        running_loss = RunningMetrics()
        model.train()

        for x, labels, targets in bar:
            optimizer.zero_grad()
            x = x.to(device)
            labels = F.one_hot(labels.long(), N_CLASSES).float().to(device)
            targets = targets.long().to(device)
            logits = model(x, labels)
            loss_t = loss(logits, targets) + 0.001 * model.backbone.reg
            loss_t.backward()
            optimizer.step()
            running_loss.step({'loss': loss_t})
            report = running_loss.report()
            report.update({'epoch': epoch, 'reg': model.backbone.reg.item(), 'lr': scheduler.get_last_lr()[0]})
            bar.set_postfix(report)

        scheduler.step()
        if epoch % validate_every == 0 and sample_frac == 1:
            calc_metrics(model, val_loader, device)
            calc_metrics(model, test_loader, device)

    return model


def parse_args():
    parser = ArgumentParser(description='Training and evaluating segmentation')
    parser.add_argument('--ckp_path',
                        action='store',
                        type=str,
                        help='path to model checkpoint')

    parser.add_argument('--dataset_path',
                        action='store',
                        type=str,
                        help='path to dataset')

    parser.add_argument('--batch_size',
                        action='store',
                        type=int,
                        default=16,
                        help='batch size')

    parser.add_argument('--lr',
                        action='store',
                        type=float,
                        default=0.1,
                        help='learning rate')

    parser.add_argument('--weight_decay',
                        action='store',
                        type=float,
                        default=1e-4,
                        help='weight decay')

    parser.add_argument('--n_epochs',
                        action='store',
                        type=int,
                        default=200,
                        help='the number of epochs')

    parser.add_argument('--save_path',
                        action='store',
                        type=str,
                        help='path where model will be saved')

    parser.add_argument('--validate_every',
                        action='store',
                        type=int,
                        default=10,
                        help='the period of validation')

    parser.add_argument('--device',
                        action='store',
                        type=str,
                        help='device id')

    parser.add_argument('--report_output_file',
                        action='store',
                        type=str,
                        help='path to output file with report')

    parser.add_argument('--finetune_head',
                        action='store_true',
                        default=False)

    parser.add_argument('--different_lr',
                        action='store_true',
                        default=False)

    parser.add_argument('--sample_frac',
                        action='store',
                        type=float,
                        default=1)

    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_args()
    exp_id = opts.ckp_path.split('_')[2] if opts.ckp_path is not None else 'none'
    report_output_file = opts.report_output_file.replace('.txt', '')
    report_output_file = f'{report_output_file}_{exp_id}.txt'

    if opts.sample_frac == 1:
        model = finetune(opts.ckp_path, opts.dataset_path, opts.batch_size,
                         opts.lr, opts.weight_decay, opts.n_epochs, opts.sample_frac, 42,
                         opts.validate_every, opts.device, opts.finetune_head, opts.different_lr)
        torch.save(model.state_dict(), f'{opts.save_path}/seg_{exp_id}.pt')

        test_loader = DataLoader(ShapeNetPartDataset(opts.dataset_path, 'test', n_points=-1,
                                                     transform=PointCloudNormalize(mode='box')),
                                 batch_size=1, num_workers=3)
        metrics = calc_metrics(model, test_loader, opts.device)

        with open(report_output_file, 'w+') as f:
            print(metrics, file=f)

    else:
        test_loader = DataLoader(ShapeNetPartDataset(opts.dataset_path, 'test', n_points=-1,
                                                     transform=PointCloudNormalize(mode='box')),
                                 batch_size=1, num_workers=3)
        r = np.random.RandomState(423425)
        seeds = r.permutation(100000)[:5]
        print(seeds)
        results = defaultdict(lambda: [])
        for seed in seeds:
            model = finetune(opts.ckp_path, opts.dataset_path, opts.batch_size,
                             opts.lr, opts.weight_decay, opts.n_epochs, opts.sample_frac, seed,
                             opts.validate_every, opts.device, opts.finetune_head, opts.different_lr)

            metrics = calc_metrics(model, test_loader, opts.device)
            for k, v in metrics.items():
                results[k].append(v)

        with open(report_output_file, 'w+') as f:
            for k in results:
                print(k, 'mean:', np.mean(results[k]), 'std:', np.std(results[k], ddof=1), file=f)
