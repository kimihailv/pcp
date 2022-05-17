import torch
import torch.nn.functional as F
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from ..models import DGCNNSegBackbone, DGCNNClassification, PointNet, PointNetClassification
from ..datasets import ModelNetDataset, PointCloudNormalize, RandomScale, RandomJitter, RandomRotation, Compose
from tqdm.auto import tqdm
from ..utils.training_routines import RunningMetrics

torch.random.manual_seed(0)
# torch.use_deterministic_algorithms(True)


def get_dataset(model, device, dataset_type, dataset_path, batch_size=10):
    if dataset_type == 'modelnet':
        train_loader = DataLoader(ModelNetDataset(dataset_path, 'train', transform=PointCloudNormalize('box')),
                                  shuffle=False, batch_size=batch_size)
        test_loader = DataLoader(ModelNetDataset(dataset_path, 'test', transform=PointCloudNormalize('box')),
                                 shuffle=False, batch_size=batch_size)

    return get_embeddings(model, train_loader, device), get_embeddings(model, test_loader, device)


@torch.no_grad()
def get_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []

    for x, l in tqdm(loader, desc='Get embeddings'):
        x = x.to(device)
        embs = model.forward_instance(model.forward_features(x), pooling='mean')
        embeddings.append(embs.cpu())
        labels.append(l)

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def train_svm(X, y):
    svm = SVC(kernel='linear')
    perm = np.random.permutation(X.shape[0])
    svm.fit(X[perm], y[perm])
    return svm


def train_eval(model, device, dataset_type, dataset_path, batch_size):
    (X_train, y_train), (X_test, y_test) = get_dataset(model, device, dataset_type, dataset_path, batch_size)
    print('X_train size:', X_train.shape[0], 'X_test size:', X_test.shape[0], 'dim:', X_train.shape[1])
    svm = train_svm(X_train, y_train)
    y_pred = svm.predict(X_test)
    return y_test, y_pred


def train_eval_finetune(model, device, dataset_type, dataset_path, batch_size, finetune_head):
    if dataset_type == 'modelnet':
        n_classes = 40
        train_loader = DataLoader(ModelNetDataset(dataset_path, 'train', transform=PointCloudNormalize('box')),
                                  shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(ModelNetDataset(dataset_path, 'test', transform=PointCloudNormalize('box')),
                                 shuffle=False, batch_size=batch_size)
    elif 'fewshot' in dataset_type:
        _, fewshot_opts = dataset_type.split('|')

        train_loader = DataLoader(ModelNetDataset(dataset_path, 'train', transform=PointCloudNormalize('box'),
                                                  few_shot=fewshot_opts),
                                  shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(ModelNetDataset(dataset_path, 'test', transform=PointCloudNormalize('box'),
                                                 few_shot=fewshot_opts, classes=train_loader.dataset.classes),
                                 shuffle=False, batch_size=batch_size)
        n_classes = len(train_loader.dataset.classes)

    # model = DGCNNClassification(model, n_classes).to(device)
    print(n_classes, train_loader.dataset.labels)
    model = PointNetClassification(model, n_classes).to(device)

    if finetune_head:
        for p in model.backbone.parameters():
            p.requires_grad = False

    n_epochs = 200
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    for epoch in range(n_epochs):
        bar = tqdm(train_loader)
        model.train()
        metrics = RunningMetrics()

        for x, labels in bar:
            optimizer.zero_grad()
            logits = model(x.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            metrics.step({'ce': loss})
            report = metrics.report()
            report.update({'epoch': epoch})
            bar.set_postfix(report)

        scheduler.step()

    model.eval()
    with torch.no_grad():
        preds = []
        labels = []
        for x, l in tqdm(test_loader):
            pred = model(x.to(device)).max(dim=1)[1]
            preds.append(pred.cpu())
            labels.append(l)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

    return labels, preds


def main(opts):
    # model = DGCNNSegBackbone().to(opts.device)
    model = PointNet().to(opts.device)
    if opts.ckp_path is not None:
        model.load_state_dict(torch.load(opts.ckp_path, map_location=opts.device)['model'])
    print('loaded', opts.ckp_path)

    if 'fewshot' not in opts.dataset_type:
        y_test, y_pred = train_eval(model,
                                    opts.device,
                                    opts.dataset_type,
                                    opts.dataset_path,
                                    opts.batch_size)
    else:
        y_test, y_pred = train_eval_finetune(model, opts.device,
                                             opts.dataset_type,
                                             opts.dataset_path,
                                             opts.batch_size,
                                             False)

    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    with open(opts.report_output_file, 'w+') as f:
        print(report, file=f)


def parse_args():
    parser = ArgumentParser(description='Training and evaluating classifier')
    parser.add_argument('--ckp_path',
                        action='store',
                        type=str,
                        help='path to model checkpoint')

    parser.add_argument('--device',
                        action='store',
                        type=str,
                        help='device id')

    parser.add_argument('--dataset_type',
                        action='store',
                        type=str,
                        help='type of dataset')

    parser.add_argument('--dataset_path',
                        action='store',
                        type=str,
                        help='path to dataset')

    parser.add_argument('--batch_size',
                        action='store',
                        type=int,
                        help='bath size')

    parser.add_argument('--report_output_file',
                        action='store',
                        type=str,
                        help='path to output file with report')

    parser.add_argument('--pooling',
                        action='store',
                        type=str,
                        help='how to pool features: mean, max, max_mean')

    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_args()
    main(opts)
