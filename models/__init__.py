from .dgcnn import DGCNNSegBackbone, DGCNNClfBackbone, DGCNNSegmentation, DGCNNClassification
from .ssl import SupCon, MoCo, SimSiam, NNSimSiam, BYOL, prepare_queue_initialization
from .multilevelsimclr import MultiLevelSimCLR
from .pointnet import PointNet, PointNetSeg, PointNetClassification
from .pvcnn import PVCNN, PVCNNSegmentation

models_invertory = {
    'moco': MoCo,
    'supcon': SupCon,
    'simsiam': SimSiam,
    'simclr': MultiLevelSimCLR,
    'nnsimsiam': NNSimSiam,
    'byol': BYOL
}
