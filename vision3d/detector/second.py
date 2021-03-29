from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
import spconv

from .layers import VoxelFeatureExtractor
from .sparse_cnn import SpMiddleFHD, SpMiddleFHDLite
from .proposal import ProposalLayer


class Second(nn.Module):

    def __init__(self, cfg):
        super(Second, self).__init__()
        self.vfe = VoxelFeatureExtractor()
        self.cnn = Middle(cfg)
        self.rpn = RPN()
        self.head = ProposalLayer(cfg)
        self.cfg = cfg

    def feature_extract(self, item):
        features = self.vfe(item['features'], item['occupancy'])
        features = self.cnn(features, item['coordinates'], item['batch_size'])
        features = self.rpn(features)
        return features

    def forward(self, item):
        features = self.feature_extract(item)
        scores, boxes = self.head(features)
        item.update(dict(P_cls=scores, P_reg=boxes))
        return item

    def inference(self, item):
        features = self.feature_extract(item)
        out = self.head.inference(features, item['anchors'])
        return out


class Middle(SpMiddleFHD):
    """Skips conversion to metric coordinates."""

    def forward(self, features, coordinates, batch_size):
        x = spconv.SparseConvTensor(
            features, coordinates.int(), self.grid_shape, batch_size
        )
        x = self.to_bev(self.blocks(x))
        return x


class RPN(nn.Module):
    """OneStage RPN from SECOND."""

    def __init__(self, C_in=128, C_up=128, C_down=128, blocks=5):
        super(RPN, self).__init__()
        self.down_block, C_in = self._make_down_block(C_in, C_down, blocks)
        self.up_block = self._make_up_block(C_in, C_up)
        self._init_weights()

    def _make_down_block(self, inplanes, planes, num_blocks, stride=1):
        block = [
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ]
        for j in range(num_blocks):
            block += [
                nn.Conv2d(planes, planes, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
        return nn.Sequential(*block), planes

    def _make_up_block(self, inplanes, planes, stride=1):
        block = nn.Sequential(
            nn.Conv2d(inplanes, planes, stride, stride=stride, bias=False),
            nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        return block

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, _BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.down_block(x)
        x = self.up_block(x)
        return x
