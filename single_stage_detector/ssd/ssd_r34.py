import os
import torch
import torch.nn as nn
from base_model import ResNet34
from torch.utils import mkldnn as mkldnn_utils

class SSD_R34(nn.Module):
    """
        Build a SSD module to take 300x300 image input,
        and output 8732 per class bounding boxes

        vggt: pretrained vgg16 (partial) model
        label_num: number of classes (including background 0)
    """
    def __init__(self, label_num, backbone='resnet34', model_path="./resnet34-333f7ec4.pth",strides=[3,3 ,2 ,2 ,2 ,2]):

        super(SSD_R34, self).__init__()

        self.label_num = label_num
        self.strides = strides
        if backbone == 'resnet34':
            self.model = ResNet34()
            if os.environ.get('USE_MKLDNN') == "1":
                self.model = mkldnn_utils.to_mkldnn(self.model)
            out_channels = 256
            self.out_chan = [out_channels, 512, 512, 256, 256, 256]
        else:
            raise ValueError('Invalid backbone chosen')

        self._build_additional_features(self.out_chan)

        # after l2norm, conv7, conv8_2, conv9_2, conv10_2, conv11_2
        # classifer 1, 2, 3, 4, 5 ,6

        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []
        for nd, oc in zip(self.num_defaults, self.out_chan):
            self.loc.append(nn.Conv2d(oc, nd*4, kernel_size=3, padding=1,stride=self.strides[0]))
            self.conf.append(nn.Conv2d(oc, nd*label_num, kernel_size=3, padding=1,stride=self.strides[1]))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        # intitalize all weights
        self._init_weights()
        if os.environ.get('USE_MKLDNN') == "1":
            self.additional_blocks = mkldnn_utils.to_mkldnn(self.additional_blocks)
            self.loc = mkldnn_utils.to_mkldnn(self.loc)
            self.conf = mkldnn_utils.to_mkldnn(self.conf)

    def _build_additional_features(self, input_channels):
        idx = 0
        self.additional_blocks = []
        
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, input_channels[idx+1], kernel_size=3, padding=1,stride=self.strides[2]),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, input_channels[idx+1], kernel_size=3, padding=1, stride=self.strides[3]),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # conv9_1, conv9_2
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_channels[idx+1], kernel_size=3, padding=1, stride=self.strides[4]),
            nn.ReLU(inplace=True),
        ))
        idx += 1

        # conv10_1, conv10_2
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_channels[idx+1], kernel_size=3,stride=self.strides[5]),
            nn.ReLU(inplace=True),
        ))
        idx += 1



        # conv11_1, conv11_2
        self.additional_blocks.append(nn.Sequential(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, input_channels[idx+1], kernel_size=3),
            nn.ReLU(inplace=True),
        ))

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):

        layers = [
            *self.additional_blocks,
            *self.loc, *self.conf]

        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf,extract_shapes=False):
        ret = []
        features_shapes = []
        for s, l, c in zip(src, loc, conf):
            if os.environ.get('USE_MKLDNN') == "1":
                ret.append((l(s).to_dense().view(s.size(0), 4, -1), c(s).to_dense().view(s.size(0), self.label_num, -1)))
            else:
                ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))
            # extract shapes for prior box initliziation 
            if extract_shapes:
                ls=l(s)
                features_shapes.append([ls.shape[2],ls.shape[3]])
        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs,features_shapes

    def forward(self, data,extract_shapes=False):
        if os.environ.get('USE_MKLDNN') == "1":
            layers = self.model(data.to_mkldnn())
        else:
            layers = self.model(data)

        # last result from network goes into additional blocks
        x = layers[-1]
        
        additional_results = []
        for i, l in enumerate(self.additional_blocks):
            
            x = l(x)
            additional_results.append(x)

        src = [*layers, *additional_results]
        # Feature maps sizes depend on the image size. For 300x300 with strides=[1,1,2,2,2,1] it is 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4 
        locs, confs,features_shapes = self.bbox_view(src, self.loc, self.conf,extract_shapes=extract_shapes)
        # For SSD 300 with strides=[1,1,2,2,2,1] , shall return nbatch x 8732 x {nlabels, nlocs} results 
        return locs, confs,features_shapes
