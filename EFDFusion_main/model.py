import torch
import torch.nn as nn
from loss import LossFunction, LowLightEnhancementLoss

class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.vis_in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.vis_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1,
                      padding=padding),
            nn.ReLU()
        )
        self.vis_out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.vis_blocks = nn.ModuleList()
        for i in range(layers):
            self.vis_blocks.append(self.vis_conv)

    def forward(self, low_input,input):

        vis_fea = self.vis_in_conv(input)
        for vis_conv in self.vis_blocks:
            vis_fea = vis_fea + vis_conv(vis_fea)
        illu_fea = self.vis_out_conv(vis_fea)
        i = torch.clamp(illu_fea, 0.000001, 1)
        r = low_input / i
        r = torch.clamp(r, 0, 1)
        return i, r


class FuseNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(FuseNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.fuse_in_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1,
                      padding=padding),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fuse_out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.fuse_blocks = nn.ModuleList()
        for i in range(layers):
            self.fuse_blocks.append(self.fuse_conv)

    def forward(self, input, inf):

        fuse_input = torch.cat([input, inf], 1)
        fuse_fea = self.fuse_in_conv(fuse_input)
        for fuse_conv in self.fuse_blocks:
            fuse_fea = fuse_fea + fuse_conv(fuse_fea)
        n = self.fuse_out_conv(fuse_fea)
        diff = n * inf*input*(1-input)
        fuse = input+diff
        fuse = torch.clamp(fuse, 0, 1)
        return n, diff, fuse


class CalibrateNetwork_1(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork_1, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.in_conv_1 = self._conv_block(3, channels, kernel_size, padding)
        self.in_conv_2 = self._conv_block(1, channels, kernel_size, padding)
        self.conv_block_1 = self._conv_block(channels, channels, kernel_size, padding, repetitions=2)
        self.conv_block_2 = self._conv_block(channels, channels, kernel_size, padding, repetitions=2)
        self.out_conv_1 = self._conv_block(channels, 3, 3, 1, final_layer=True)
        self.out_conv_2 = self._conv_block(channels, 1, 3, 1, final_layer=True)
        self.blocks_in_1 = nn.ModuleList([self.conv_block_1 for _ in range(layers)])
        self.blocks_out_1 = nn.ModuleList([self.conv_block_1 for _ in range(layers)])
        self.blocks_in_2 = nn.ModuleList([self.conv_block_2 for _ in range(layers)])
        self.blocks_out_2 = nn.ModuleList([self.conv_block_2 for _ in range(layers)])

    def _conv_block(self, in_channels, out_channels, kernel_size, padding, repetitions=1, final_layer=False):
        layers = []
        for _ in range(repetitions):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding))
            layers.append(nn.ReLU())
            in_channels = out_channels
        if final_layer:
            layers[-1] = nn.Sigmoid()
        return nn.Sequential(*layers)

    def _process_through_blocks(self, input, blocks):
        output = input
        for block in blocks:
            output = output + block(output)
        return output

    def forward(self, enh_input, fuse_input, inf_input):

        fuse_fea = self._process_through_blocks(self.in_conv_1(fuse_input), self.blocks_in_1)
        enh_fea = self._process_through_blocks(self.in_conv_1(enh_input), self.blocks_in_1)
        inf_fea = self._process_through_blocks(self.in_conv_2(inf_input), self.blocks_in_2)
        enh = self.out_conv_1(self._process_through_blocks(enh_fea, self.blocks_out_1))
        enh_N = self.out_conv_1(self._process_through_blocks(inf_fea, self.blocks_out_1))
        inf = self.out_conv_2(self._process_through_blocks(inf_fea, self.blocks_out_2))
        inf_enh = self.out_conv_2(self._process_through_blocks(enh_fea, self.blocks_out_2))
        d = self.out_conv_2(self._process_through_blocks(fuse_fea - enh_fea, self.blocks_out_2))
        enh = torch.clamp(enh, 0, 1)
        inf = torch.clamp(inf, 0, 1)
        d = torch.clamp(d, 0, 1)
        enh_N = torch.clamp(enh_N, 0, 1)
        inf_enh = torch.clamp(inf_enh, 0, 1)
        return enh, inf, d, enh_N, inf_enh


class Network(nn.Module):

    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.fuse = FuseNetwork(layers=1, channels=3)
        self.calibrate_1 = CalibrateNetwork_1(layers=2, channels=16)
        self._criterion = LossFunction()
        self._l1loss = torch.nn.L1Loss()
        self._enhloss = LowLightEnhancementLoss()
        self.rgb_to_ycbcr_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ]).float()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input, ir):
        inlist, ilist, nlist, dlist, rlist, fulist, inf_oplist, enhlist, inflist, enh_Nlist, inf_enhlist, difflist = [], [], [], [], [], [], [], [], [], [], [], []

        vis_input_op = input
        inf_op = ir

        for i in range(self.stage):
            inlist.append(vis_input_op)
            inf_oplist.append(inf_op)
            i, r = self.enhance(input,vis_input_op)
            n, d, fuse = self.fuse(r.detach(), inf_op)
            enh, inf, diff, enh_N, inf_enh = self.calibrate_1(r.detach(), fuse, inf_op)
            vis_input_op = input + r - enh
            vis_input_op = torch.clamp(vis_input_op, 0, 1)
            inf_op = ir + inf_op - inf
            inf_op = torch.clamp(inf_op, 0, 1)

            ilist.append(i)
            nlist.append(n)
            dlist.append(d)
            rlist.append(r)
            fulist.append(fuse)
            enhlist.append(enh)
            enh_Nlist.append(enh_N)
            inflist.append(inf)
            inf_enhlist.append(inf_enh)
            difflist.append(diff)

        return inlist, ilist, nlist, dlist, rlist, fulist, inf_oplist, enhlist, inflist, enh_Nlist, inf_enhlist, difflist

    def _loss(self, input, ir):
        inlist, ilist, nlist, dlist, rlist, fulist, inf_oplist, enhlist, inflist, enh_Nlist, inf_enhlist, difflist = self(
            input, ir)
        loss = 0

        eps = 1e-9
        input = input + eps
        luminance = input[:, 0, :, :] * 0.299 + input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.144
        mean_luminance = torch.mean(luminance, dim=(1, 2))
        luminance_scale = 0.5/ (mean_luminance + eps)
        luminance_scale = luminance_scale.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        luminance_scale = torch.clamp(luminance_scale, 1, 25)

        for i in range(self.stage):
            loss += 0.5 * self._enhloss(input, ilist[i], luminance_scale)
            loss += 10 * self._l1loss(inf_oplist[i], ir)
            loss += 20 * self._l1loss(inlist[i], input)
            loss += 1 * self._l1loss(enhlist[i], rlist[i])
            loss += 1 * self._l1loss(enh_Nlist[i], rlist[i])
            loss += 1 * self._l1loss(inflist[i], inf_oplist[i])
            loss += 1 * self._l1loss(inf_enhlist[i], inf_oplist[i])
            loss += 0.1 * self._l1loss(difflist[i], inf_oplist[i])
        return loss

    def rgb_to_ycbcr(self, rgb_img):
        if rgb_img.is_cuda:
            self.rgb_to_ycbcr_matrix = self.rgb_to_ycbcr_matrix.cuda()
        rgb_img = rgb_img.permute(0, 2, 3, 1)
        ycbcr_img = torch.matmul(rgb_img, self.rgb_to_ycbcr_matrix.T)
        ycbcr_img = ycbcr_img.permute(0, 3, 1, 2)
        return ycbcr_img


class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.fuse = FuseNetwork(layers=1, channels=3)
        self._enhloss = LowLightEnhancementLoss()
        self.rgb_to_ycbcr_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ]).float()
        self._l1loss = torch.nn.L1Loss()

        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
    def rgb_to_ycbcr(self, rgb_img):

        if rgb_img.is_cuda:
            self.rgb_to_ycbcr_matrix = self.rgb_to_ycbcr_matrix.cuda()
        rgb_img = rgb_img.permute(0, 2, 3, 1)
        ycbcr_img = torch.matmul(rgb_img, self.rgb_to_ycbcr_matrix.T)
        ycbcr_img = ycbcr_img.permute(0, 3, 1, 2)
        return ycbcr_img

    def forward(self, input1,input2, ir):
        i, r = self.enhance(input1,input2)
        n, d, fuse = self.fuse(r.detach(), ir)
        return i,  r, n,d,fuse

    def _loss(self, input1,input2, ir):
        i, r, n, d, fuse=self(input1,input2, ir)
        loss = 0
        eps = 1e-9
        input = input1 + eps
        luminance = input[:, 0, :, :] * 0.299 + input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.144
        mean_luminance = torch.mean(luminance, dim=(1, 2))
        luminance_scale = 0.5 / (mean_luminance + eps)
        luminance_scale = luminance_scale.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        luminance_scale = torch.clamp(luminance_scale, 1, 25)
        loss += 0.05 * self._enhloss(input, i, luminance_scale)

        return loss
