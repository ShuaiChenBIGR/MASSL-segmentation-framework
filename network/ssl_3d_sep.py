import torch.nn as nn
import torch
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform(m.weight.data)


class MSSL_norm(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(MSSL_norm, self).__init__()
        # Define basic parameters
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        # Define basic layers
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool3d(kernel_size=2)

        ###############################################################
        # Encoder Level 1 pathway
        self.e_conv3d_l1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                       bias=True)
        self.e_conv3d_l1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                       bias=True)

        # Encoder Level 2 pathway
        self.e_conv3d_l2_1 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                       bias=True)
        self.e_conv3d_l2_2 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 3 pathway
        self.e_conv3d_l3_1 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l3_2 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 4 pathway
        self.e_conv3d_l4_1 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l4_2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 5 pathway
        self.e_conv3d_l5_1 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l5_2 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 16, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        ###############################################################
        self.upscale_conv_norm_lrelu_l0_seg = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                           self.base_n_filter * 8)
        self.upscale_conv_norm_lrelu_l0_rec = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                           self.base_n_filter * 8)
        self.conv3d_l0_seg = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.conv3d_l0_rec = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.inorm3d_l0_seg = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.inorm3d_l0_rec = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l1_ssl = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l2_ssl = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv_norm_lrelu_l4_ssl = self.conv_norm_lrelu(self.base_n_filter, self.base_n_filter)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l4_ssl = nn.Conv3d(self.base_n_filter, self.in_channels, kernel_size=1, stride=1, padding=0,
                                       bias=True)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x, phase, network_switch):

        # Set trainable parameters given Labeled images:
        if phase == 'trainLabeled':
            encoder = network_switch['trainL_encoder']
            decoder_seg = network_switch['trainL_decoder_seg']
            decoder_rec = network_switch['trainL_decoder_rec']

            # Set trainable parameters for Shared Encoder
            for param in self.e_conv3d_l1_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l1_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_2.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = decoder_seg

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = decoder_rec

        # Set trainable parameters given Unlabeled images:
        if phase == 'trainUnlabeled':
            encoder = network_switch['trainU_encoder']
            decoder_seg = network_switch['trainU_decoder_seg']
            decoder_rec = network_switch['trainU_decoder_rec']

            # Set trainable parameters for Shared Encoder
            for param in self.e_conv3d_l1_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l1_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_2.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = decoder_seg

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = decoder_rec

        # Shared Encoder
        out = self.e_conv3d_l1_1(x)  # 16
        out = nn.InstanceNorm3d(self.base_n_filter)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l1_2(out)  # 16
        out = nn.InstanceNorm3d(self.base_n_filter)(out)
        e_level1 = self.lrelu(out)
        out = (e_level1)

        out = self.e_conv3d_l2_1(out)  # 32
        out = nn.InstanceNorm3d(self.base_n_filter * 2)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l2_2(out)  # 32
        out = nn.InstanceNorm3d(self.base_n_filter * 2)(out)
        e_level2 = self.lrelu(out)
        out = (e_level2)

        out = self.e_conv3d_l3_1(out)  # 64
        out = nn.InstanceNorm3d(self.base_n_filter * 4)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l3_2(out)  # 64
        out = nn.InstanceNorm3d(self.base_n_filter * 4)(out)
        e_level3 = self.lrelu(out)
        out = (e_level3)

        out = self.e_conv3d_l4_1(out)  # 128
        out = nn.InstanceNorm3d(self.base_n_filter * 8)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l4_2(out)  # 128
        out = nn.InstanceNorm3d(self.base_n_filter * 8)(out)
        e_level4 = self.lrelu(out)
        out = (e_level4)

        out = self.e_conv3d_l5_1(out)  # 256
        out = nn.InstanceNorm3d(self.base_n_filter * 16)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l5_2(out)  # 256
        out = nn.InstanceNorm3d(self.base_n_filter * 16)(out)
        out_encoder = self.lrelu(out)

        # Decoder 1 for segmentation
        out_d1 = self.upscale_conv_norm_lrelu_l0_seg(out_encoder)  # 256
        out_d1 = self.conv3d_l0_seg(out_d1)  # 256
        out_d1 = self.inorm3d_l0_seg(out_d1)
        out_d1 = self.lrelu(out_d1)
        out_d1 = torch.cat([out_d1, e_level4], dim=1)  # 512

        out_d1 = self.conv_norm_lrelu_l1(out_d1)  # 512
        out_d1 = self.conv3d_l1(out_d1)  # 256
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg(out_d1)  # 128
        out_d1 = torch.cat([out_d1, e_level3], dim=1)  # 256

        out_d1 = self.conv_norm_lrelu_l2(out_d1)  # 256
        out_d1 = self.conv3d_l2(out_d1)  # 128
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg(out_d1)  # 64
        out_d1 = torch.cat([out_d1, e_level2], dim=1)  # 128

        out_d1 = self.conv_norm_lrelu_l3(out_d1)  # 128
        out_d1 = self.conv3d_l3(out_d1)  # 64
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg(out_d1)  # 32
        out_d1 = torch.cat([out_d1, e_level1], dim=1)  # 64

        out_d1 = self.conv_norm_lrelu_l4(out_d1)  # 64
        out_seg = self.conv3d_l4(out_d1)  # 1

        # Decoder 2 for reconstruction
        out_d2 = self.upscale_conv_norm_lrelu_l0_rec(out_encoder)  # 256
        out_d2 = self.conv3d_l0_rec(out_d2)  # 256
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec(out_d2)  # 128
        out_d2 = self.conv3d_l1_ssl(out_d2)  # 128
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec(out_d2)  # 64
        out_d2 = self.conv3d_l2_ssl(out_d2)  # 64
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec(out_d2)  # 32
        out_d2 = self.conv_norm_lrelu_l4_ssl(out_d2)  # 32
        out_rec = self.conv3d_l4_ssl(out_d2)  # 1

        return F.sigmoid(out_seg), F.sigmoid(out_rec)
    
    
class MSSL_norm_bias(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(MSSL_norm_bias, self).__init__()
        # Define basic parameters
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        # Define basic layers
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool3d(kernel_size=2)

        ###############################################################
        # Encoder Level 1 pathway
        self.e_conv3d_l1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                       bias=True)
        self.e_conv3d_l1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                       bias=True)

        # Encoder Level 2 pathway
        self.e_conv3d_l2_1 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                       bias=True)
        self.e_conv3d_l2_2 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 3 pathway
        self.e_conv3d_l3_1 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l3_2 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 4 pathway
        self.e_conv3d_l4_1 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l4_2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 5 pathway
        self.e_conv3d_l5_1 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l5_2 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 16, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        ###############################################################
        self.upscale_conv_norm_lrelu_l0_seg = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                           self.base_n_filter * 8)
        self.upscale_conv_norm_lrelu_l0_rec = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                           self.base_n_filter * 8)
        self.conv3d_l0_seg = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.conv3d_l0_rec = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.inorm3d_l0_seg = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.inorm3d_l0_rec = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l1_ssl = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l2_ssl = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv_norm_lrelu_l4_ssl = self.conv_norm_lrelu(self.base_n_filter, self.base_n_filter)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l4_ssl = nn.Conv3d(self.base_n_filter, self.in_channels, kernel_size=1, stride=1, padding=0,
                                       bias=True)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x, phase, network_switch):

        # Set trainable parameters given Labeled images:
        if phase == 'trainLabeled':
            encoder = network_switch['trainL_encoder']
            decoder_seg = network_switch['trainL_decoder_seg']
            decoder_rec = network_switch['trainL_decoder_rec']

            # Set trainable parameters for Shared Encoder
            for param in self.e_conv3d_l1_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l1_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_2.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = decoder_seg

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = decoder_rec

        # Set trainable parameters given Unlabeled images:
        if phase == 'trainUnlabeled':
            encoder = network_switch['trainU_encoder']
            decoder_seg = network_switch['trainU_decoder_seg']
            decoder_rec = network_switch['trainU_decoder_rec']

            # Set trainable parameters for Shared Encoder
            for param in self.e_conv3d_l1_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l1_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_2.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = decoder_seg

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = decoder_rec

        # Shared Encoder
        out = self.e_conv3d_l1_1(x)  # 16
        out = nn.InstanceNorm3d(self.base_n_filter)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l1_2(out)  # 16
        out = nn.InstanceNorm3d(self.base_n_filter)(out)
        e_level1 = self.lrelu(out)
        out = (e_level1)

        out = self.e_conv3d_l2_1(out)  # 32
        out = nn.InstanceNorm3d(self.base_n_filter * 2)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l2_2(out)  # 32
        out = nn.InstanceNorm3d(self.base_n_filter * 2)(out)
        e_level2 = self.lrelu(out)
        out = (e_level2)

        out = self.e_conv3d_l3_1(out)  # 64
        out = nn.InstanceNorm3d(self.base_n_filter * 4)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l3_2(out)  # 64
        out = nn.InstanceNorm3d(self.base_n_filter * 4)(out)
        e_level3 = self.lrelu(out)
        out = (e_level3)

        out = self.e_conv3d_l4_1(out)  # 128
        out = nn.InstanceNorm3d(self.base_n_filter * 8)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l4_2(out)  # 128
        out = nn.InstanceNorm3d(self.base_n_filter * 8)(out)
        e_level4 = self.lrelu(out)
        out = (e_level4)

        out = self.e_conv3d_l5_1(out)  # 256
        out = nn.InstanceNorm3d(self.base_n_filter * 16)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l5_2(out)  # 256
        out = nn.InstanceNorm3d(self.base_n_filter * 16)(out)
        out_encoder = self.lrelu(out)

        # Decoder 1 for segmentation
        out_d1 = self.upscale_conv_norm_lrelu_l0_seg(out_encoder)  # 256
        out_d1 = self.conv3d_l0_seg(out_d1)  # 256
        out_d1 = self.inorm3d_l0_seg(out_d1)
        out_d1 = self.lrelu(out_d1)
        out_d1 = torch.cat([out_d1, e_level4], dim=1)  # 512

        out_d1 = self.conv_norm_lrelu_l1(out_d1)  # 512
        out_d1 = self.conv3d_l1(out_d1)  # 256
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg(out_d1)  # 128
        out_d1 = torch.cat([out_d1, e_level3], dim=1)  # 256

        out_d1 = self.conv_norm_lrelu_l2(out_d1)  # 256
        out_d1 = self.conv3d_l2(out_d1)  # 128
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg(out_d1)  # 64
        out_d1 = torch.cat([out_d1, e_level2], dim=1)  # 128

        out_d1 = self.conv_norm_lrelu_l3(out_d1)  # 128
        out_d1 = self.conv3d_l3(out_d1)  # 64
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg(out_d1)  # 32
        out_d1 = torch.cat([out_d1, e_level1], dim=1)  # 64

        out_d1 = self.conv_norm_lrelu_l4(out_d1)  # 64
        out_seg = self.conv3d_l4(out_d1)  # 1

        # Decoder 2 for reconstruction
        out_d2 = self.upscale_conv_norm_lrelu_l0_rec(out_encoder)  # 256
        out_d2 = self.conv3d_l0_rec(out_d2)  # 256
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec(out_d2)  # 128
        out_d2 = self.conv3d_l1_ssl(out_d2)  # 128
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec(out_d2)  # 64
        out_d2 = self.conv3d_l2_ssl(out_d2)  # 64
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec(out_d2)  # 32
        out_d2 = self.conv_norm_lrelu_l4_ssl(out_d2)  # 32
        out_rec = self.conv3d_l4_ssl(out_d2)  # 1

        return F.sigmoid(out_seg), F.sigmoid(out_rec)


class MSSL_norm_double_encoder_capacity_unet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(MSSL_norm_double_encoder_capacity_unet, self).__init__()
        # Define basic parameters
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        # Define basic layers
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool3d(kernel_size=2)

        ###############################################################
        # Encoder Level 1 pathway
        self.e_conv3d_l1_1 = nn.Conv3d(self.in_channels, self.base_n_filter * 2, kernel_size=3, stride=1, padding=1,
                                       bias=True)
        self.e_conv3d_l1_2 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=3, stride=1, padding=1,
                                       bias=True)

        # Encoder Level 2 pathway
        self.e_conv3d_l2_1 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                       bias=True)
        self.e_conv3d_l2_2 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 3 pathway
        self.e_conv3d_l3_1 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l3_2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 4 pathway
        self.e_conv3d_l4_1 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l4_2 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 16, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 5 pathway
        self.e_conv3d_l5_1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 32, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l5_2 = nn.Conv3d(self.base_n_filter * 32, self.base_n_filter * 32, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        ###############################################################
        self.upscale_conv_norm_lrelu_l0_seg = self.upscale_conv_norm_lrelu(self.base_n_filter * 32,
                                                                           self.base_n_filter * 8)
        self.upscale_conv_norm_lrelu_l0_rec = self.upscale_conv_norm_lrelu(self.base_n_filter * 32,
                                                                           self.base_n_filter * 8)
        self.conv3d_l0_seg = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.conv3d_l0_rec = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.inorm3d_l0_seg = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.inorm3d_l0_rec = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 24, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l1_ssl = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 12, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l2_ssl = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 6, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 3, self.base_n_filter * 2)
        self.conv_norm_lrelu_l4_ssl = self.conv_norm_lrelu(self.base_n_filter, self.base_n_filter)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l4_ssl = nn.Conv3d(self.base_n_filter, self.in_channels, kernel_size=1, stride=1, padding=0,
                                       bias=True)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x, phase, network_switch):

        # Set trainable parameters given Labeled images:
        if phase == 'trainLabeled':
            encoder = network_switch['trainL_encoder']
            decoder_seg = network_switch['trainL_decoder_seg']
            decoder_rec = network_switch['trainL_decoder_rec']

            # Set trainable parameters for Shared Encoder
            for param in self.e_conv3d_l1_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l1_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_2.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = decoder_seg

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = decoder_rec

        # Set trainable parameters given Unlabeled images:
        if phase == 'trainUnlabeled':
            encoder = network_switch['trainU_encoder']
            decoder_seg = network_switch['trainU_decoder_seg']
            decoder_rec = network_switch['trainU_decoder_rec']

            # Set trainable parameters for Shared Encoder
            for param in self.e_conv3d_l1_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l1_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_2.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = decoder_seg

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = decoder_rec

        # Shared Encoder
        out = self.e_conv3d_l1_1(x)  # 16
        out = nn.InstanceNorm3d(self.base_n_filter)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l1_2(out)  # 16
        out = nn.InstanceNorm3d(self.base_n_filter)(out)
        e_level1 = self.lrelu(out)
        out = (e_level1)

        out = self.e_conv3d_l2_1(out)  # 32
        out = nn.InstanceNorm3d(self.base_n_filter * 2)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l2_2(out)  # 32
        out = nn.InstanceNorm3d(self.base_n_filter * 2)(out)
        e_level2 = self.lrelu(out)
        out = (e_level2)

        out = self.e_conv3d_l3_1(out)  # 64
        out = nn.InstanceNorm3d(self.base_n_filter * 4)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l3_2(out)  # 64
        out = nn.InstanceNorm3d(self.base_n_filter * 4)(out)
        e_level3 = self.lrelu(out)
        out = (e_level3)

        out = self.e_conv3d_l4_1(out)  # 128
        out = nn.InstanceNorm3d(self.base_n_filter * 8)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l4_2(out)  # 128
        out = nn.InstanceNorm3d(self.base_n_filter * 8)(out)
        e_level4 = self.lrelu(out)
        out = (e_level4)

        out = self.e_conv3d_l5_1(out)  # 256
        out = nn.InstanceNorm3d(self.base_n_filter * 16)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l5_2(out)  # 256
        out = nn.InstanceNorm3d(self.base_n_filter * 16)(out)
        out_encoder = self.lrelu(out)

        # Decoder 1 for segmentation
        out_d1 = self.upscale_conv_norm_lrelu_l0_seg(out_encoder)  # 256
        out_d1 = self.conv3d_l0_seg(out_d1)  # 256
        out_d1 = self.inorm3d_l0_seg(out_d1)
        out_d1 = self.lrelu(out_d1)
        out_d1 = torch.cat([out_d1, e_level4], dim=1)  # 512

        out_d1 = self.conv_norm_lrelu_l1(out_d1)  # 512
        out_d1 = self.conv3d_l1(out_d1)  # 256
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg(out_d1)  # 128
        out_d1 = torch.cat([out_d1, e_level3], dim=1)  # 256

        out_d1 = self.conv_norm_lrelu_l2(out_d1)  # 256
        out_d1 = self.conv3d_l2(out_d1)  # 128
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg(out_d1)  # 64
        out_d1 = torch.cat([out_d1, e_level2], dim=1)  # 128

        out_d1 = self.conv_norm_lrelu_l3(out_d1)  # 128
        out_d1 = self.conv3d_l3(out_d1)  # 64
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg(out_d1)  # 32
        out_d1 = torch.cat([out_d1, e_level1], dim=1)  # 64

        out_d1 = self.conv_norm_lrelu_l4(out_d1)  # 64
        out_seg = self.conv3d_l4(out_d1)  # 1

        # Decoder 2 for reconstruction
        out_d2 = self.upscale_conv_norm_lrelu_l0_rec(out_encoder)  # 256
        out_d2 = self.conv3d_l0_rec(out_d2)  # 256
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec(out_d2)  # 128
        out_d2 = self.conv3d_l1_ssl(out_d2)  # 128
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec(out_d2)  # 64
        out_d2 = self.conv3d_l2_ssl(out_d2)  # 64
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec(out_d2)  # 32
        out_d2 = self.conv_norm_lrelu_l4_ssl(out_d2)  # 32
        out_rec = self.conv3d_l4_ssl(out_d2)  # 1

        return F.sigmoid(out_seg), F.sigmoid(out_rec)


class MSSL_norm_double_decoder_capacity_unet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(MSSL_norm_double_decoder_capacity_unet, self).__init__()
        # Define basic parameters
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        # Define basic layers
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool3d(kernel_size=2)

        ###############################################################
        # Encoder Level 1 pathway
        self.e_conv3d_l1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                       bias=True)
        self.e_conv3d_l1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                       bias=True)

        # Encoder Level 2 pathway
        self.e_conv3d_l2_1 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                       bias=True)
        self.e_conv3d_l2_2 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 3 pathway
        self.e_conv3d_l3_1 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l3_2 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 4 pathway
        self.e_conv3d_l4_1 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l4_2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 5 pathway
        self.e_conv3d_l5_1 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l5_2 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 16, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        ###############################################################
        self.upscale_conv_norm_lrelu_l0_seg = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                           self.base_n_filter * 32)
        self.upscale_conv_norm_lrelu_l0_rec = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                           self.base_n_filter * 8)
        self.conv3d_l0_seg = nn.Conv3d(self.base_n_filter * 32, self.base_n_filter * 32, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.conv3d_l0_rec = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.inorm3d_l0_seg = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.inorm3d_l0_rec = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 40, self.base_n_filter * 32)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 32, self.base_n_filter * 16, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l1_ssl = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                                 self.base_n_filter * 8)

        self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 12, self.base_n_filter * 16)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l2_ssl = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 6, self.base_n_filter * 8)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 3, self.base_n_filter * 4)
        self.conv_norm_lrelu_l4_ssl = self.conv_norm_lrelu(self.base_n_filter, self.base_n_filter)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l4_ssl = nn.Conv3d(self.base_n_filter, self.in_channels, kernel_size=1, stride=1, padding=0,
                                       bias=True)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x, phase, network_switch):

        # Set trainable parameters given Labeled images:
        if phase == 'trainLabeled':
            encoder = network_switch['trainL_encoder']
            decoder_seg = network_switch['trainL_decoder_seg']
            decoder_rec = network_switch['trainL_decoder_rec']

            # Set trainable parameters for Shared Encoder
            for param in self.e_conv3d_l1_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l1_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_2.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = decoder_seg

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = decoder_rec

        # Set trainable parameters given Unlabeled images:
        if phase == 'trainUnlabeled':
            encoder = network_switch['trainU_encoder']
            decoder_seg = network_switch['trainU_decoder_seg']
            decoder_rec = network_switch['trainU_decoder_rec']

            # Set trainable parameters for Shared Encoder
            for param in self.e_conv3d_l1_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l1_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_2.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = decoder_seg

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = decoder_rec

        # Shared Encoder
        out = self.e_conv3d_l1_1(x)  # 16
        out = nn.InstanceNorm3d(self.base_n_filter)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l1_2(out)  # 16
        out = nn.InstanceNorm3d(self.base_n_filter)(out)
        e_level1 = self.lrelu(out)
        out = (e_level1)

        out = self.e_conv3d_l2_1(out)  # 32
        out = nn.InstanceNorm3d(self.base_n_filter * 2)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l2_2(out)  # 32
        out = nn.InstanceNorm3d(self.base_n_filter * 2)(out)
        e_level2 = self.lrelu(out)
        out = (e_level2)

        out = self.e_conv3d_l3_1(out)  # 64
        out = nn.InstanceNorm3d(self.base_n_filter * 4)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l3_2(out)  # 64
        out = nn.InstanceNorm3d(self.base_n_filter * 4)(out)
        e_level3 = self.lrelu(out)
        out = (e_level3)

        out = self.e_conv3d_l4_1(out)  # 128
        out = nn.InstanceNorm3d(self.base_n_filter * 8)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l4_2(out)  # 128
        out = nn.InstanceNorm3d(self.base_n_filter * 8)(out)
        e_level4 = self.lrelu(out)
        out = (e_level4)

        out = self.e_conv3d_l5_1(out)  # 256
        out = nn.InstanceNorm3d(self.base_n_filter * 16)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l5_2(out)  # 256
        out = nn.InstanceNorm3d(self.base_n_filter * 16)(out)
        out_encoder = self.lrelu(out)

        # Decoder 1 for segmentation
        out_d1 = self.upscale_conv_norm_lrelu_l0_seg(out_encoder)  # 256
        out_d1 = self.conv3d_l0_seg(out_d1)  # 256
        out_d1 = self.inorm3d_l0_seg(out_d1)
        out_d1 = self.lrelu(out_d1)
        out_d1 = torch.cat([out_d1, e_level4], dim=1)  # 512

        out_d1 = self.conv_norm_lrelu_l1(out_d1)  # 512
        out_d1 = self.conv3d_l1(out_d1)  # 256
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg(out_d1)  # 128
        out_d1 = torch.cat([out_d1, e_level3], dim=1)  # 256

        out_d1 = self.conv_norm_lrelu_l2(out_d1)  # 256
        out_d1 = self.conv3d_l2(out_d1)  # 128
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg(out_d1)  # 64
        out_d1 = torch.cat([out_d1, e_level2], dim=1)  # 128

        out_d1 = self.conv_norm_lrelu_l3(out_d1)  # 128
        out_d1 = self.conv3d_l3(out_d1)  # 64
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg(out_d1)  # 32
        out_d1 = torch.cat([out_d1, e_level1], dim=1)  # 64

        out_d1 = self.conv_norm_lrelu_l4(out_d1)  # 64
        out_seg = self.conv3d_l4(out_d1)  # 1

        # Decoder 2 for reconstruction
        out_d2 = self.upscale_conv_norm_lrelu_l0_rec(out_encoder)  # 256
        out_d2 = self.conv3d_l0_rec(out_d2)  # 256
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec(out_d2)  # 128
        out_d2 = self.conv3d_l1_ssl(out_d2)  # 128
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec(out_d2)  # 64
        out_d2 = self.conv3d_l2_ssl(out_d2)  # 64
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec(out_d2)  # 32
        out_d2 = self.conv_norm_lrelu_l4_ssl(out_d2)  # 32
        out_rec = self.conv3d_l4_ssl(out_d2)  # 1

        return F.sigmoid(out_seg), F.sigmoid(out_rec)


class MSSL_norm_large_AE(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(MSSL_norm_large_AE, self).__init__()
        # Define basic parameters
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        # Define basic layers
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool3d(kernel_size=2)

        ###############################################################
        # Encoder Level 1 pathway
        self.e_conv3d_l1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                       bias=True)
        self.e_conv3d_l1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                       bias=True)

        # Encoder Level 2 pathway
        self.e_conv3d_l2_1 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                       bias=True)
        self.e_conv3d_l2_2 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 3 pathway
        self.e_conv3d_l3_1 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l3_2 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 4 pathway
        self.e_conv3d_l4_1 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l4_2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        # Encoder Level 5 pathway
        self.e_conv3d_l5_1 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True)
        self.e_conv3d_l5_2 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 16, kernel_size=3, stride=1,
                                       padding=1,
                                       bias=True)

        ###############################################################
        self.upscale_conv_norm_lrelu_l0_seg = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                           self.base_n_filter * 8)
        self.upscale_conv_norm_lrelu_l0_rec = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                           self.base_n_filter * 8)

        self.conv3d_l0_seg = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.conv3d_l0_rec = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.inorm3d_l0_seg = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.inorm3d_l0_rec = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l1_ssl = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l2_ssl = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv_norm_lrelu_l4_ssl = self.conv_norm_lrelu(self.base_n_filter, self.base_n_filter)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l4_ssl = nn.Conv3d(self.base_n_filter, self.in_channels, kernel_size=1, stride=1, padding=0,
                                       bias=True)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x, phase, network_switch):

        # Set trainable parameters given Labeled images:
        if phase == 'trainLabeled':
            encoder = network_switch['trainL_encoder']
            decoder_seg = network_switch['trainL_decoder_seg']
            decoder_rec = network_switch['trainL_decoder_rec']

            # Set trainable parameters for Shared Encoder
            for param in self.e_conv3d_l1_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l1_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_2.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = decoder_seg

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = decoder_rec

        # Set trainable parameters given Unlabeled images:
        if phase == 'trainUnlabeled':
            encoder = network_switch['trainU_encoder']
            decoder_seg = network_switch['trainU_decoder_seg']
            decoder_rec = network_switch['trainU_decoder_rec']

            # Set trainable parameters for Shared Encoder
            for param in self.e_conv3d_l1_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l1_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l2_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l3_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l4_2.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_1.parameters():
                param.requires_grad = encoder
            for param in self.e_conv3d_l5_2.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = decoder_seg

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = decoder_rec

        # Shared Encoder
        out = self.e_conv3d_l1_1(x)  # 16
        out = nn.InstanceNorm3d(self.base_n_filter)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l1_2(out)  # 16
        out = nn.InstanceNorm3d(self.base_n_filter)(out)
        e_level1 = self.lrelu(out)
        out = (e_level1)
        # print(e_level1.shape)

        out = self.e_conv3d_l2_1(out)  # 32
        out = nn.InstanceNorm3d(self.base_n_filter * 2)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l2_2(out)  # 32
        out = nn.InstanceNorm3d(self.base_n_filter * 2)(out)
        e_level2 = self.lrelu(out)
        out = (e_level2)

        out = self.e_conv3d_l3_1(out)  # 64
        out = nn.InstanceNorm3d(self.base_n_filter * 4)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l3_2(out)  # 64
        out = nn.InstanceNorm3d(self.base_n_filter * 4)(out)
        e_level3 = self.lrelu(out)
        out = (e_level3)

        out = self.e_conv3d_l4_1(out)  # 128
        out = nn.InstanceNorm3d(self.base_n_filter * 8)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l4_2(out)  # 128
        out = nn.InstanceNorm3d(self.base_n_filter * 8)(out)
        e_level4 = self.lrelu(out)
        out = (e_level4)

        out = self.e_conv3d_l5_1(out)  # 256
        out = nn.InstanceNorm3d(self.base_n_filter * 16)(out)
        out = self.lrelu(out)
        out = self.e_conv3d_l5_2(out)  # 256
        out = nn.InstanceNorm3d(self.base_n_filter * 16)(out)
        out_encoder = self.lrelu(out)

        # Decoder 1 for segmentation
        out_d1 = self.upscale_conv_norm_lrelu_l0_seg(out_encoder)  # 256
        out_d1 = self.conv3d_l0_seg(out_d1)  # 256
        out_d1 = self.inorm3d_l0_seg(out_d1)
        out_d1 = self.lrelu(out_d1)
        out_d1 = torch.cat([out_d1, e_level4], dim=1)  # 512

        out_d1 = self.conv_norm_lrelu_l1(out_d1)  # 512
        out_d1 = self.conv3d_l1(out_d1)  # 256
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg(out_d1)  # 128
        out_d1 = torch.cat([out_d1, e_level3], dim=1)  # 256

        out_d1 = self.conv_norm_lrelu_l2(out_d1)  # 256
        out_d1 = self.conv3d_l2(out_d1)  # 128
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg(out_d1)  # 64
        out_d1 = torch.cat([out_d1, e_level2], dim=1)  # 128

        out_d1 = self.conv_norm_lrelu_l3(out_d1)  # 128
        out_d1 = self.conv3d_l3(out_d1)  # 64
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg(out_d1)  # 32
        out_d1 = torch.cat([out_d1, e_level1], dim=1)  # 64

        out_d1 = self.conv_norm_lrelu_l4(out_d1)  # 64
        out_seg = self.conv3d_l4(out_d1)  # 1

        # Decoder 2 for reconstruction
        out_d2 = self.upscale_conv_norm_lrelu_l0_rec(out_encoder)  # 256
        out_d2 = self.conv3d_l0_rec(out_d2)  # 256
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec(out_d2)  # 128
        out_d2 = self.conv3d_l1_ssl(out_d2)  # 128
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec(out_d2)  # 64
        out_d2 = self.conv3d_l2_ssl(out_d2)  # 64
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec(out_d2)  # 32
        out_d2 = self.conv_norm_lrelu_l4_ssl(out_d2)  # 32
        out_rec = self.conv3d_l4_ssl(e_level1)  # 1

        return F.sigmoid(out_seg), out_rec


class semiSupervised3D_sep(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(semiSupervised3D_sep, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=True)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=True)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=True)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=True)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=True)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
                                   bias=True)
        self.inorm3d_c5 = nn.InstanceNorm3d(self.base_n_filter * 16)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.upscale_conv_norm_lrelu_l0_seg = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                           self.base_n_filter * 8)
        self.upscale_conv_norm_lrelu_l0_rec = self.upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                           self.base_n_filter * 8)
        self.conv3d_l0_seg = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.conv3d_l0_rec = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.inorm3d_l0_seg = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.inorm3d_l0_rec = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l1_ssl = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 4, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                                 self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l2_ssl = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 2, kernel_size=1, stride=1,
                                       padding=0,
                                       bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                                 self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                                 self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv_norm_lrelu_l4_ssl = self.conv_norm_lrelu(self.base_n_filter, self.base_n_filter)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv3d_l4_ssl = nn.Conv3d(self.base_n_filter, self.in_channels, kernel_size=1, stride=1, padding=0,
                                       bias=True)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=True)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True))

    def upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x, phase, network_switch):

        if phase == 'trainLabeled':
            encoder = network_switch['trainL_encoder']
            decoder_seg = network_switch['trainL_decoder_seg']
            decoder_rec = network_switch['trainL_decoder_rec']

            # Shared encoder path
            for param in self.conv3d_c1_1.parameters():
                param.requires_grad = encoder
            for param in self.inorm3d_c1.parameters():
                param.requires_grad = encoder
            for param in self.conv3d_c1_2.parameters():
                param.requires_grad = encoder
            for param in self.conv3d_c2.parameters():
                param.requires_grad = encoder
            for param in self.norm_lrelu_conv_c2.parameters():
                param.requires_grad = encoder
            for param in self.inorm3d_c2.parameters():
                param.requires_grad = encoder
            for param in self.conv3d_c3.parameters():
                param.requires_grad = encoder
            for param in self.norm_lrelu_conv_c3.parameters():
                param.requires_grad = encoder
            for param in self.inorm3d_c3.parameters():
                param.requires_grad = encoder
            for param in self.conv3d_c4.parameters():
                param.requires_grad = encoder
            for param in self.norm_lrelu_conv_c4.parameters():
                param.requires_grad = encoder
            for param in self.inorm3d_c4.parameters():
                param.requires_grad = encoder
            for param in self.conv3d_c5.parameters():
                param.requires_grad = encoder
            for param in self.norm_lrelu_conv_c5.parameters():
                param.requires_grad = encoder
            for param in self.inorm3d_c5.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = True

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = True

        if phase == 'trainUnlabeled':
            encoder = network_switch['trainU_encoder']
            decoder_seg = network_switch['trainU_decoder_seg']
            decoder_rec = network_switch['trainU_decoder_rec']

            # Shared encoder path
            for param in self.conv3d_c1_1.parameters():
                param.requires_grad = encoder
            for param in self.inorm3d_c1.parameters():
                param.requires_grad = encoder
            for param in self.conv3d_c1_2.parameters():
                param.requires_grad = encoder
            for param in self.conv3d_c2.parameters():
                param.requires_grad = encoder
            for param in self.norm_lrelu_conv_c2.parameters():
                param.requires_grad = encoder
            for param in self.inorm3d_c2.parameters():
                param.requires_grad = encoder
            for param in self.conv3d_c3.parameters():
                param.requires_grad = encoder
            for param in self.norm_lrelu_conv_c3.parameters():
                param.requires_grad = encoder
            for param in self.inorm3d_c3.parameters():
                param.requires_grad = encoder
            for param in self.conv3d_c4.parameters():
                param.requires_grad = encoder
            for param in self.norm_lrelu_conv_c4.parameters():
                param.requires_grad = encoder
            for param in self.inorm3d_c4.parameters():
                param.requires_grad = encoder
            for param in self.conv3d_c5.parameters():
                param.requires_grad = encoder
            for param in self.norm_lrelu_conv_c5.parameters():
                param.requires_grad = encoder
            for param in self.inorm3d_c5.parameters():
                param.requires_grad = encoder

            # Decoder path 1 for segmentation
            for param in self.upscale_conv_norm_lrelu_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.inorm3d_l0_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l1.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l2.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l3.parameters():
                param.requires_grad = decoder_seg
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv_norm_lrelu_l4.parameters():
                param.requires_grad = decoder_seg
            for param in self.conv3d_l4.parameters():
                param.requires_grad = True

            # Decoder path 2 for reconstruction
            for param in self.upscale_conv_norm_lrelu_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l0_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l1_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l2_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv_norm_lrelu_l4_ssl.parameters():
                param.requires_grad = decoder_rec
            for param in self.conv3d_l4_ssl.parameters():
                param.requires_grad = True

        # Shared encoder path
        out = self.conv3d_c1_1(x)  # 32
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)  # 32
        out = self.inorm3d_c1(out)
        context_1 = self.lrelu(out)

        out = self.conv3d_c2(context_1)  # 64
        out = self.norm_lrelu_conv_c2(out)  # 64
        out = self.inorm3d_c2(out)
        context_2 = self.lrelu(out)

        out = self.conv3d_c3(context_2)  # 128
        out = self.norm_lrelu_conv_c3(out)  # 128
        out = self.inorm3d_c3(out)
        context_3 = self.lrelu(out)

        out = self.conv3d_c4(context_3)  # 256
        out = self.norm_lrelu_conv_c4(out)  # 256
        out = self.inorm3d_c4(out)
        context_4 = self.lrelu(out)

        out = self.conv3d_c5(context_4)  # 512
        out = self.norm_lrelu_conv_c5(out)  # 512
        out = self.inorm3d_c5(out)
        out_encoder = self.lrelu(out)

        # Decoder 1 for segmentation
        out_d1 = self.upscale_conv_norm_lrelu_l0_seg(out_encoder)  # 256
        out_d1 = self.conv3d_l0_seg(out_d1)  # 256
        out_d1 = self.inorm3d_l0_seg(out_d1)
        out_d1 = self.lrelu(out_d1)
        out_d1 = torch.cat([out_d1, context_4], dim=1)  # 512

        out_d1 = self.conv_norm_lrelu_l1(out_d1)  # 512
        out_d1 = self.conv3d_l1(out_d1)  # 256
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_seg(out_d1)  # 128
        out_d1 = torch.cat([out_d1, context_3], dim=1)  # 256

        out_d1 = self.conv_norm_lrelu_l2(out_d1)  # 256
        out_d1 = self.conv3d_l2(out_d1)  # 128
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_seg(out_d1)  # 64
        out_d1 = torch.cat([out_d1, context_2], dim=1)  # 128

        out_d1 = self.conv_norm_lrelu_l3(out_d1)  # 128
        out_d1 = self.conv3d_l3(out_d1)  # 64
        out_d1 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_seg(out_d1)  # 32
        out_d1 = torch.cat([out_d1, context_1], dim=1)  # 64

        out_d1 = self.conv_norm_lrelu_l4(out_d1)  # 64
        out_seg = self.conv3d_l4(out_d1)  # 1

        # Decoder 2 for reconstruction
        out_d2 = self.upscale_conv_norm_lrelu_l0_rec(out_encoder)  # 256
        out_d2 = self.conv3d_l0_rec(out_d2)  # 256
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l1_rec(out_d2)  # 128
        out_d2 = self.conv3d_l1_ssl(out_d2)  # 128
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l2_rec(out_d2)  # 64
        out_d2 = self.conv3d_l2_ssl(out_d2)  # 64
        out_d2 = self.norm_lrelu_upscale_conv_norm_lrelu_l3_rec(out_d2)  # 32
        out_d2 = self.conv_norm_lrelu_l4_ssl(out_d2)  # 32
        out_rec = self.conv3d_l4_ssl(out_d2)  # 1

        return F.sigmoid(out_seg), F.sigmoid(out_rec)
