import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import functional as F
from module import upsample2, P_Conv, N_Conv, PNP_Conv_operation, rescale_module,\
                    LSTM, SeparableConvBlock, MemoryEfficientSwish, Swish
import pdb

SAMPLE_RATE = 16000
N_CLASS = 360
N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 1024
CONST = 1997.3794084376191

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale

class Snake(nn.Module):
    def __init__(self, a=5):
        super(Snake, self).__init__()
        self.a = a
    def forward(self, x):
        return (x + (torch.sin(self.a * x) ** 2) / self.a)

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])
        x = self.conv(x)
        return x

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        output, _ = self.gru(x)
        return output


class HybridPitchEstimator(nn.Module):
    def __init__(self, chin=1, hidden=6, kernel_size=4, stride=4, depth=5,
                 normalize=True, floor=1e-3, pfactor=[17, 13, 11, 7, 5], npfactor=[0.2],
                 bifpn_channels=48, bigru_hidden_size=256, bigru_layers=1, rescale=0.1):
        super().__init__()
        self.chin = chin
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.normalize = normalize
        self.floor = floor
        self.pfactor = pfactor
        self.npfactor = npfactor
        self.bifpn_channels = bifpn_channels
        self.bigru_hidden_size = bigru_hidden_size
        self.bigru_layers = bigru_layers
        self.rescale = rescale

        self.p_conv1 = nn.ModuleList()
        self.np_conv1 = nn.ModuleList()
        self.p_conv2 = nn.ModuleList()
        self.np_conv2 = nn.ModuleList()
        self.p_convs = nn.ModuleList()

        self.p1_ch = self.hidden
        self.p2_ch = self.hidden * 2
        self.p3_ch = self.hidden * 4
        self.p4_ch = self.hidden * 8
        self.p5_ch = self.hidden * 16

        pconv1 = P_Conv(self.chin, self.p1_ch, self.kernel_size, self.stride, self.pfactor[0])
        npconv1 = N_Conv(self.chin, self.p1_ch, self.kernel_size, self.stride, self.npfactor[0])
        self.p_conv1.append(nn.Sequential(*pconv1))
        self.np_conv1.append(nn.Sequential(*npconv1))

        pconv2 = P_Conv(self.p1_ch, self.p2_ch, self.kernel_size, self.stride, self.pfactor[1])
        npconv2 = N_Conv(self.p1_ch, self.p2_ch, self.kernel_size, self.stride, self.npfactor[0])
        self.p_conv2.append(nn.Sequential(*pconv2))
        self.np_conv2.append(nn.Sequential(*npconv2))

        pconv3 = P_Conv(self.p2_ch, self.p3_ch, self.kernel_size*2, self.stride, self.pfactor[2])
        pconv4 = P_Conv(self.p3_ch, self.p4_ch, self.kernel_size*2, self.stride, self.pfactor[3])
        pconv5 = P_Conv(self.p4_ch, self.p5_ch, self.kernel_size*3, self.stride, self.pfactor[4])
        self.p_convs.append(nn.Sequential(*pconv3))
        self.p_convs.append(nn.Sequential(*pconv4))
        self.p_convs.append(nn.Sequential(*pconv5))

        self.lstm = LSTM(self.p5_ch, bi=False)

        if self.rescale:
            rescale_module(self, reference=self.rescale)

        self.light_bifpn = Light_BiFPN(
            in_channels_p1=self.p1_ch,
            in_channels_p2=self.p2_ch,
            in_channels_p3=self.p3_ch,
            in_channels_p4=self.p4_ch,
            in_channels_p5=self.p5_ch,
            num_channels=bifpn_channels
        )

        self.bigru_input_dim = bifpn_channels
        self.bigru = BiGRU(input_features=self.bigru_input_dim, hidden_features=self.bigru_hidden_size, num_layers=self.bigru_layers)
        self.ln = nn.Linear(in_features=self.bigru_hidden_size, out_features=N_CLASS)

    def valid_length(self, length):
        length = math.ceil(length * 4)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / 4))
        return int(length)

    def forward(self, wav):
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        if self.normalize:
            mono = wav.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            wav = wav / (self.floor + std)
        else:
            std = 1
            
        length = wav.shape[-1]
        x = wav
        x = F.pad(x, (0, self.valid_length(length) - length))
        
        x = upsample2(x)
        x = upsample2(x)

        multi_feat = []
        
        x = PNP_Conv_operation(x, self.p_conv1, self.np_conv1)
        multi_feat.append(x)
        
        x = PNP_Conv_operation(x, self.p_conv2, self.np_conv2)
        multi_feat.append(x)
        
        for p in self.p_convs:
            x = p(x)
            if len(multi_feat) < 5:
                multi_feat.append(x)
            
        x_lstm = x.permute(2,0,1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.permute(1,2,0)

        p1, p2, p3, p4, p5 = multi_feat

        p1 = p1[:,:,:,None]
        p2 = p2[:,:,:,None]
        p3 = p3[:,:,:,None]
        p4 = p4[:,:,:,None]
        p5 = p5[:,:,:,None]

        features = (p1, p2, p3, p4, p5)
        f0_features = self.light_bifpn(features)

        B, C_final, T_final, F_final = f0_features.shape
        f0_feature = f0_features.permute(0, 2, 1, 3).contiguous().view(B, T_final, C_final * F_final)

        bigru_out = self.bigru(f0_feature)

        f0out = self.ln(bigru_out)
        f0out = torch.sigmoid(f0out)

        return f0out


class Light_BiFPN(nn.Module):
    def __init__(self, in_channels_p1, in_channels_p2, in_channels_p3, in_channels_p4, in_channels_p5,
                 num_channels, epsilon=1e-8, orig_swish=False):
        super(Light_BiFPN, self).__init__()
        self.epsilon = epsilon
        self.num_channels = num_channels

        self.p1_upch = SeparableConvBlock(in_channels_p1, num_channels)
        self.p2_upch = SeparableConvBlock(in_channels_p2, num_channels)
        self.p3_upch = SeparableConvBlock(in_channels_p3, num_channels)
        self.p4_dnch = SeparableConvBlock(in_channels_p4, num_channels)
        self.p5_dnch = SeparableConvBlock(in_channels_p5, num_channels)

        self.conv6_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv5_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv4_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv3_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv4_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv5_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv6_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv7_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)

        self.swish = MemoryEfficientSwish() if not orig_swish else Swish()

        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()
        self.p1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_w1_relu = nn.ReLU()
        self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w2_relu = nn.ReLU()
        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

    def pre_resize(self, inputs):
        p1, p2, p3, p4, p5 = inputs
        B, C1, T1, _ = p1.shape
        B, C2, T2, _ = p2.shape
        B, C3, T3, _ = p3.shape
        B, C4, T4, _ = p4.shape
        B, C5, T5, _ = p5.shape

        T_target = T3 // 2

        kernel_p1 = max(T1 // T_target, 1)
        stride_p1 = kernel_p1
        p1_resized = F.avg_pool2d(p1, kernel_size=(kernel_p1, 1), stride=(stride_p1, 1))
        current_T1 = p1_resized.shape[2]
        if current_T1 < T_target:
            pad_needed = T_target - current_T1
            p1_resized = F.pad(p1_resized, (0, 0, 0, pad_needed))
        elif current_T1 > T_target:
             p1_resized = p1_resized[:, :, :T_target, :]

        kernel_p2 = max(T2 // T_target, 1)
        stride_p2 = kernel_p2
        p2_resized = F.avg_pool2d(p2, kernel_size=(kernel_p2, 1), stride=(stride_p2, 1))
        current_T2 = p2_resized.shape[2]
        if current_T2 < T_target:
            pad_needed = T_target - current_T2
            p2_resized = F.pad(p2_resized, (0, 0, 0, pad_needed))
        elif current_T2 > T_target:
             p2_resized = p2_resized[:, :, :T_target, :]

        kernel_p3 = max(T3 // T_target, 1)
        stride_p3 = kernel_p3
        p3_resized = F.avg_pool2d(p3, kernel_size=(kernel_p3, 1), stride=(stride_p3, 1))
        current_T3 = p3_resized.shape[2]
        if current_T3 < T_target:
            pad_needed = T_target - current_T3
            p3_resized = F.pad(p3_resized, (0, 0, 0, pad_needed))
        elif current_T3 > T_target:
             p3_resized = p3_resized[:, :, :T_target, :]

        scale_factor_p4 = T_target / T4
        p4_resized = F.interpolate(p4, size=(T_target, 1), mode='bilinear', align_corners=False)

        scale_factor_p5 = T_target / T5
        p5_resized = F.interpolate(p5, size=(T_target, 1), mode='bilinear', align_corners=False)

        p1_out = self.p1_upch(p1_resized)
        p2_out = self.p2_upch(p2_resized)
        p3_out = self.p3_upch(p3_resized)
        p4_out = self.p4_dnch(p4_resized)
        p5_out = self.p5_dnch(p5_resized)

        return p1_out, p2_out, p3_out, p4_out, p5_out

    def forward(self, inputs):
        p1_in, p2_in, p3_in, p4_in, p5_in = self.pre_resize(inputs)

        p4_up = F.interpolate(p5_in, size=p4_in.shape[2:], mode='bilinear', align_corners=False)
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        w = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_out = self.conv6_up(self.swish(w[0] * p4_in + w[1] * p4_up))

        p3_up = F.interpolate(p4_out, size=p3_in.shape[2:], mode='bilinear', align_corners=False)
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        w = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv5_up(self.swish(w[0] * p3_in + w[1] * p3_up))

        p2_up = F.interpolate(p3_out, size=p2_in.shape[2:], mode='bilinear', align_corners=False)
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        w = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_out = self.conv4_up(self.swish(w[0] * p2_in + w[1] * p2_up))

        p1_up = F.interpolate(p2_out, size=p1_in.shape[2:], mode='bilinear', align_corners=False)
        p1_w1 = self.p1_w1_relu(self.p1_w1)
        w = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)
        p1_out = self.conv3_up(self.swish(w[0] * p1_in + w[1] * p1_up))

        p1_dn = F.avg_pool2d(p1_out, kernel_size=(2,1), stride=(2,1))
        p2_w2 = self.p2_w2_relu(self.p2_w2)
        w = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        p2_out = self.conv4_down(self.swish(w[0] * p2_in + w[1] * p2_out + w[2] * p1_dn))

        p2_dn = F.avg_pool2d(p2_out, kernel_size=(2,1), stride=(2,1))
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        w = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.conv5_down(self.swish(w[0] * p3_in + w[1] * p3_out + w[2] * p2_dn))

        p3_dn = F.avg_pool2d(p3_out, kernel_size=(2,1), stride=(2,1))
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        w = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv6_down(self.swish(w[0] * p4_in + w[1] * p4_out + w[2] * p3_dn))

        p4_dn = F.avg_pool2d(p4_out, kernel_size=(2,1), stride=(2,1))
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        w = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv7_down(self.swish(w[0] * p5_in + w[1] * p4_dn))

        return p1_out