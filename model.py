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
    """
    The real keras/tensorflow conv2d with same padding
    """
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

# --- BiGRU Module from RMVPE (and DJCM for 3D sequences) ---
class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        """
        input_features: Input dimension for each time step (e.g., C * F after flattening)
        hidden_features: Hidden state dimension of the GRU
        num_layers: Number of layers in the GRU
        """
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)
        # The output of the bidirectional GRU is 2 * hidden_features

    def forward(self, x):
        """
        x: Input tensor of shape [B, T, input_features]
        """
        # x: [B, T, input_features]
        output, _ = self.gru(x) # output: [B, T, 2 * hidden_features]
        return output # Returns the complete sequence


# --- Improved Hybrid Architecture (Faithful to MF-PAM in pre_resize) ---
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

        # --- Analysis Stage (MF-PAM) ---
        self.p_conv1 = nn.ModuleList()
        self.np_conv1 = nn.ModuleList()
        self.p_conv2 = nn.ModuleList()
        self.np_conv2 = nn.ModuleList()
        self.p_convs = nn.ModuleList()

        # Channels for pre_resize in BiFPN
        self.p1_ch = self.hidden # 6
        self.p2_ch = self.hidden * 2 # 12
        self.p3_ch = self.hidden * 4 # 24
        self.p4_ch = self.hidden * 8 # 48
        self.p5_ch = self.hidden * 16 # 96

        # PNP-Conv1
        pconv1 = P_Conv(self.chin, self.p1_ch, self.kernel_size, self.stride, self.pfactor[0])
        npconv1 = N_Conv(self.chin, self.p1_ch, self.kernel_size, self.stride, self.npfactor[0])
        self.p_conv1.append(nn.Sequential(*pconv1))
        self.np_conv1.append(nn.Sequential(*npconv1))

        # PNP-Conv2
        pconv2 = P_Conv(self.p1_ch, self.p2_ch, self.kernel_size, self.stride, self.pfactor[1])
        npconv2 = N_Conv(self.p1_ch, self.p2_ch, self.kernel_size, self.stride, self.npfactor[0])
        self.p_conv2.append(nn.Sequential(*pconv2))
        self.np_conv2.append(nn.Sequential(*npconv2))

        # P-Conv3, 4, 5
        pconv3 = P_Conv(self.p2_ch, self.p3_ch, self.kernel_size*2, self.stride, self.pfactor[2])
        pconv4 = P_Conv(self.p3_ch, self.p4_ch, self.kernel_size*2, self.stride, self.pfactor[3])
        pconv5 = P_Conv(self.p4_ch, self.p5_ch, self.kernel_size*3, self.stride, self.pfactor[4])
        self.p_convs.append(nn.Sequential(*pconv3))
        self.p_convs.append(nn.Sequential(*pconv4))
        self.p_convs.append(nn.Sequential(*pconv5))

        # Optional LSTM at the end of the analysis stage
        self.lstm = LSTM(self.p5_ch, bi=False) # self.p5_ch = hidden*16

        if self.rescale:
            rescale_module(self, reference=self.rescale)

        # --- Light BiFPN (MF-PAM) ---
        # Pass input channels dynamically
        # We adjust the channels for pre_resize based on the hidden ratio
        # Original: num_channels//4, num_channels//2, num_channels*2, num_channels*2
        # With hidden=6, bifpn_channels=48 -> p1_ch=6, p2_ch=12, p3_ch=24, p4_ch=48, p5_ch=96
        # Original used num_channels=48 -> num_channels//4=12, num_channels//2=24, num_channels*2=96
        # Our BiFPN expects p1_ch=6, p2_ch=12, p3_ch=24, p4_ch=48, p5_ch=96 for channel adjustment blocks
        # But Light_BiFPN original uses fixed ratios for pooling/upsampling kernels.
        # We will use the original logic of fixed kernels, assuming the analysis stage produces
        # specific temporal sizes (as in the original code comments).
        # The original code assumes p1, p2, p3, p4, p5 have specific temporal sizes
        # (approximately [16020, 4004, 1000, 249, 249] for 4 seconds and certain parameters).
        # The valid_length calculation and strided convolutions in the Analysis_stage
        # are designed so that temporal sizes are consistent with pre_resize.
        # Therefore, pre_resize can use fixed kernels if the Analysis_stage is consistent.
        # We will use the original kernels from MF-PAM code, adapting input channels.
        # Fixed kernels (32,1), (8,1), (2,1) and factors (2,1) assume a specific T for p1, p2, p3, p4, p5.
        # If our Analysis_stage produces different temporal sizes, this pre_resize will fail.
        # The key is that valid_length and Analysis_stage operations must produce
        # temporal sizes consistent with the fixed kernels of pre_resize.
        # The original MF-PAM code has valid_length that considers upsample x4 and downsampling x4,
        # which is consistent with the code provided here.
        # The original pre_resize assumes that the final T after the analysis stage
        # and lstm (if applied) is such that fixed kernels align to ~500 frames.
        # MF-PAM's valid_length logic ensures that the input temporal length
        # is such that after convolutions (stride=4) and upsamples, it aligns correctly.
        # Therefore, if our Analysis_stage is identical (except for channels), the temporal sizes
        # of p1-p5 should be similar to those in the original code for the same input length.
        # For example, for 4 seconds (64000 samples at 16kHz):
        # - valid_length(64000) -> 64000 (assuming it's already valid)
        # - upsample x4 -> 256000
        # - Conv1 (stride=4) -> ~64000
        # - Conv2 (stride=4) -> ~16000
        # - Conv3 (stride=4) -> ~4000
        # - Conv4 (stride=4) -> ~1000
        # - Conv5 (stride=4) -> ~250
        # - LSTM -> ~250 (maintains temporal)
        # Original sizes in comments were [16020, 4004, 1000, 249, 249].
        # Ours will be [T1, T2, T3, T4, T5] where T5 ~ T4 ~ 250, T3 ~ 1000, T2 ~ 4000, T1 ~ 16000.
        # Original pre_resize:
        # - p1: avg_pool2d((32,1)) -> T1//32 ~ 16000//32 = 500
        # - p2: avg_pool2d((8,1)) -> T2//8 ~ 4000//8 = 500
        # - p3: avg_pool2d((2,1)) -> T3//2 ~ 1000//2 = 500
        # - p4: upsample(2,1) -> T4 * 2 ~ 250 * 2 = 500
        # - p5: upsample(2,1) -> T5 * 2 ~ 250 * 2 = 500
        # These fixed kernels (32, 8, 2, 2) are correct if the T outputs from Analysis_stage
        # are proportional to those in the original code.
        # Our Analysis_stage uses the same strides (all 4) and upsamples (x2 x2 = x4).
        # The difference is kernel_size (4,4,8,8,12 in the paper, 4,4,8,8,12 in original code -> 4,4,8,8,12 here).
        # Kernel_sizes affect the final T SUBTRACTIVELY, not multiplicatively like stride.
        # length_after_conv = floor((length_before_conv - kernel_size) / stride) + 1
        # But valid_length compensates so the final T is coherent.
        # Therefore, assuming valid_length works correctly for our Analysis_stage,
        # the temporal sizes of p1-p5 should be adequate for the fixed kernels of pre_resize.
        # We will use the modified Light_BiFPN that accepts dynamic input channels,
        # but maintain the fixed kernel logic in pre_resize, as in the original MF-PAM,
        # adapting kernels to match the Analysis_stage logic.
        # The original code used fixed kernels based on the downsampling progression.
        # Stride=4 in each block of the Analysis_stage (conv1, conv2, conv3, conv4, conv5).
        # T_initial (after upsample x4) = T_0
        # T_p1 = T_0 // 4 (after conv1)
        # T_p2 = T_p1 // 4 = T_0 // 16 (after conv2)
        # T_p3 = T_p2 // 4 = T_0 // 64 (after conv3)
        # T_p4 = T_p3 // 4 = T_0 // 256 (after conv4)
        # T_p5 = T_p4 // 4 = T_0 // 1024 (after conv5)
        # The goal of pre_resize is to align all to T_p3 (approximately).
        # Then:
        # - p1: T_0 // 4 -> want T_0 // 64. Kernel = (T_0//4) / (T_0//64) = 16. But original used 32.
        # - p2: T_0 // 16 -> want T_0 // 64. Kernel = (T_0//16) / (T_0//64) = 4. But original used 8.
        # - p3: T_0 // 64 -> want T_0 // 64. Kernel = 1. But original used 2. (Perhaps to halve to 500).
        # - p4: T_0 // 256 -> want T_0 // 64. Factor = (T_0//64) / (T_0//256) = 4. Original used 2. (Perhaps to double to 500).
        # - p5: T_0 // 1024 -> want T_0 // 64. Factor = (T_0//64) / (T_0//1024) = 16. Original used 2. (Perhaps to double to 500).
        # The logic of fixed kernels in the original code doesn't seem to follow exactly
        # the downsampling ratio, but is probably empirically adjusted
        # to align to a specific target size (like ~500 frames for 4 seconds).
        # The comment says "500 = frame number while the input_length is 4sec & hop_length is 128 & sampling rate is 16 kHz".
        # 4 sec * 16000 Hz = 64000 samples.
        # With hop_length=128 (assuming STFT-like), frames = 64000 / 128 = 500.
        # So, the fixed kernels in the original code are *designed* to align
        # the Analysis_stage outputs to this target frame number (500).
        # The `valid_length` function ensures that the input length produces outputs
        # from the Analysis_stage with temporal sizes that align correctly
        # with these fixed kernels of `pre_resize` to result in ~500 frames.
        # Therefore, in our hybrid, we must use fixed kernels in `pre_resize`
        # that are consistent with our `Analysis_stage` and the `valid_length` used,
        # to align to ~500 frames (or the target temporal size the BiFPN expects).
        # The target size is approximately T_p3 // 2 (the T of p3 divided by 2).
        # In the original Analysis_stage, T_p3 ~ 1000 (for 4 sec), target = 500.
        # Our Analysis_stage (stride=4 in all) should have the same ratio.
        # T_p3 = T_initial // (4**3) = T_initial // 64.
        # Target = T_p3 // 2 = T_initial // 128.
        # Fixed kernels to align:
        # - p1: T_in // 4 -> T_in // 128. Kernel = (T_in//4) / (T_in//128) = 32. (Same as original)
        # - p2: T_in // 16 -> T_in // 128. Kernel = (T_in//16) / (T_in//128) = 8. (Same as original)
        # - p3: T_in // 64 -> T_in // 128. Kernel = (T_in//64) / (T_in//128) = 2. (Same as original)
        # - p4: T_in // 256 -> T_in // 128. Factor = (T_in//128) / (T_in//256) = 2. (Same as original)
        # - p5: T_in // 1024 -> T_in // 128. Factor = (T_in//128) / (T_in//1024) = 8. (NOT same as original, original used 2)
        # The original code used (2,1) for p5. This indicates the kernel for p5
        # doesn't follow the exact downsampling ratio to align with p3//2,
        # but was probably adjusted to align with p4 or the final target.
        # To be faithful to the *original code*, we will use the fixed kernels from the original code:
        # p1: (32,1), p2: (8,1), p3: (2,1), p4: (2,1), p5: (2,1) (with upsample).
        # But we must adapt the input channels for the SeparableConvBlock adjustment blocks.
        # Original: num_channels//4, num_channels//2, num_channels*2, num_channels*2
        # Our case: num_channels=bifpn_channels, in_channels_p1=hidden, in_channels_p2=hidden*2, etc.
        # The channel adjustment block in pre_resize must accept in_channels_pX and produce num_channels.
        self.light_bifpn = Light_BiFPN(
            in_channels_p1=self.p1_ch,
            in_channels_p2=self.p2_ch,
            in_channels_p3=self.p3_ch,
            in_channels_p4=self.p4_ch,
            in_channels_p5=self.p5_ch,
            num_channels=bifpn_channels
        )

        # --- Final Projection ---
        # BiFPN output is [B, bifpn_channels, T_final, F_final] (approximately)
        # It's reformatted to [B, T_final, bifpn_channels * F_final] for BiGRU
        # Assuming F_final ~ 1, bigru_input_dim = bifpn_channels.
        self.bigru_input_dim = bifpn_channels
        self.bigru = BiGRU(input_features=self.bigru_input_dim, hidden_features=self.bigru_hidden_size, num_layers=self.bigru_layers)
        self.ln = nn.Linear(in_features=self.bigru_hidden_size, out_features=N_CLASS) # N_CLASS = 360

    def valid_length(self, length):
        length = math.ceil(length * 4) # Upsample x2 * x2 = x4
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
        
        # Upsample x2 x2
        x = upsample2(x)
        x = upsample2(x)

        multi_feat = []
        
        x = PNP_Conv_operation(x, self.p_conv1, self.np_conv1)
        multi_feat.append(x) # p1 (after PNP-Conv1)
        
        x = PNP_Conv_operation(x, self.p_conv2, self.np_conv2)
        multi_feat.append(x) # p2 (after PNP-Conv2)
        
        for p in self.p_convs:
            x = p(x)
            if len(multi_feat) < 5: # p3, p4, p5
                multi_feat.append(x)
            
        # Apply LSTM if present (after p5)
        x_lstm = x.permute(2,0,1) # [T, B, C]
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.permute(1,2,0) # [B, C, T]

        # --- Estimation Stage ---
        p1, p2, p3, p4, p5 = multi_feat # [B, C, T]

        # Add dimension for BiFPN
        p1 = p1[:,:,:,None] # [B, 6, T, 1]
        p2 = p2[:,:,:,None] # [B, 12, T, 1]
        p3 = p3[:,:,:,None] # [B, 24, T, 1]
        p4 = p4[:,:,:,None] # [B, 48, T, 1]
        p5 = p5[:,:,:,None] # [B, 96, T, 1]

        features = (p1, p2, p3, p4, p5)
        # Important: BiFPN output must be a single 4D feature
        f0_features = self.light_bifpn(features) # [B, bifpn_channels, T_final, F_final]

        # --- BiGRU Integration inspired by RMVPE ---
        # Assuming f0_features.shape = [B, C_final, T_final, F_final]
        # Flatten spatial dimensions (C, F) for each temporal frame T
        # Shape: [B, C_final, T_final, F_final] -> [B, T_final, C_final * F_final]
        # If F_final = 1, then C_final * F_final = C_final
        # Shape: [B, bifpn_channels, T, 1] -> [B, T, bifpn_channels]
        B, C_final, T_final, F_final = f0_features.shape
        f0_feature = f0_features.permute(0, 2, 1, 3).contiguous().view(B, T_final, C_final * F_final) # [B, T, C*F]

        # --- Sequential Temporal Modeling (BiGRU from DJCM/RMVPE) ---
        bigru_out = self.bigru(f0_feature) # [B, T, bigru_hidden_size]

        # --- Final Projection ---
        f0out = self.ln(bigru_out) # [B, T, N_CLASS (360)]
        f0out = torch.sigmoid(f0out) # [B, T, 360]

        return f0out


class Light_BiFPN(nn.Module):
    """
    Adapted from MF-PAM implementation, with dynamic input channels for pre_resize,
    but using fixed pooling/upsampling kernels from the original code for alignment.
    """
    def __init__(self, in_channels_p1, in_channels_p2, in_channels_p3, in_channels_p4, in_channels_p5,
                 num_channels, epsilon=1e-8, orig_swish=False):
        super(Light_BiFPN, self).__init__()
        self.epsilon = epsilon
        self.num_channels = num_channels

        # Pre resize: Adjust channels dynamically, but use fixed kernels for temporal alignment
        # Based on original MF-PAM code, which used fixed kernels (32,1), (8,1), (2,1), etc.
        # to align to ~500 frames. We adapt input channels.
        # Original: num_channels//4 -> num_channels, etc.
        # Our case: in_channels_pX -> num_channels
        self.p1_upch = SeparableConvBlock(in_channels_p1, num_channels)
        self.p2_upch = SeparableConvBlock(in_channels_p2, num_channels)
        self.p3_upch = SeparableConvBlock(in_channels_p3, num_channels)
        # Assuming in_channels_p4 and in_channels_p5 are >= num_channels, use dnch (downchannel)
        # as in the original code.
        self.p4_dnch = SeparableConvBlock(in_channels_p4, num_channels)
        self.p5_dnch = SeparableConvBlock(in_channels_p5, num_channels)

        # BiFPN conv layers
        self.conv6_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv5_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv4_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv3_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv4_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv5_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv6_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv7_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)

        self.swish = MemoryEfficientSwish() if not orig_swish else Swish()

        # Weights for fusion
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
        # Assuming inputs = (p1, p2, p3, p4, p5)
        # Where p1: [B, p1_ch, T1, 1], p2: [B, p2_ch, T2, 1], ..., p5: [B, p5_ch, T5, 1]
        # The goal is to temporally align to ~500 frames (or T_p3 // 2) using fixed kernels.
        # Based on original MF-PAM code.
        p1, p2, p3, p4, p5 = inputs

        # Pre resizing: Use fixed kernels as in original MF-PAM code
        # Assuming T1 ~ 16000, T2 ~ 4000, T3 ~ 1000, T4 ~ 250, T5 ~ 250 for 4 sec.
        # Target ~ 500.
        # p1: [B, p1_ch, T1, 1] -> avg_pool2d((32,1)) -> [B, p1_ch, T1//32, 1] -> ~[B, p1_ch, 500, 1]
        p1_resized = self.p1_upch(F.avg_pool2d(p1, (32,1)))
        # p2: [B, p2_ch, T2, 1] -> avg_pool2d((8,1)) -> [B, p2_ch, T2//8, 1] -> ~[B, p2_ch, 500, 1]
        # Original also did F.pad(..., (0,0,1,0)) -> adds 1 at the temporal start. Why?
        # Perhaps to compensate for avg_pool offset or for specific alignment. We replicate it.
        p2_resized = F.pad(self.p2_upch(F.avg_pool2d(p2, (8,1))), (0,0,1,0))
        # p3: [B, p3_ch, T3, 1] -> avg_pool2d((2,1)) -> [B, p3_ch, T3//2, 1] -> ~[B, p3_ch, 500, 1]
        # Original also did F.pad(..., (0,0,1,0)).
        p3_resized = F.pad(self.p3_upch(F.avg_pool2d(p3, (2,1))), (0,0,1,0))
        # p4: [B, p4_ch, T4, 1] -> upsample(2,1) -> [B, p4_ch, T4*2, 1] -> ~[B, p4_ch, 500, 1]
        # Original used nn.Upsample(scale_factor=(2,1), mode='nearest')
        # and did F.pad(p4, (0,0,2,1)) BEFORE the upsample. Why?
        # Nearest upsample may interpolate differently if T is not exactly divisible.
        # The pad before may be for internal alignment of the upsample or to compensate for offset.
        # We replicate it: pad -> upsample -> dnch.
        p4_upsampled = self.p4_dnch(nn.Upsample(scale_factor=(2,1), mode='nearest')(F.pad(p4,(0,0,2,1))))
        # p5: [B, p5_ch, T5, 1] -> upsample(2,1) -> [B, p5_ch, T5*2, 1] -> ~[B, p5_ch, 500, 1]
        # Same as p4: pad -> upsample -> dnch.
        p5_upsampled = self.p5_dnch(nn.Upsample(scale_factor=(2,1), mode='nearest')(F.pad(p5,(0,0,2,1))))

        return p1_resized, p2_resized, p3_resized, p4_upsampled, p5_upsampled


    def forward(self, inputs):
        p1_in, p2_in, p3_in, p4_in, p5_in = self.pre_resize(inputs)
        # [B, num_channels, ~500, 1] each (approximately)

        # Top-down path
        # Upsample p5 to match p4
        p4_up = F.interpolate(p5_in, size=p4_in.shape[2:], mode='bilinear', align_corners=False)
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        w = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_out = self.conv6_up(self.swish(w[0] * p4_in + w[1] * p4_up))

        # Upsample p4_out to match p3
        p3_up = F.interpolate(p4_out, size=p3_in.shape[2:], mode='bilinear', align_corners=False)
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        w = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv5_up(self.swish(w[0] * p3_in + w[1] * p3_up))

        # Upsample p3_out to match p2
        p2_up = F.interpolate(p3_out, size=p2_in.shape[2:], mode='bilinear', align_corners=False)
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        w = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_out = self.conv4_up(self.swish(w[0] * p2_in + w[1] * p2_up))

        # Upsample p2_out to match p1
        p1_up = F.interpolate(p2_out, size=p1_in.shape[2:], mode='bilinear', align_corners=False)
        p1_w1 = self.p1_w1_relu(self.p1_w1)
        w = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)
        p1_out = self.conv3_up(self.swish(w[0] * p1_in + w[1] * p1_up))

        # Bottom-up path
        # Downsample p1_out to match p2
        p1_dn = F.avg_pool2d(p1_out, kernel_size=(2,1), stride=(2,1))
        p2_w2 = self.p2_w2_relu(self.p2_w2)
        w = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        p2_out = self.conv4_down(self.swish(w[0] * p2_in + w[1] * p2_out + w[2] * p1_dn))

        # Downsample p2_out to match p3
        p2_dn = F.avg_pool2d(p2_out, kernel_size=(2,1), stride=(2,1))
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        w = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.conv5_down(self.swish(w[0] * p3_in + w[1] * p3_out + w[2] * p2_dn))

        # Downsample p3_out to match p4
        p3_dn = F.avg_pool2d(p3_out, kernel_size=(2,1), stride=(2,1))
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        w = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv6_down(self.swish(w[0] * p4_in + w[1] * p4_out + w[2] * p3_dn))

        # Downsample p4_out to match p5
        p4_dn = F.avg_pool2d(p4_out, kernel_size=(2,1), stride=(2,1))
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        w = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv7_down(self.swish(w[0] * p5_in + w[1] * p4_dn))

        # Return only the final output from the up/down fusion process: p1_out
        # This has shape [B, num_channels, T_target, 1]
        return p1_out # [B, bifpn_channels, T, 1]


# --- Usage Example ---
if __name__ == "__main__":
    model = HybridPitchEstimator(
        chin=1,
        hidden=6, # Adjust as needed
        kernel_size=4,
        stride=4,
        depth=5,
        normalize=True,
        floor=1e-3,
        pfactor=[17, 13, 11, 7, 5],
        npfactor=[0.2],
        bifpn_channels=48,
        bigru_hidden_size=256,
        bigru_layers=1,
        rescale=0.1
    )

    # Example input: batch_size=2, length=16000 (1 second at 16kHz)
    # To properly test pre_resize, use a longer length (e.g., 4 seconds = 64000)
    input_wav = torch.randn(2, 64000) # [B, T_samples]

    try:
        # Forward pass
        output = model(input_wav)
        print(f"Input shape: {input_wav.shape}")
        print(f"Output shape: {output.shape}") # Should be [B, T_frames, 360]
        print("Forward pass successful!")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()