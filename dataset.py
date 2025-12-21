import math
import os
import random
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio
import augment
import scipy.signal as ss
import soundfile as sf
import colorednoise as cn


def hz_to_onehot(hz, freq_bins=360, bins_per_octave=48):
    fmin = 32.7
    # Fix UserWarning for torch.tensor
    if isinstance(hz, torch.Tensor):
        hz = hz.clone().detach()
    else:
        hz = torch.tensor(hz)
    indexs = (
        torch.log((hz + 0.0000001) / fmin) / np.log(2.0 ** (1.0 / bins_per_octave))
        + 0.5
    ).long()
    mask = (indexs >= 0).long()
    mask = torch.unsqueeze(mask, dim=1)
    onehot = F.one_hot(torch.clip(indexs, 0, freq_bins - 1), freq_bins)
    onehot = onehot * mask
    return onehot


class Audioset:
    def __init__(
        self, files=None, length=None, stride=None, pad=True, sample_rate=None
    ):
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.sample_rate = sample_rate
        for file in self.files:
            info = torchaudio.info(file)
            file_length = info.num_frames
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            else:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for file, examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            offset = 0
            num_frames = -1
            if self.length is not None:
                offset = self.stride * index
                num_frames = int(self.length)
            out, sr = torchaudio.load(
                str(file), frame_offset=offset, num_frames=num_frames
            )
            if out.shape[0] > 1:
                out = torch.mean(out, dim=0, keepdim=True)
            if num_frames > 0 and out.shape[-1] < num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            return out, file, offset


class F0Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        hop_size,
        sampling_rate,
        group="train",
        split=True,
        shuffle=True,
        train=True,
        length=4.5 * 16000,
        stride=1 * 16000,
        rir_dir=None,
    ):
        group_path = os.path.join(path, group)
        self.files = [
            os.path.join(group_path, f)
            for f in os.listdir(group_path)
            if f.endswith(".wav")
            and os.path.exists(os.path.join(group_path, f.replace(".wav", ".pv")))
        ]
        if shuffle:
            random.seed(1234)
            random.shuffle(self.files)
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.train = train
        self.rir_dir = rir_dir
        self.length = length if split else None
        self.stride = stride if split else length
        if rir_dir is not None:
            rir_csv = "train.csv" if train else "test.csv"
            self.rir_list = pd.read_csv(os.path.join(rir_dir, rir_csv))
        self.audio_set = Audioset(
            self.files, length=self.length, stride=stride, sample_rate=sampling_rate
        )
        self.augment = torch.nn.Sequential(augment.Shift(8000, True)) if train else None

    def add_reverb(self, wav, rir_list):
        one_rir = np.random.choice(rir_list["path"])
        rir_sig, _ = sf.read(one_rir)
        if max(abs(rir_sig)) != 1:
            rir_sig = rir_sig / max(abs(rir_sig))
        cut = np.argmax(rir_sig)
        rir_sig = rir_sig[cut:]
        wav_np = wav.squeeze().numpy()
        reverb = ss.convolve(rir_sig, wav_np)[: len(wav_np)]
        return torch.FloatTensor(reverb).unsqueeze(0)

    def __getitem__(self, index):
        cleanaudio, audio_path, offset = self.audio_set[index]
        noisyaudio = cleanaudio.clone()
        if self.train:
            if self.rir_dir is not None:
                mode = np.random.choice(
                    ["noise", "rir", "both", "clean"], p=(0.25, 0.25, 0.4, 0.1)
                )
                if mode in ["rir", "both"]:
                    noisyaudio = self.add_reverb(cleanaudio, self.rir_list)
            else:
                mode = np.random.choice(["noise", "clean"], p=(0.9, 0.1))
            if mode in ["noise", "both"]:
                noi = cn.powerlaw_psd_gaussian(
                    random.uniform(0, 2), noisyaudio.shape[-1]
                )
                noi = torch.from_numpy(noi).float() * (10 ** random.uniform(-6, -1))
                noisyaudio = noisyaudio + noi
            sources = torch.stack([noisyaudio.unsqueeze(0), cleanaudio.unsqueeze(0)])
            sources = self.augment(sources)
            noisyaudio, cleanaudio = sources.squeeze(1)
            max_amp = float(torch.max(torch.abs(noisyaudio))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            noisyaudio = noisyaudio * (10**log10_vol_shift)
            cleanaudio = cleanaudio * (10**log10_vol_shift)
        pv_path = str(audio_path).replace(".wav", ".pv")
        f0_hz_full = np.loadtxt(pv_path)
        start_frame = int(round(offset / self.hop_size))
        num_f0_frames = int(cleanaudio.shape[-1] / self.hop_size)
        f0_hz = f0_hz_full[start_frame : start_frame + num_f0_frames]
        if len(f0_hz) < num_f0_frames:
            f0_hz = np.pad(f0_hz, (0, num_f0_frames - len(f0_hz)), mode="constant")
        else:
            f0_hz = f0_hz[:num_f0_frames]
        cleanf0 = torch.from_numpy(f0_hz)
        cleanf0_quant = hz_to_onehot(cleanf0)
        return (
            cleanf0.squeeze(),
            cleanf0_quant.squeeze(),
            cleanaudio.squeeze(0),
            noisyaudio.squeeze(0),
            os.path.basename(str(audio_path)),
        )

    def __len__(self):
        return len(self.audio_set)
