import os
import time
import argparse
import json
import librosa
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from env import AttrDict, build_env
from dataset import F0Dataset
from model import Estimation_stage
from utils import scan_checkpoint, load_checkpoint, save_checkpoint, plot_f0_compared
import torch.nn as nn
import pkbar
import mir_eval

torch.backends.cudnn.benchmark = True


def onehot_to_hz(onehot, bins_per_octave=48, threshold=0.6):
    """
    Threshold could effect the performance.
    Setting the threshold to 0 will give you the best performance when evaluating RPA and RCA in the voicing region
    """
    # input: [b x T x freq_bins]
    # output: [b x T]
    fmin = 32.7
    max_onehot = torch.max(onehot, dim=2)
    indexs = max_onehot[1]
    mask = (max_onehot[0] > threshold).float()

    hz = fmin * (2 ** (indexs / bins_per_octave))
    # Set freq to 0 if activate val below threshold
    hz = hz * mask

    return hz, max_onehot[0]


def train(rank, a, h):

    torch.cuda.manual_seed(h.seed)
    device = torch.device("cuda:{:d}".format(rank))

    estimator = Estimation_stage().to(device)

    if rank == 0:
        print(estimator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, "g_")

    steps = 0
    state_dict_g = None
    if cp_g is None:
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        estimator.load_state_dict(state_dict_g["estimator"])
        steps = state_dict_g["steps"] + 1
        last_epoch = state_dict_g["epoch"]

    optim_g = torch.optim.Adam(
        estimator.parameters(), lr=h.learning_rate, betas=(h.adam_b1, h.adam_b2)
    )

    if state_dict_g is not None:
        optim_g.load_state_dict(state_dict_g["optim_g"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )

    trainset = F0Dataset(
        a.data_path,
        h.hop_size,
        h.sampling_rate,
        group="train",
        shuffle=True,
        train=True,
        rir_dir=a.rir_dir,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=True,
        sampler=None,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        validset = F0Dataset(
            a.data_path,
            h.hop_size,
            h.sampling_rate,
            group="test",
            split=False,
            shuffle=False,
            train=False,
            rir_dir=a.rir_dir,
        )

        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
        )

        sw = SummaryWriter(os.path.join(a.checkpoint_path, "logs"))

    estimator.train()
    criterion = nn.BCELoss()
    criterion2 = nn.L1Loss()

    #################################### Training ####################################
    for epoch in range(max(0, last_epoch + 1), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        for ii, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            cleanf0, cleanf0_quant, cleanaudio, noisyaudio, filename = batch
            # cleanf0.size() = [B, T/hop_len]
            # cleanf0_quant.size() = [B, T/hop_len, 360]
            # cleanaudio.size() = noisyaudio.size() = [B, T]

            cleanf0 = torch.autograd.Variable(cleanf0.to(device, non_blocking=True))
            cleanf0_quant = torch.autograd.Variable(
                cleanf0_quant.to(device, non_blocking=True)
            )
            noisyaudio = torch.autograd.Variable(
                noisyaudio.to(device, non_blocking=True)
            )

            onehot_hat = estimator(noisyaudio)

            optim_g.zero_grad()

            loss_f0_bin = criterion(onehot_hat, cleanf0_quant.float())
            loss_gen_all = loss_f0_bin
            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    print(
                        "Steps : {:d}, Gen Loss Total : {:4.3f}, F0-bin-loss : {:4.3f}, s/b : {:4.3f}".format(
                            steps, loss_gen_all, loss_f0_bin, time.time() - start_b
                        )
                    )

                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "estimator": estimator.state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_f0_bin_loss", loss_f0_bin, steps)

                #################################### Validation ####################################
                if steps % a.validation_interval == 0:  # and steps != 0:
                    estimator.eval()
                    torch.cuda.empty_cache()
                    f0bin_error = 0
                    f0_mae = 0
                    RPA = 0
                    RCA = 0
                    OA = 0
                    VR = 0
                    VFA = 0
                    with torch.no_grad():
                        pcount = 0
                        pbar = pkbar.Pbar("validation", len(validation_loader))
                        for i, batch in enumerate(validation_loader):
                            pbar.update(pcount)
                            cleanf0, cleanf0_quant, cleanaudio, noisyaudio, filename = (
                                batch
                            )

                            onehot_hat = estimator(noisyaudio.to(device))

                            filename = filename[0].split(".")[0]
                            if cleanf0.size(1) != onehot_hat.size(1):
                                minlen = min(cleanf0.size(1), onehot_hat.size(1))
                                onehot_hat = onehot_hat[:, :minlen]
                                cleanf0 = cleanf0[:, :minlen]
                                cleanf0_quant = cleanf0_quant[:, :minlen]

                            cleanf0 = cleanf0.squeeze(0)
                            cleanf0_quant = cleanf0_quant.squeeze(0)
                            onehot_hat = onehot_hat.squeeze(0)

                            f0bin_error += criterion(
                                onehot_hat.to(device), cleanf0_quant.to(device).float()
                            ).item()

                            # Set threshold to 0 to evaluate the pure estimation performance on voicing region
                            f0_hat, max_onehot = onehot_to_hz(
                                onehot_hat.unsqueeze(0), 48, threshold=0.0
                            )
                            f0_hat = f0_hat.squeeze().to(torch.float64).to(device)
                            v_cleanf0_eval = cleanf0.to(torch.float64).to(device)
                            f0_hat_vad = torch.where(v_cleanf0_eval == 0, 0.0, f0_hat)

                            # MAE
                            f0_mae += criterion2(v_cleanf0_eval, f0_hat_vad).item()

                            # Metrics Calculation
                            freq_pred = f0_hat.squeeze().cpu().numpy()
                            freq_ref = cleanf0.cpu().numpy()
                            time_slice = (
                                np.arange(len(freq_ref)) * h.hop_size / h.sampling_rate
                            )
                            ref_v, ref_c, est_v, est_c = (
                                mir_eval.melody.to_cent_voicing(
                                    time_slice, freq_ref, time_slice, freq_pred
                                )
                            )

                            RPA += mir_eval.melody.raw_pitch_accuracy(
                                ref_v, ref_c, est_v, est_c
                            )
                            RCA += mir_eval.melody.raw_chroma_accuracy(
                                ref_v, ref_c, est_v, est_c
                            )
                            OA += mir_eval.melody.overall_accuracy(
                                ref_v, ref_c, est_v, est_c
                            )
                            VR += mir_eval.melody.voicing_recall(ref_v, est_v)
                            VFA += mir_eval.melody.voicing_false_alarm(ref_v, est_v)

                            if i < 6:
                                sw.add_figure(
                                    "generatedf0/f0_hat_{}".format(filename),
                                    plot_f0_compared(freq_pred, freq_ref),
                                    steps,
                                )
                                sw.add_figure(
                                    "generatedf0_vad/f0_hat_vad{}".format(filename),
                                    plot_f0_compared(
                                        f0_hat_vad.cpu().numpy(), freq_ref
                                    ),
                                    steps,
                                )

                            pcount += 1

                        val_f0bin_err = f0bin_error / (i + 1)
                        val_f0_mae = f0_mae / (i + 1)
                        val_RPA = RPA / (i + 1)
                        val_RCA = RCA / (i + 1)
                        val_OA = OA / (i + 1)
                        val_VR = VR / (i + 1)
                        val_VFA = VFA / (i + 1)

                        sw.add_scalar("validation/f0_bin_error", val_f0bin_err, steps)
                        sw.add_scalar("validation/f0_mae", val_f0_mae, steps)
                        sw.add_scalar("validation/RPA", val_RPA * 100, steps)
                        sw.add_scalar("validation/RCA", val_RCA * 100, steps)
                        sw.add_scalar("validation/OA", val_OA * 100, steps)
                        sw.add_scalar("validation/VR", val_VR * 100, steps)
                        sw.add_scalar("validation/VFA", val_VFA * 100, steps)

                        print(
                            f"\nSteps: {steps} | RPA: {val_RPA*100:.2f}% | OA: {val_OA*100:.2f}%"
                        )

                    estimator.train()
            steps += 1

        scheduler_g.step()

        if rank == 0:
            print(
                "Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, int(time.time() - start)
                )
            )


def main():
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--rir_dir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training_epochs", default=3100, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=2000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--validation_interval", default=2000, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, "config.json", a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        print("Batch size per GPU :", h.batch_size)

    train(0, a, h)


if __name__ == "__main__":
    main()
