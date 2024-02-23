import os
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import torch

from neural_lam.weather_dataset import AnalysisDataset
from neural_lam import constants


def pressure_level_weight(x: float):
    # Create similar weighting to pressure level as in graphcast paper.
    # See fig 6 in graphcast paper
    # In summaru the weights are normalized to sum to 1 so that the highest
    # pressure level has the smallest weight.

    plevels = np.asarray([300, 500, 700, 850, 925, 1000])
    plevels_norm = plevels / np.sum(plevels)

    y = plevels_norm[np.where(plevels == x)][0]

    return round(y, 4)


def main():
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="meps_analysis",
        help="Dataset to compute weights for (default: meps_analysis)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input file (default: None)",
    )

    args = parser.parse_args()

    static_dir_path = os.path.join("data", args.dataset, "static")

    # Create parameter weights based on height and parameter name

    w_list = []
    for par in constants.param_names:
        name, leveln, levelv = par.split("_")
        if leveln == "isobaricInhPa":
            w = pressure_level_weight(int(levelv))
            if name in ("u", "v"):
                w = w * 0.5
        else:
            if name in ("ucorr", "vcorr"):
                w = 0.5
            elif name in ("tcorr", "rcorr", "fgcorr"):
                w = 1.0
            elif name == "pres" and leveln == "heightAboveSea":
                w = 1.0
            else:
                w = 0.2
        w_list.append(w)

    w_list = np.array(w_list)

    assert round(np.sum(w_list), 1) == 9.6, "weights do not sum to 9.6 (sum={})".format(
        np.sum(w_list)
    )

    print("Saving parameter weights...")

    np.save(
        os.path.join(static_dir_path, "parameter_weights.npy"), w_list.astype("float32")
    )

    # Load dataset without any subsampling
    ds = AnalysisDataset(
        args.dataset, split="trainval", standardize=False, input_file=args.input_file
    )  # Without standardization

    loader = torch.utils.data.DataLoader(
        ds, args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    # Compute mean and std.-dev. of each parameter (+ flux forcing) across full dataset
    print("Computing mean and std.-dev. for parameters...")
    means = []
    squares = []

    # we don't have fluxes, unlike original neural-lam
    # (use sun elevation angle instead)

    for init_batch, target_batch, _, forcing_batch in tqdm(loader):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t, N_grid, d_features)
        means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
        squares.append(torch.mean(batch**2, dim=(1, 2)))  # (N_batch, d_features,)
    mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    print("Saving mean, std.-dev...")
    torch.save(mean, os.path.join(static_dir_path, "parameter_mean.pt"))
    torch.save(std, os.path.join(static_dir_path, "parameter_std.pt"))

    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")

    ds_standard = AnalysisDataset(
        args.dataset, split="trainval", standardize=True, input_file=args.input_file
    )

    loader_standard = torch.utils.data.DataLoader(
        ds_standard, args.batch_size, shuffle=False, num_workers=args.n_workers
    )

    diff_means = []
    diff_squares = []

    for init_batch, target_batch, _, _ in tqdm(loader_standard):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t', N_grid, d_features)
        # Note: batch contains only 1h-steps

        batch_diffs = batch[:, 1:] - batch[:, :-1]
        # (N_batch', N_t-1, N_grid, d_features)

        diff_means.append(
            torch.mean(batch_diffs, dim=(1, 2))
        )  # (N_batch', d_features,)
        diff_squares.append(
            torch.mean(batch_diffs**2, dim=(1, 2))
        )  # (N_batch', d_features,)

    assert len(diff_means), "unable to calculate diffs"
    diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
    diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

    print("Saving one-step difference mean and std.-dev...")
    torch.save(diff_mean, os.path.join(static_dir_path, "diff_mean.pt"))
    torch.save(diff_std, os.path.join(static_dir_path, "diff_std.pt"))


if __name__ == "__main__":
    main()
