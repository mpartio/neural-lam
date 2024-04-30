import torch

# import pytorch_lightning as pl
import numpy as np
from argparse import ArgumentParser
import time
import datetime as dt
import os
import xarray as xr
import rioxarray
import rasterio
import cartopy
import eccodes as ecc
from tqdm import tqdm
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from io import BytesIO
from neural_lam.weather_dataset import AnalysisDataset
from neural_lam import constants, utils

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


def parse_args():
    parser = ArgumentParser(description="Train or evaluate NeurWP models for LAM")

    # General options
    parser.add_argument(
        "--dataset",
        type=str,
        default="meps_analysis",
        help="Dataset, corresponding to name in data directory (default: meps_analysis)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hi_lam",
        help="Model architecture to train/evaluate (default: hi_lam)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="upper epoch limit (default: 200)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--loss", type=str, default="mse", help="Loss function to use (default: mse)"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="hierarchical",
        help="Graph to load and use in graph-based model (default: multiscale)",
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model checkpoint (default: None)",
        required=True,
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation (default: 1)",
    )
    parser.add_argument(
        "--pred_length",
        type=int,
        default=12,
        help="Number of hours to predict (default: 12)",
    )
    # Training options
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (default: 4)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) (default: sum)",
    )
    parser.add_argument(
        "--interleave",
        default=False,
        action="store_true",
        help="Interleave source data",
    )
    parser.add_argument("--output_file", type=str, default="output.grib2")

    # Evaluation options
    args = parser.parse_args()

    return args


args = parse_args()


def create_spatial_ref(x, y):
    lambert_proj_params = {
        "a": 6371229,
        "b": 6371229,
        "lat_0": 63.3,
        "lat_1": 63.3,
        "lat_2": 63.3,
        "lon_0": 15.0,
        "proj": "lcc",
    }

    grid_limits = [np.min(x), np.max(x), np.min(y), np.max(y)]

    # Create projection
    proj = cartopy.crs.LambertConformal(
        central_longitude=lambert_proj_params["lon_0"],
        central_latitude=lambert_proj_params["lat_0"],
        standard_parallels=(lambert_proj_params["lat_1"], lambert_proj_params["lat_2"]),
        globe=cartopy.crs.Globe(
            ellipse="sphere",
            semimajor_axis=lambert_proj_params["a"],
            semiminor_axis=lambert_proj_params["b"],
        ),
    )

    return proj


def create_xarray(data, times, variables):
    coords = np.load("data/meps_analysis/static/nwp_xy.npy")

    coords = (coords[0][0], coords[1].transpose()[0])

    x = coords[0]
    y = coords[1]

    times_ = []
    for t in times:
        x_ = []
        x_ = t[1:]
        # for t_ in t[2:]:
        # x_.append(t_.astype("datetime64[s]").astype("int"))
        times_.append(x_)

    #    times = [x.astype("datetime64[s]").astype("int") for x in times]

    times = np.asarray(times_)
    if len(times.shape) == 1:
        times = np.expand_dims(times, axis=0)

    pressure = [1000, 925, 850, 500, 700, 300]

    ds = xr.Dataset(
        coords={
            "x": x,
            "y": y,
            "forecast": list(range(data.shape[0])),
            "leadtime": list(range(data.shape[1])),
            "pressure": pressure,
            "height0": [0],
            "height2": [2],
            "height10": [10],
            "msl": [0],
        },
        data_vars={
            "forecast_time": (["forecast", "leadtime"], times),
        },
    )

    units = ["m/s", "m/s", "m2 s-2", "K", "-"]
    for i, p in enumerate(["u", "v", "z", "t", "r"]):
        arr = []

        for l in pressure:
            index = variables.index(f"{p}_isobaricInhPa_{l}")
            arr.append(data[:, :, :, :, index])

        # (1, 3, 268, 238, 6)
        arr = np.stack(arr, axis=-1)
        # (3, 6, 268, 238)
        arr = np.moveaxis(arr, 4, 2)
        arr = np.flip(arr, axis=3)
        ds[p] = (
            ["forecast", "leadtime", "pressure", "y", "x"],
            arr,
            {"units": units[i]},
        )

    units = ["m/s", "m/s", "m/s"]
    for i, p in enumerate(["ucorr", "vcorr", "fgcorr"]):
        index = variables.index(f"{p}_heightAboveGround_10")
        arr = data[:, :, :, :, index]
        arr = np.expand_dims(arr, axis=-1)
        # (3, 268, 238, 6)
        # (3, 6, 268, 238)
        arr = np.moveaxis(arr, 4, 2)
        arr = np.flip(arr, axis=3)
        ds[p] = (
            ["forecast", "leadtime", "height10", "y", "x"],
            arr,
            {"units": units[i]},
        )

    units = ["K", "-"]
    for i, p in enumerate(["tcorr", "rcorr"]):
        index = variables.index(f"{p}_heightAboveGround_2")
        arr = data[:, :, :, :, index]
        arr = np.expand_dims(arr, axis=-1)
        # (3, 268, 238, 6)
        # (3, 6, 268, 238)
        arr = np.moveaxis(arr, 4, 2)
        arr = np.flip(arr, axis=3)
        ds[p] = (
            ["forecast", "leadtime", "height2", "y", "x"],
            arr,
            {"units": units[i]},
        )

    units = ["Pa", "K", "m"]
    for i, p in enumerate(["pres", "t", "mld"]):
        index = variables.index(f"{p}_heightAboveGround_0")
        arr = data[:, :, :, :, index]
        arr = np.expand_dims(arr, axis=-1)
        arr = np.moveaxis(arr, 4, 2)
        arr = np.flip(arr, axis=3)
        ds[p] = (
            ["forecast", "leadtime", "height2", "y", "x"],
            arr,
            {"units": units[i]},
        )

    units = ["Pa"]
    for i, p in enumerate(["pres"]):
        index = variables.index(f"{p}_heightAboveSea_0")
        arr = data[:, :, :, :, index]
        arr = np.expand_dims(arr, axis=-1)
        arr = np.moveaxis(arr, 4, 2)
        arr = np.flip(arr, axis=3)
        ds[p] = (["forecast", "leadtime", "msl", "y", "x"], arr, {"units": units[i]})

    ds.rio.write_crs(create_spatial_ref(x, y), inplace=True)

    return ds

    ds = xr.Dataset(
        coords={
            "x": x,
            "y": y,
            "forecast": list(range(data.shape[0])),
            "leadtime": list(range(data.shape[1])),
        },
        data_vars={
            "forecast_time": (["forecast", "leadtime"], times),
        },
    )

    for i, param in enumerate(variables):
        var_data = data[:, :, :, :, i]
        var_data = np.flip(var_data, axis=2)  # .swapaxes(2,3)
        ds[param] = (["forecast", "leadtime", "y", "x"], var_data)

    ds.rio.write_crs(create_spatial_ref(x, y), inplace=True)

    return ds


def save_grib(output_data, times, filepath, grib_options=None):
    assert filepath[-5:] == "grib2"
    bio = BytesIO()
    times = times[0]
    analysistime = dt.datetime.utcfromtimestamp(int(times[0]) / 1e9)
    # (1, 6, 268, 238, 39)
    assert (
        len(times) == output_data.shape[1]
    ), "times ({}) do not match data ({})".format(len(times), output_data.shape[1])

    param_keys = {
        "gust": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 22,
            "productDefinitionTemplateNumber": 8,
            "typeOfStatisticalProcessing": 2,
            "typeOfFirstFixedSurface": 103,
            "level": 10,
            "lengthOfTimeRange": 1,
        },
        "mld": {
            "discipline": 0,
            "parameterCategory": 3,
            "parameterNumber": 18,
            "typeOfFirstFixedSurface": 103,
            "level": 0,
        },
        "pres": {
            "discipline": 0,
            "parameterCategory": 3,
            "parameterNumber": 0,
            "typeOfFirstFixedSurface": 103,
            "level": 0,
        },
        "r": {
            "discipline": 0,
            "parameterCategory": 1,
            "parameterNumber": 1,
            "typeOfFirstFixedSurface": 100,
            "level": 1000,
        },
        "t": {
            "discipline": 0,
            "parameterCategory": 0,
            "parameterNumber": 0,
            "typeOfFirstFixedSurface": 100,
            "level": 1000,
        },
        "u": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 2,
            "typeOfFirstFixedSurface": 100,
            "level": 1000,
        },
        "v": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 3,
            "typeOfFirstFixedSurface": 100,
            "level": 1000,
        },
        "z": {
            "discipline": 0,
            "parameterCategory": 3,
            "parameterNumber": 4,
            "typeOfFirstFixedSurface": 100,
            "level": 1000,
        },
    }

    def pk(param, override={}):
        x = param_keys[param].copy()
        if override is not None:
            for k, v in override.items():
                x[k] = v
        return x

    grib_keys = [
        pk("gust"),
        pk("mld"),
        pk("pres"),
        pk("pres", {"typeOfFirstFixedSurface": 101}),
        pk("r"),
        pk("r", {"level": 300}),
        pk("r", {"level": 500}),
        pk("r", {"level": 700}),
        pk("r", {"level": 850}),
        pk("r", {"level": 925}),
        pk("r", {"typeOfFirstFixedSurface": 103, "level": 2}),
        pk("t", {"typeOfFirstFixedSurface": 103, "level": 0}),
        pk("t"),
        pk("t", {"level": 300}),
        pk("t", {"level": 500}),
        pk("t", {"level": 700}),
        pk("t", {"level": 850}),
        pk("t", {"level": 925}),
        pk("t", {"typeOfFirstFixedSurface": 103, "level": 2}),
        pk("u"),
        pk("u", {"level": 300}),
        pk("u", {"level": 500}),
        pk("u", {"level": 700}),
        pk("u", {"level": 850}),
        pk("u", {"level": 925}),
        pk("u", {"typeOfFirstFixedSurface": 103, "level": 10}),
        pk("v"),
        pk("v", {"level": 300}),
        pk("v", {"level": 500}),
        pk("v", {"level": 700}),
        pk("v", {"level": 850}),
        pk("v", {"level": 925}),
        pk("v", {"typeOfFirstFixedSurface": 103, "level": 10}),
        pk("z"),
        pk("z", {"level": 300}),
        pk("z", {"level": 500}),
        pk("z", {"level": 700}),
        pk("z", {"level": 850}),
        pk("z", {"level": 925}),
    ]
    assert len(grib_keys) == output_data.shape[-1]
    for i in range(len(times)):
        forecasttime = analysistime + dt.timedelta(hours=i)

        for j in range(0, output_data.shape[-1]):
            data = output_data[0, i, :, :, j]
            h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            ecc.codes_set(h, "gridType", "lambert")
            ecc.codes_set(h, "shapeOfTheEarth", 6)
            ecc.codes_set(h, "Nx", data.shape[1])
            ecc.codes_set(h, "Ny", data.shape[0])
            ecc.codes_set(h, "DxInMetres", 2370000 / (data.shape[1] - 1))
            ecc.codes_set(h, "DyInMetres", 2670000 / (data.shape[0] - 1))
            ecc.codes_set(h, "jScansPositively", 1)
            ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 50.319616)
            ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0.27828)
            ecc.codes_set(h, "Latin1InDegrees", 63.3)
            ecc.codes_set(h, "Latin2InDegrees", 63.3)
            ecc.codes_set(h, "LoVInDegrees", 15)
            ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
            ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
            ecc.codes_set(h, "dataDate", int(analysistime.strftime("%Y%m%d")))
            ecc.codes_set(h, "dataTime", int(analysistime.strftime("%H%M")))
            ecc.codes_set(h, "centre", 86)
            ecc.codes_set(h, "generatingProcessIdentifier", 251)
            ecc.codes_set(h, "packingType", "grid_ccsds")
            ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 1)

            ecc.codes_set(h, "typeOfGeneratingProcess", 2)  # deterministic forecast
            ecc.codes_set(h, "typeOfProcessedData", 2)  # analysis and forecast products

            for k, v in grib_keys[j].items():
                ecc.codes_set(h, k, v)

            forecasthour = int((forecasttime - analysistime).total_seconds() / 3600)

            if j == 0:
                forecasthour -= 1

            ecc.codes_set(h, "forecastTime", forecasthour)

            data = np.flipud(data)
            ecc.codes_set_values(h, data.flatten())
            ecc.codes_write(h, bio)
            ecc.codes_release(h)

    if filepath[0:5] == "s3://":
        write_to_s3(filepath, bio)
    else:
        try:
            os.makedirs(os.path.dirname(filepath))
        except FileExistsError as e:
            pass
        except FileNotFoundError as e:
            pass
        with open(filepath, "wb") as fp:
            fp.write(bio.getbuffer())

    print(f"Wrote file {filepath}")


def test_single(m, prev_state, prev_prev_state, batch_static_features, forcing):
    with torch.no_grad():
        prev_state = prev_state.to(m.device)
        prev_prev_state = prev_prev_state.to(m.device)
        batch_static_features = batch_static_features.to(m.device)
        forcing = forcing.to(m.device)
        out = m.predict_step(
            prev_state, prev_prev_state, batch_static_features, forcing
        )

    return out


def print_gpu_memory():
    print(
        "torch.cuda.memory_allocated: {:.0f}MB".format(
            torch.cuda.memory_allocated(0) / 1024 / 1024
        )
    )
    print(
        "torch.cuda.memory_reserved: {:.0f}MB".format(
            torch.cuda.memory_reserved(0) / 1024 / 1024
        )
    )
    print(
        "torch.cuda.max_memory_reserved: {:.0f}MB".format(
            torch.cuda.max_memory_reserved(0) / 1024 / 1024
        )
    )

def replace_border_with_truth(border_state, predicted_state):
    pass


def test(m, ds):
    ds_stats = utils.load_dataset_stats(args.dataset, "cpu")
    data_mean, data_std = (
        ds_stats["data_mean"],
        ds_stats["data_std"],
    )
    static_data_dict = utils.load_static_data(args.dataset)

    border_mask = static_data_dict["border_mask"]
    interior_mask = 1 - border_mask

    forecasts = []

    print_gpu_memory()

    for x, y, static_features, forcing in tqdm(ds):
        prev_prev_state = x[0, :, :].unsqueeze(0).to(m.device)
        prev_state = x[1, :, :].unsqueeze(0).to(m.device)
        static_features = static_features.unsqueeze(0).to(m.device)

        # y = y * data_std + data_mean

        initial_time = (
            x[1, :, :].unsqueeze(0).reshape(1, 268, 238, 39) * data_std + data_mean
        )

        prediction_list = [x[1, :, :].unsqueeze(0).reshape(1, 268, 238, 39)]
        prediction_list = [initial_time]
        # unroll the forecast manually to save memory

        for i in range(args.pred_length):
            assert prev_state.shape == (
                1,
                63784,
                39,
            ), f"prev_state.shape is {prev_state.shape}, should be (1, 63784, 39)"
            assert prev_prev_state.shape == (
                1,
                63784,
                39,
            ), f"prev_prev_state.shape is {prev_prev_state.shape}, should be (1, 63784, 39)"
            assert static_features.shape == (
                1,
                63784,
                1,
            ), f"static_features.shape is {static_features.shape}, should be (1, 63784, 1)"

            this_forcing = forcing[i, :, :].unsqueeze(0).to(m.device)

            assert this_forcing.shape == (
                1,
                63784,
                15,
            ), f"forcing.shape is {this_forcing.shape}, should be (1, 63784, 15)"

            predicted_state = m.predict_step(
                prev_state, prev_prev_state, static_features, this_forcing
            ).to("cpu")

            border_state = y[i, ...].unsqueeze(0)
            # Overwrite border with true state
            if True:
                new_state = replace_border_with_truth(border_state, predicted_state)

            new_state = border_mask * border_state + interior_mask * predicted_state
            if False:
                import matplotlib.pyplot as plt
                plt.figure(1)
                plt.imshow(border_state.reshape(1, 268, 238, 39)[0, :, :, 0])
                plt.figure(2)
                plt.imshow(border_mask.reshape(268, 238, 1)[:, :, 0])
                plt.figure(3)
                plt.imshow(predicted_state.reshape(1, 268, 238, 39)[0, :, :, 0])
                plt.figure(4)
                plt.imshow(interior_mask.reshape(268, 238, 1)[:, :, 0])
                plt.figure(5)
                plt.imshow(new_state.reshape(1, 268, 238, 39)[0, :, :, 0])
                plt.show()
                sys.exit(1)
            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

            output = new_state[:] * data_std + data_mean

            # output = border_mask * border_state + interior_mask * output
            output = output.reshape(1, 268, 238, 39)
            prediction_list.append(output)

            prev_state = prev_state.to(m.device)

        forecasts.append(prediction_list)

    forecasts = np.squeeze(np.asarray(forecasts))
    if len(forecasts.shape) == 4:
        forecasts = np.expand_dims(forecasts, axis=0)
    return forecasts


def main():
    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"

    # Load data

    ds = AnalysisDataset(
        args.dataset,
        split="test",
        pred_length=args.pred_length,
        interleave=args.interleave,
    )

    # Instatiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        # torch.set_float32_matmul_precision("high")  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]

    m = model_class.load_from_checkpoint(
        "{}/min_val_loss.ckpt".format(args.load), args=args
    )

    m.eval()
    with torch.no_grad():
        output_data = test(m, ds)

    truth_times = ds.get_times()  # [0][1:-1]
    truth_times[0] = truth_times[0][1:]

    if args.output_file[-5:] == "grib2":
        save_grib(output_data, truth_times, args.output_file)
        truth = np.expand_dims(np.moveaxis(ds.data(), 1, -1), 0)  # (15, 39, 268, 238)
        truth = truth[:, 1:, :, :, :]
        save_grib(truth, truth_times, "truth.grib2")

    elif args.output_file[-3:] == "nc":
        xds = create_xarray(output_data, ds.get_times(), constants.param_names)
        xds.to_netcdf(args.output_file)


if __name__ == "__main__":
    main()
