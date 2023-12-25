import cartopy
import numpy as np

wandb_project = "neural-lam"

seconds_in_year = 365 * 24 * 60 * 60  # Assuming no leap years in dataset (2024 is next)

# Log prediction error for these lead times
# should not be larger than ar_steps
val_step_log_errors = np.array([1, 2])  # , 5, 10, 12])

# Variable names
param_names = [
    "pres_heightAboveGround_0",
    "pres_heightAboveSea_0",
    "rcorr_heightAboveGround_2",
    "tcorr_heightAboveGround_2",
    "t_heightAboveGround_0",
    "fgcorr_heightAboveGround_10",
    "ucorr_heightAboveGround_10",
    "vcorr_heightAboveGround_10",
    "mld_heightAboveGround_0",
    "r_isobaricInhPa_1000",
    "r_isobaricInhPa_925",
    "r_isobaricInhPa_850",
    "r_isobaricInhPa_700",
    "r_isobaricInhPa_500",
    "r_isobaricInhPa_300",
    "t_isobaricInhPa_1000",
    "t_isobaricInhPa_925",
    "t_isobaricInhPa_850",
    "t_isobaricInhPa_700",
    "t_isobaricInhPa_500",
    "t_isobaricInhPa_300",
    "u_isobaricInhPa_1000",
    "u_isobaricInhPa_925",
    "u_isobaricInhPa_850",
    "u_isobaricInhPa_700",
    "u_isobaricInhPa_500",
    "u_isobaricInhPa_300",
    "v_isobaricInhPa_1000",
    "v_isobaricInhPa_925",
    "v_isobaricInhPa_850",
    "v_isobaricInhPa_700",
    "v_isobaricInhPa_500",
    "v_isobaricInhPa_300",
    "z_isobaricInhPa_1000",
    "z_isobaricInhPa_925",
    "z_isobaricInhPa_850",
    "z_isobaricInhPa_700",
    "z_isobaricInhPa_500",
    "z_isobaricInhPa_300",
]

param_names.sort()

assert len(param_names) == 39
param_names_short = []

for p in param_names:
    if p == "pres_heightAboveGround_0":
        name = "psurf"
    else:
        name = p.split("_")[0]

    level = p.split("_")[2]

    param_names_short.append("{}{}".format(name, level))

assert len(param_names_short) == len(param_names)

param_units = [
    "Pa",
    "Pa",
    "W/m\\textsuperscript{2}",
    "W/m\\textsuperscript{2}",
    "-",  # unitless
    "-",
    "K",
    "K",
    "K",
    "K",
    "m/s",
    "m/s",
    "m/s",
    "m/s",
    "kg/m\\textsuperscript{2}",
    "m\\textsuperscript{2}/s\\textsuperscript{2}",
    "m\\textsuperscript{2}/s\\textsuperscript{2}",
]

param_units = ["" for x in range(39)]

# Projection and grid
# TODO Do not hard code this, make part of static dataset files
grid_shape = (268, 238)  # (y, x)

lambert_proj_params = {
    "a": 6367470,
    "b": 6367470,
    "lat_0": 63.3,
    "lat_1": 63.3,
    "lat_2": 63.3,
    "lon_0": 15.0,
    "proj": "lcc",
}

grid_limits = [  # In projection
    -1059506.5523409774,  # min x
    1310493.4476590226,  # max x
    -1331732.4471934352,  # min y
    1338267.5528065648,  # max y
]

# Create projection
lambert_proj = cartopy.crs.LambertConformal(
    central_longitude=lambert_proj_params["lon_0"],
    central_latitude=lambert_proj_params["lat_0"],
    standard_parallels=(lambert_proj_params["lat_1"], lambert_proj_params["lat_2"]),
)
