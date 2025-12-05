import os
import random

import numpy as np
import pandas as pd
from numba import njit


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def calculate_interpolation_grid(main_shock_location, cell_size_degs, num_cells_half):
    min_lat = main_shock_location[0] - num_cells_half * cell_size_degs
    max_lat = main_shock_location[0] + num_cells_half * cell_size_degs
    min_lon = main_shock_location[1] - num_cells_half * cell_size_degs
    max_lon = main_shock_location[1] + num_cells_half * cell_size_degs
    n_pixels_lat = 2 * num_cells_half  # Number of discretizations in the latitude direction
    n_pixels_lon = n_pixels_lat  # Number of discretizations in the longitude direction (cells)
    return min_lat, max_lat, min_lon, max_lon, n_pixels_lat, n_pixels_lon


def return_regions():
    # useful dictionary to identify some regions in the world
    regions = {}
    regions['Japan'] = (22, 46, 123, 148)
    regions['Italy'] = (36, 47, 6, 19)
    regions['Turkey'] = (36, 41, 26, 46)
    regions['Us'] = (25, 51, -127, -64)
    return regions


# Earth's radius in kilometers at the equator
earth_radius_km = 6371  # km


def convert_lat_lon_to_km(data, origin_lat, origin_lon):
    """
    Convert latitude and longitude to kilometers relative to an origin.

    Parameters:
    data (DataFrame): The DataFrame containing 'lat' and 'lon' columns.
    origin_lat (float): Latitude of the origin point.
    origin_lon (float): Longitude of the origin point.

    Returns:
    DataFrame: The original DataFrame with added 'km_lat' and 'km_lon' columns.
    """

    # Conversion factors
    km_per_degree_lat = np.pi * earth_radius_km / 180.0
    km_per_degree_lon = km_per_degree_lat * np.cos(np.radians(origin_lat))

    # Convert degrees to kilometers
    data['km_lat'] = (data['lat'] - origin_lat) * km_per_degree_lat
    data['km_lon'] = (data['lon'] - origin_lon) * km_per_degree_lon


@njit(nogil=True)
def haversine_distances(stations_positions_rads, grid_point):
    """
    Compute the haversine distances between a grid point (in radians) and all stations.
    args:
    stations_positions_rads: (Nstations, 2) array with the latitude and longitude of the stations in radians.
    grid_point: (2,) array with the latitude and longitude of the grid point in radians.
    returns:
    (Nstations, ) array with the haversine distances.
    """
    lat1 = stations_positions_rads[:, 0]
    lon1 = stations_positions_rads[:, 1]
    lat2 = grid_point[0]
    lon2 = grid_point[1]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def get_ngl_stations(save_path, url_list="http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"):
    """ Get the list of stations from the Nevada Geodetic Laboratory. """
    if os.path.exists(save_path):
        print("ngl stations already downloaded")
        return pd.read_csv(save_path, sep=" ", parse_dates=['begin', 'end'])
    print("Downloading ngl stations")
    df_list = pd.read_csv(url_list, sep=r"\s+", on_bad_lines='skip', parse_dates=['Dtbeg', 'Dtend'])
    df_list.rename(columns={'Sta': 'name', 'Lat(deg)': 'lat', 'Long(deg)': 'lon', 'Dtbeg': 'begin', 'Dtend': 'end'},
                   inplace=True)
    df_list['lon'] = ((df_list['lon'] + 180) % 360) - 180  # Convert to [-180, 180]
    # df_list['lon'] = df_list['lon'] % 360
    # mask = df_list['lon'] > 180
    # df_list[mask] = df_list[mask] - 360
    selection = df_list[['name', 'lat', 'lon', 'begin', 'end']]
    selection.to_csv(save_path, sep=" ", index=False)
    return selection


def get_ngl_gps_data(station_name, data_type):
    """ Get the GPS data from the Nevada Geodetic Laboratory. """
    base_url = "http://geodesy.unr.edu/gps_timeseries/" + data_type
    data = pd.read_csv(base_url + "/IGS14/" + station_name + "." + data_type, sep=r"\s+", parse_dates=["YYMMMDD"],
                       date_format='%y%b%d')
    data.rename(columns={'YYMMMDD': 'date', '_latitude(deg)': 'lat', '_longitude(deg)': 'lon', '__height(m)': 'z'},
                inplace=True)
    data['lon'] = ((data['lon'] + 180) % 360) - 180  # Convert to [-180, 180]

    data.sort_values('date', inplace=True)
    # data[['lat', 'lon']] = np.radians(data[['lat', 'lon']]) * earth_radius # Convert to km
    data['z'] /= 1000  # Convert to km
    return data[['date', 'lat', 'lon', 'z']]


# @njit(nogil=True)
def interpolate_displacements(grid, main_shock_location_rads, gps_stations_positions_rads, gps_stations_displacements,
                              n_rows, n_cols, latr_origin, lonr_origin, spacing_rads, sigma_rads,
                              min_station_pixel_distance_km):
    """
    grid:   (pixel,pixel,channel)  is modified in-place
        channel:     0,1,2 -> n,e,u displacements
        channel:     -1 -> (the masking value) is stored there (the distance to the closest station)
    gps_stations_displacements: (Nstations,3)  # GPS input data for a sinlge day.
    n_rows, n_cols: shape of out_array
    latr_origin, lonr_origin : origin of the image (point (0,0), or lower left corner of the image in lat,lon)
    spacing_rads: cell size (in radians)
    sigma_rads:   sigma of the gaussian used for interpolation
    min_station_pixel_distance_km:  minimum distance to a station to consider the pixel
    """
    # (i,j): sweep over all pixels

    for i in range(n_rows):
        grid_latr = latr_origin + i * spacing_rads
        for j in range(n_cols):
            grid_lonr = lonr_origin + j * spacing_rads
            # grid_latr, grid_lonr: are absolute coordinates of the pixel
            # d is the array of all distances between stations and the current pixel (in radians)
            grid_pos_rads = np.array([grid_latr, grid_lonr])
            # distance pixel-main shock (in km)
            grid[i, j, -2] = haversine_distances(main_shock_location_rads.reshape(-1, 2),
                                                 grid_pos_rads) * earth_radius_km
            d = haversine_distances(gps_stations_positions_rads, grid_pos_rads)  # size Nstations
            d_min_km = d.min() * earth_radius_km
            w = np.exp(-0.5 * (d / sigma_rads) ** 2)  # size Nstations
            w_sum = w.sum()
            # when stations are far, the max will pick up essentially the weight relative to the closest station
            # max_w = w.max()
            # print("max_w", max_w, "sum w", w_sum)
            # when stations are close, the max may be larger than 1, but it's ok, we don't care.
            # if w_sum > 0 and max_w >= min_w:  # exclude the pixel if no station is close enough (max_w too small)
            if d_min_km <= min_station_pixel_distance_km:  # the pixel is valid
                grid[i, j, -1] = 1.0

            w /= w_sum  # normalize the weights
            interpolated_displacements = np.dot(w.flatten(),
                                                gps_stations_displacements)  # zero-th order method, only averages ## product (N,)@(N,3) -> (3,)
            grid[i, j, :3] = interpolated_displacements


# @njit(nogil=True)
def elasticity_interpolation(grid, main_shock_location_rads, stations_positions_rads, gps_stations_displacements,
                             n_rows, n_cols, latr_origin, lonr_origin, spacing_rads, sigma_rads,
                             min_station_pixel_distance_km, reg_factor=3, nu=0.5, index_ratio=0.6):
    '''
    This interpolation assumes the top crust to be modelled as a 2D thin elastic sheet
    Then one interplate displacements based on that model.
    Reference to the original paper:
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016GL070340
    '''

    cutoff = 2 * np.pi * spacing_rads * reg_factor
    N = stations_positions_rads.shape[0]
    bf_matr = np.zeros((2 * N, 2 * N))
    for i, i_pos in enumerate(stations_positions_rads):
        d_ij = haversine_distances(stations_positions_rads, i_pos) + cutoff
        for j, j_pos in enumerate(stations_positions_rads):
            q_ij = (3 - nu) * np.log(d_ij[j]) + (1 + nu) * ((i_pos[1] - j_pos[1]) ** 2) / d_ij[j] ** 2
            p_ij = (3 - nu) * np.log(d_ij[j]) + (1 + nu) * ((i_pos[0] - j_pos[0]) ** 2) / d_ij[j] ** 2
            w_ij = (-(1 + nu)) * (i_pos[0] - j_pos[0]) * (i_pos[1] - j_pos[1]) / d_ij[j] ** 2
            bf_matr[2 * i, 2 * j] = q_ij
            bf_matr[2 * i, 2 * j + 1] = w_ij
            bf_matr[2 * i + 1, 2 * j] = w_ij
            bf_matr[2 * i + 1, 2 * j + 1] = p_ij
    uu, vv, vvh = np.linalg.svd(bf_matr)
    index_cut = int(index_ratio * bf_matr.shape[0])
    known_vec = np.stack([gps_stations_displacements[:, 0], gps_stations_displacements[:, 1]]).T.flatten()
    # body_forces = np.linalg.inv(bf_matr)@known_vec
    body_forces = ((vvh.conj().T)[:, :index_cut] @ np.diag(1 / vv[:index_cut]) @ (uu.conj().T)[:index_cut,
                                                                                 :]) @ known_vec
    # v = np.zeros((n_rows, n_cols, 3))
    for i in range(n_rows):
        grid_latr = latr_origin + i * spacing_rads
        for j in range(n_cols):
            grid_lonr = lonr_origin + j * spacing_rads
            grid_pos_rads = np.array([grid_latr, grid_lonr])
            # distance pixel-main shock (in km)
            grid[i, j, -2] = haversine_distances(main_shock_location_rads.reshape(-1, 2),
                                                 grid_pos_rads) * earth_radius_km
            d = haversine_distances(stations_positions_rads, grid_pos_rads)
            d_min_km = d.min() * earth_radius_km
            if d_min_km <= min_station_pixel_distance_km:  # the pixel is valid
                grid[i, j, -1] = 1.0
            closest_station_idx = np.argmin(d)
            grid[i, j, 2] = np.exp(-0.5 * ((d[closest_station_idx]) / sigma_rads) ** 2)
            d += cutoff  # add the cutoff
            q_s = (3 - nu) * np.log(d) + (1 + nu) * ((grid_lonr - stations_positions_rads[:, 1]) ** 2) / d ** 2
            p_s = (3 - nu) * np.log(d) + (1 + nu) * ((grid_latr - stations_positions_rads[:, 0]) ** 2) / d ** 2
            w_s = (-(1 + nu)) * (grid_latr - stations_positions_rads[:, 0]) * (
                    grid_lonr - stations_positions_rads[:, 1]) / d ** 2
            grid[i, j, 0] = np.sum(q_s * body_forces[::2] + w_s * body_forces[1::2])
            grid[i, j, 1] = np.sum(w_s * body_forces[::2] + p_s * body_forces[1::2])


# @njit(nogil=True)
def create_classification_soft_labels(y, row_indices, col_indices, spacing, sigma, mode='max'):
    # allows to soften the hard labels in {0,1} by a small Gaussian smoothing.
    # we go through Aftershocks and add to the local pixel (i,j) a Gaussian (r=distance between i,j and the AS)
    # y is initially full of zeros
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            for (r, c) in zip(row_indices, col_indices):
                distance_squared = (i - r) ** 2 + (j - c) ** 2
                weight = np.exp(-0.5 * distance_squared * ((spacing / sigma) ** 2))
                if mode == 'max':
                    y[i, j] = max(y[i, j], weight)
                else:
                    y[i, j] += weight
    if mode != 'max':
        np.minimum(y, 1.0, out=y)


def create_regression_soft_labels(y, seismic_moments, row_indices, col_indices, spacing, sigma):
    #  Smooths seismic moment values across a grid using a Gaussian kernel.
    # we go through Aftershocks and add to the local pixel (i,j) a Gaussian (r=distance between i,j and the AS)
    # y is initially full of zeros
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            for (r, c, seismic_moment) in zip(row_indices, col_indices, seismic_moments):
                distance_squared = (i - r) ** 2 + (j - c) ** 2
                weight = np.exp(-0.5 * distance_squared * ((spacing / sigma) ** 2))
                y[i, j] += weight * seismic_moment
