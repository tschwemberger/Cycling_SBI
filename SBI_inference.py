import numpy as np
from numpy.random import uniform, normal
import sys
import matplotlib as mpl
import corner
import torch
import torch.nn as nn
import torch.distributions as dist
from pathlib import Path
from scipy.signal import lfilter
from scipy.ndimage import binary_dilation, median_filter, gaussian_filter, uniform_filter1d, gaussian_filter1d
import xml.etree.ElementTree as ET
import fitdecode
import requests
import srtm
from datetime import datetime, timezone
from SBI_embedding import PhysicsEmbedding
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FixedFormatter, NullFormatter
#Options
my_colors = ['#785EF0', '#DC267F', '#FFB000', '#006352', 'turquoise', 'b', 'r', 'g', 'y']
from matplotlib import rc, rcParams 
params = {'text.usetex' : True, 
          'font.size' : 18, 
          'font.family' : 'cmr10', 
          'axes.formatter.use_mathtext' : True,
          'axes.unicode_minus' : False,
          'axes.labelsize' : 20,
          'xtick.major.pad': 5,
          'axes.labelpad' : 4,
          'figure.figsize' : (7,5),
          'xtick.top' : True,
          'xtick.direction' : 'in',
          'ytick.right' : True,
          'ytick.direction' : 'in',
          'axes.prop_cycle' : mpl.cycler(color=my_colors),
          'legend.frameon' : False}
rcParams.update(params)

rcParams["text.usetex"] = False
rcParams["mathtext.fontset"] = "cm"
rcParams["font.family"] = "serif"
    
cmap = plt.get_cmap("tab10")

prior_min = np.array( [ 40.0,  0.15, 0.0018, 0.0001 ] )
prior_max = np.array( [ 150.0, 0.50, 0.006,  0.01   ] )

p_law_index = 0.25
G = 9.81
sig = 3.0 # std for mass prior (kg)

def sigmoid(x, loc, width):
    return 1 / (1 + np.exp(-(x-loc)/width))

def F_imp_over_m(v, grad, k, v_roll, sigma_roll, amp):
    return ((k * G * np.cos(np.arctan(grad))
             * ( (v+1)**p_law_index + amp * np.exp(-( ( v-v_roll) / sigma_roll )**2) ) ))

def accel_min(grad, speed, wind, rho, params_l=prior_min, params_h=prior_max, imp_params=(1,5,1)):
    m_min, CdA_min, Crr_min, k_min = params_l
    m_max, CdA_max, Crr_max, k_max = params_h
    amp, v_roll, sigma_roll = imp_params
    theta = np.arctan(grad)
    return -(G * np.sin(theta) + G * Crr_max * np.cos(theta)
             + 0.5*rho * CdA_max * (speed+wind)**2/m_min
             + F_imp_over_m(speed, grad, k_max, v_roll, sigma_roll, amp))

def braking_mask(accel, power, speed, omega, grad, wind, rho, P_lo=30.0, t_strong=0.8,
                 v_hi=9.0, buf=3):
    lo, hi = np.asarray(prior_min), np.asarray(prior_max)      # explicit, so the stale-default trap can't bite
    a_s, low_p = gaussian_filter(accel, 3), power < P_lo
    flag = (a_s < accel_min(grad, speed, wind, rho, params_l=lo, params_h=hi)) & low_p   # weak tier
    if t_strong is not None:                                    # None disables the strong tier
        a_mid = accel_min(grad, speed, wind, rho,
                          params_l=(1 - t_strong)*lo + t_strong*hi,
                          params_h=(1 - t_strong)*hi + t_strong*lo) + 0.0
        strong = (a_s < a_mid) & low_p & (speed < v_hi)
        flag |= strong
    return binary_dilation(flag, iterations=buf) if buf else flag

def sharp_turn_mask(x, y, speed, sample_rate_hz=1, baseline_m=25.0, r_min=20.0,
                    a_lat_max=1.5, smooth_m=6.0, ds=2.0, dilate_s=3.0):
    """
    Flags samples in or near sharp turns. x, y are along-ground coordinates in metres.
    Curvature is measured over a fixed arclength baseline, not a fixed time step,
    so low-speed GPS jitter doesn't masquerade as turning.
    """
    n = len(x)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    s = s + np.arange(n) * 1e-9              # strictly increasing, safe when stationary
    if s[-1] < 2 * baseline_m:
        return np.zeros(n, dtype=bool)

    sg = np.arange(0.0, s[-1], ds)           # uniform arclength grid
    xg = gaussian_filter1d(np.interp(sg, s, x), smooth_m / ds, mode='nearest')
    yg = gaussian_filter1d(np.interp(sg, s, y), smooth_m / ds, mode='nearest')
    psi = np.unwrap(np.arctan2(np.gradient(yg), np.gradient(xg)))

    m = max(1, int(round(baseline_m / (2 * ds))))
    dpsi = np.empty_like(psi)
    dpsi[m:-m] = psi[2 * m:] - psi[:-2 * m]
    dpsi[:m], dpsi[-m:] = dpsi[m], dpsi[-m - 1]
    kappa = np.interp(s, sg, np.abs(dpsi) / (2 * m * ds))   # 1/m, back on the time grid

    a_lat = speed ** 2 * kappa
    flag = ((kappa > 1.0 / r_min) & (speed > 1.0)) | (a_lat > a_lat_max)

    k = int(round(dilate_s * sample_rate_hz))
    if k > 0:
        flag = binary_dilation(flag, structure=np.ones(2 * k + 1, dtype=bool))
    return flag

def compute_clean_mask(t, speed, acceleration, power, grad, dhdt, wind, rho,
                       x=None, y=None, grad_limit=-0.03, accel_physical_limit=2.0,
                       sample_rate_hz=1, **turn_kw):
    """
    Sample-level rejection mask shared by real rides and simulations.
    Returns (is_clean, flags). x, y are optional; without them the turn filter is a no-op.
    """
    flags = {
        'implausible_accel': np.abs(acceleration) > accel_physical_limit,
        'motorized':         (speed > 15.0) & (power < 5.0) & (grad >= -0.01),
        'time_gap':          np.append(1.0, np.diff(t)) > 3.0,
        'steep_descent':     uniform_filter1d(grad, size=5, mode='nearest') < grad_limit,
        'walking':           (uniform_filter1d(grad, size=5, mode='nearest') > 0.08) & (power < 50.0),
        'braking':           braking_mask(acceleration, power, speed, dhdt, grad, wind, rho),
        'sharp_turn':        (sharp_turn_mask(x, y, speed, sample_rate_hz, **turn_kw)
                              if x is not None else np.zeros(len(t), dtype=bool)),
    }
    is_clean = ~np.logical_or.reduce(list(flags.values()))
    return is_clean, flags

def pick_bins(t_epoch, bin_s=900):
    bin_t = np.unique(np.r_[np.arange(t_epoch[0], t_epoch[-1], bin_s), t_epoch[-1]])
    idx = np.clip(np.searchsorted(t_epoch, bin_t), 0, len(t_epoch) - 1)
    return idx, t_epoch[idx]

def wind_from_responses(resp, bin_t, t_epoch):
    U, V, G, T, P = [], [], [], [], []
    for tb, r in zip(bin_t, resp):
        h = r["hourly"]; th = np.asarray(h["time"], float)
        s = np.asarray(h["wind_speed_10m"], float); d = np.radians(h["wind_direction_10m"])
        U.append(np.interp(tb, th, s * np.sin(d)))          # components of the "from" vector
        V.append(np.interp(tb, th, s * np.cos(d)))
        G.append(np.interp(tb, th, h["wind_gusts_10m"]))
        T.append(np.interp(tb, th, h["temperature_2m"]))
        P.append(np.interp(tb, th, h["surface_pressure"]))
    u, v, g, T_, P_ = [np.interp(t_epoch, bin_t, np.asarray(x)) for x in (U, V, G, T, P)]
    return np.hypot(u, v), np.degrees(np.arctan2(u, v)) % 360, g, T_, P_

def fetch_wind(lat, lon, t_epoch, bin_s=900):
    idx, bin_t = pick_bins(t_epoch, bin_s)
    day = lambda x: str(np.datetime64(int(x), 's').astype('datetime64[D]'))
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=dict(
        latitude=",".join(f"{x:.4f}" for x in lat[idx]),
        longitude=",".join(f"{x:.4f}" for x in lon[idx]),
        start_date=day(t_epoch[0]), end_date=day(t_epoch[-1]),   # handles rides crossing UTC midnight
        hourly="wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,surface_pressure",
        wind_speed_unit="ms", timezone="GMT", timeformat="unixtime"), timeout=60).json()
    return wind_from_responses(r if isinstance(r, list) else [r], bin_t, t_epoch)

def add_open_source_elevations(latitudes, longitudes):
    """Fills or replaces elevation data using offline NASA SRTM files.
    Downloads and caches tiles locally under ~/.cache/srtm/
    """
    # Initialize the SRTM data grabber
    elevation_data = srtm.get_data()

    # Query elevations for each coordinate pair
    dem_elevations = []
    for lat, lon in zip(latitudes, longitudes):
        alt = elevation_data.get_elevation(lat, lon)
        dem_elevations.append(alt if alt is not None else 0.0)

    return np.array(dem_elevations)

def hampel_mask(v, half_window=5, n_sigma=4.0, min_dev=1.5):
    w = 2 * half_window + 1
    med = median_filter(v, size=w, mode="nearest")
    dev = np.abs(v - med)
    mad = median_filter(dev, size=w, mode="nearest")
    thresh = np.maximum(n_sigma * 1.4826 * mad, min_dev)
    return dev > thresh

def complementary_filter(z_baro, z_gps, alpha=0.90):
    """Fuses barometric and GPS elevation to eliminate dynamic pressure steps.
    alpha: Weight given to the barometric trend.
           0.98 means 98% baro (smooth) and 2% GPS (stable, no wind error).
    """
    baro_change = np.diff(z_baro, prepend=z_baro[0])
    x = alpha * baro_change + (1 - alpha) * z_gps
    x[0] = z_gps[0]
    b = np.array([1.0])
    a = np.array([1.0, -alpha])
    return lfilter(b, a, x)

def clean_gpx_anomalies(t, x, y, z, z_gps, z_bar, moving_threshold_mps=0.1):
    # --- Filter out Stopped Time & Duplicate Timestamps ---
    dx_raw = np.diff(x)
    dy_raw = np.diff(y)
    dt_raw = np.diff(t)

    # Avoid division by zero for duplicates
    dt_raw_safe = np.where(dt_raw == 0, 1e-5, dt_raw)
    raw_speeds = np.sqrt(dx_raw**2 + dy_raw**2) / dt_raw_safe

    is_moving = (raw_speeds >= moving_threshold_mps) & (dt_raw > 0.1)

    # We always keep the 1st point, then any point where the transition was 'moving'
    keep_indices = np.where(is_moving)[0] + 1
    keep_indices = np.insert(keep_indices, 0, 0)  # Ensure we keep the starting point

    # Filter primary arrays
    t = t[keep_indices]
    x = x[keep_indices]
    y = y[keep_indices]
    z = z[keep_indices]
    z_gps = z_gps[keep_indices]
    z_bar = z_bar[keep_indices]

    dt_clean = np.diff(t)
    dt_clean_safe = np.where(dt_clean == 0, 1e-5, dt_clean)
    climbing_rate = np.diff(z) / dt_clean_safe

    # Identify physically impossible vertical jumps
    max_vertical_rate = 2.0  # m/s
    anomalous_z = climbing_rate > max_vertical_rate

    if np.any(anomalous_z):
        for idx in np.where(anomalous_z)[0]:
            if 0 < idx < len(z) - 1:
                z[idx + 1] = z[idx]
    return t, x, y, z, z_gps, z_bar

SEMICIRCLE_TO_DEG = 180.0 / 2**31

def import_fit(file_path):
    """Raw read: same return signature as import_gpx."""
    lats, lons, z, times, power = [], [], [], [], []

    with fitdecode.FitReader(file_path) as fit:
        for frame in fit:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA or frame.name != "record":
                continue
            if not (frame.has_field("position_lat") and frame.has_field("position_long")
                    and frame.has_field("timestamp")):
                continue  # no GPS fix / malformed record

            lat = frame.get_value("position_lat")
            lon = frame.get_value("position_long")
            if lat is None or lon is None:
                continue
            # Depending on the processor, these may already be degrees. Real
            # semicircle values are huge integers, so the magnitude is unambiguous.
            if abs(lat) > 180 or abs(lon) > 180:
                lat *= SEMICIRCLE_TO_DEG
                lon *= SEMICIRCLE_TO_DEG

            ts = frame.get_value("timestamp")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)  # FIT timestamps are UTC

            ele = None
            for name in ("enhanced_altitude", "altitude"):
                if frame.has_field(name):
                    ele = frame.get_value(name)
                    if ele is not None:
                        break

            p = frame.get_value("power") if frame.has_field("power") else None

            lats.append(lat)
            lons.append(lon)
            z.append(0.0 if ele is None else float(ele))
            times.append(ts.timestamp())
            power.append(0.0 if p is None else float(p))

    return (np.array(lats, dtype=float), np.array(lons, dtype=float),
            np.array(z, dtype=float), np.array(times, dtype=float),
            np.array(power, dtype=float))

def import_gpx(file_path):
    """Raw read: returns lats, lons, z, times (unix s), power as float arrays."""
    # XML Parsing using C-implemented ElementTree
    tree = ET.parse(file_path)
    root = tree.getroot()
 
    ns = {"gpx": root.tag.split("}")[0].strip("{")}
 
    points = root.findall(".//gpx:trkpt", ns)
 
    num_points = len(points)
 
    lats = np.zeros(num_points)
    lons = np.zeros(num_points)
    z = np.zeros(num_points)
    times = np.zeros(num_points)
    power = np.zeros(num_points)
 
    # Fast extraction loop (keeping Python overhead to a absolute minimum)
    for i, pt in enumerate(points):
        lats[i] = float(pt.attrib["lat"])
        lons[i] = float(pt.attrib["lon"])
 
        # Extract elevation
        ele = pt.find("gpx:ele", ns)
        if ele is not None:
            z[i] = float(ele.text)
 
        # Extract time and convert to unix timestamp
        t_node = pt.find("gpx:time", ns)
        if t_node is not None:
            # ISO time string parsing
            dt = datetime.fromisoformat(t_node.text.replace("Z", "+00:00"))
            times[i] = dt.timestamp()
 
        # Extract power from extensions (using a relaxed find to catch varying schemas)
        pow_node = pt.find(".//gpx:power", ns)
        if pow_node is None:
            # Fallback for Garmin extension style namespaces
            pow_node = pt.find(".//{*}power")
 
        if pow_node is not None:
            power[i] = float(pow_node.text)
 
    return lats, lons, z, times, power

def process_track(lats, lons, z, times, power):
    num_points = len(lats)
    if num_points < 2:
        raise ValueError("Not enough points to process.")
 
    t = times - times[0]
 
    z_bar = np.copy(z)
    z_gps = add_open_source_elevations(lats, lons)
    z_gps = gaussian_filter(z_gps, sigma=10)
    z = complementary_filter(z, z_gps, alpha=0.995)
    
    # Importing achival wind data including simulated gusts
    print("Importing archival weather data...")
    base_wind_speed, wind_direction, wind_gusts, temp, pres = fetch_wind(
        lats, lons, times)
 
    # Equirectangular Projection
    R = 6371000.0
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
 
    lat0 = lat_rad[0]
    lon0 = lon_rad[0]
 
    x = R * (lon_rad - lon0) * np.cos(lat0)
    y = R * (lat_rad - lat0)
 
    # Calculating the parallel component of wind...
    t_copy = np.copy(t)
    t, x, y, z, z_gps, z_bar = clean_gpx_anomalies(t, x, y, z, z_gps, z_bar)
    mask = np.isin(t_copy, t)
    power = power[mask]
    temp = temp[mask]
    pres = pres[mask]
    base_wind_speed = base_wind_speed[mask]
    wind_direction = wind_direction[mask]
    
    dt = np.append(np.diff(t), 1.0)
 
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    dt = np.diff(t)
    # Avoid nan/inf from paused or duplicate timestamps
    dt = np.where(dt == 0, 1e-5, dt)
 
    grad = np.nan_to_num(dz/np.sqrt(dx**2 + dy**2), nan=0.0, posinf=0.0, neginf=0.0)
    grad = np.append(grad, 0.0)
    grad = gaussian_filter(grad, sigma=3)
    dist = np.cumsum(np.sqrt(dx**2 + dy**2))
    dist = np.insert(dist, 0, 0.0)
 
    dist_steps = np.sqrt(dx**2 + dy**2)
    speed_mid = dist_steps / dt
    step_bad = (hampel_mask(speed_mid) | (speed_mid > 18))
    bad = np.zeros(len(x), dtype=bool)
    bad[:-1] |= step_bad
    bad[1:]  |= step_bad
    bad = binary_dilation(bad, iterations=1)   # one extra sample each side
    good = ~bad
 
    for arr in (x, y):
        arr[bad] = np.interp(t[bad], t[good], arr[good])
 
    bike_headings = np.arctan2(dy, dx)
    bike_headings = np.append(bike_headings, bike_headings[-1]) # match length
    wind_dir_rad = np.radians(90 - wind_direction)
    angle_diff = bike_headings - wind_dir_rad
    wind_parallel = base_wind_speed * np.cos(angle_diff)
    heading_diff = np.diff(bike_headings)
    heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
    heading_rate = np.abs(np.append(heading_diff, 0)) # Pad to match length
 
    # recompute speed from the cleaned track instead of interpolating it
    speed_mid = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt
 
    accel_mid = np.diff(speed_mid) / dt[:-1]
    accel_mid = np.append(accel_mid, 0.0)  # Pad end
 
    t_mid = t[:-1] + (dt / 2.0)
    speed = np.interp(t, t_mid, speed_mid)
    acceleration = np.interp(t, t_mid, accel_mid)
    rho_0 = 100*pres / (287.058 * (273.15+temp))
    rho = rho_0
 
    return t, dist, x, y, z, z_gps, z_bar, grad, power, speed, acceleration, wind_parallel, heading_rate, rho
 
def import_track(file_path, fmt=None):
    """New entry point: read by format, then process. Same return as the old import_gpx."""
    fmt = (fmt or Path(file_path).suffix.lstrip(".")).lower()
    if fmt == "gpx":
        raw = import_gpx(file_path)
    elif fmt == "fit":
        raw = import_fit(file_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return process_track(*raw)

def process_ride_to_chunks(f_name, data=None, sample_rate_hz=1, window_duration_sec=90, grad_limit=-0.03, low_power_threshold=30 ):
    """
    Slices and filters cycling time-series data including parallel wind data
    """
    if data is not None:
        t, d, x, y, z, z_gps, z_bar, grad, power, speed, acceleration, wind, dhdt, rho = data
    else:
        t, d, x, y, z, z_gps, z_bar, grad, power, speed, acceleration, wind, dhdt, rho = import_gpx(f_name)
    window_size = int(window_duration_sec * sample_rate_hz)
    is_clean, flags = compute_clean_mask(t, speed, acceleration, power, grad, dhdt, wind, rho,
                                        x=x, y=y, grad_limit=grad_limit,
                                        sample_rate_hz=sample_rate_hz)
    
    # Find contiguous "clean" segments [cite: 125]
    clean_transitions = np.diff(is_clean.astype(int))
    start_indices = np.where(clean_transitions == 1)[0] + 1
    end_indices = np.where(clean_transitions == -1)[0] + 1
    
    if is_clean[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if is_clean[-1]:
        end_indices = np.append(end_indices, len(is_clean))
        
    chunks = []
    
    for start, end in zip(start_indices, end_indices):
        segment_len = end - start
        if segment_len < window_size:
            continue
            
        # Extract features for this segment
        segment_features = np.column_stack((
            gaussian_filter(power[start:end], 1.5), 
            gaussian_filter(speed[start:end], 1.5), 
            gaussian_filter(acceleration[start:end], 1.5),
            grad[start:end],
            wind[start:end],
            rho[start:end]))
        
        # Slice into windows
        num_windows = segment_len // window_size
        for i in range(num_windows):
            chunk_start = i * window_size
            chunk_end = chunk_start + window_size
            chunks.append(segment_features[chunk_start:chunk_end])
            
    if len(chunks) == 0:
        print("Warning: No clean segments met the 60-second length criteria!")
        return np.empty((0, window_size, 7))
        
    return np.stack(chunks, axis=0)

def prepare_observation(real_ride_tensor):
    """Same preprocessing as the single-ride cell: subsample/pad, normalize, add mask channel."""
    actual_segments = real_ride_tensor.shape[0]
    if actual_segments > MAX_SEGMENTS:
        idx = torch.linspace(0, actual_segments - 1, steps=MAX_SEGMENTS).long()
        x_obs = real_ride_tensor[idx]
    elif actual_segments < MAX_SEGMENTS:
        pad = torch.zeros((MAX_SEGMENTS - actual_segments, 90, 6))
        x_obs = torch.cat([real_ride_tensor, pad], dim=0)
    else:
        x_obs = real_ride_tensor

    x_obs = x_obs.unsqueeze(0)
    mask_obs = (x_obs != 0.0)
    x_norm = torch.zeros_like(x_obs)
    x_norm[mask_obs] = ((x_obs - mean) / std)[mask_obs]

    real_mask = torch.zeros(MAX_SEGMENTS, dtype=torch.bool)
    real_mask[:min(actual_segments, MAX_SEGMENTS)] = True
    mask_channel = real_mask.float().view(1, MAX_SEGMENTS, 1, 1).expand(-1, -1, x_norm.shape[2], -1)
    return torch.cat([x_norm, mask_channel], dim=-1)

def wquantile(x, w, q):
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w) - 0.5 * w
    return np.interp(q, cdf / w.sum(), x)

def ess(w):
    return 1.0 / np.sum(w**2)

torch.set_num_threads(4)
theta_min, theta_max = 0.0, 1.0

V_ROLL, SIG_ROLL = 6.5, 1.0
LAGS = (0, 1, 2, 3)              # candidate lags (samples) between kinematics and power; negatives allowed
ROBUST_C = 2.5                   # a window's weight drops below 0.5 once its misfit exceeds ROBUST_C x the ride median
N_IRLS = 3                       # robust re-fit iterations
N_REG = 5



def scale_theta_to_unit(theta_phys):
    return (theta_phys - theta_min) / (theta_max - theta_min)

def unscale_theta_to_phys(theta_norm):
    return theta_norm * (theta_max - theta_min) + theta_min

def analyze_ride(real_ride_tensor, USE_MASS_PRIOR=False, sys_mass=90, widths=False):
    """
    mode="linear_k": replicates the original script, w = 10^(alpha * 1000k)
                     (exponential tilt in k, in units of 1000*k)
    mode="power":    true power law, w = k^alpha
    """
    x_norm = prepare_observation(real_ride_tensor)
    with torch.no_grad():
        samples_norm = posterior.sample((NUM_SAMPLES,), x=x_norm, show_progress_bars=False)
    samples_phys = unscale_theta_to_phys(samples_norm)

    # float64 copy in physical units (does not alias samples_phys)
    s = samples_phys.cpu().numpy().astype(np.float64)
    log10k = s[:, 3].copy()          # true log10 k, taken before unit conversion
    s[:, 1] = 10**s[:, 1]            # CdA
    s[:, 2] = 100 * 10**s[:, 2]      # 100 x Crr
    s[:, 3] = 1000 * 10**s[:, 3]     # 1000 x k_surf

    if USE_MASS_PRIOR:
        m = samples_phys[:, 0]
        p_new_mass = dist.Normal(loc=sys_mass, scale=sig)
        logw = (p_new_mass.log_prob(m) - p_train_mass.log_prob(m)).cpu().numpy().astype(np.float64)
    else:
        logw = np.zeros(NUM_SAMPLES)

    logw[np.isinf(logw)] = np.nan
    w = np.nan_to_num(np.exp(logw - np.nanmax(logw)))
    w /= w.sum()

    med = [wquantile(s[:, i], w, 0.5) for i in range(4)]
    if widths:
        med25 = [wquantile(s[:, i], w, 0.25) for i in range(4)]
        med75 = [wquantile(s[:, i], w, 0.75) for i in range(4)]
        return (med25, med, med75), ess(w)
    else:
        return med, ess(w)

def clean_log_axis(axis):
    axis.set_major_locator(LogLocator(base=10, numticks=10))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    axis.set_minor_formatter(NullFormatter())
    axis.set_major_formatter(FixedFormatter(['0.3', '1', '3']))
    axis.set_ticks([0.3, 1.0, 3.0])

def plot_corner(real_ride_tensor, sys_mass=None, MAX_SEGMENTS=100, NUM_SAMPLES=100000):
    if sys_mass is not None:
        USE_MASS_PRIOR = True
    else:
        USE_MASS_PRIOR = False

    actual_segments = real_ride_tensor.shape[0]
    if actual_segments > MAX_SEGMENTS:
        # Option A: If the real ride has more segments than your max training capacity 
        indices = torch.linspace(0, actual_segments - 1, steps=MAX_SEGMENTS).long()
        x_observed = real_ride_tensor[indices]
    elif actual_segments < MAX_SEGMENTS:
        # Option B: If it has fewer, zero-pad it exactly like we did the training data
        padding = torch.zeros((MAX_SEGMENTS - actual_segments, 90, 6))
        x_observed = torch.cat([real_ride_tensor, padding], dim=0)
    else:
        x_observed = real_ride_tensor

    x_observed = x_observed.unsqueeze(0) 
    mask_obs = (x_observed != 0.0)
    mean = x_mean_b.to(device)
    std = x_std_b.to(device)
    x_observed_norm = torch.zeros_like(x_observed)
    x_observed_norm[mask_obs] = ((x_observed - mean) / std)[mask_obs]

    real_mask = torch.zeros(MAX_SEGMENTS, dtype=torch.bool)
    real_mask[:min(actual_segments, MAX_SEGMENTS)] = True
    mask_channel_obs = real_mask.float().view(1, MAX_SEGMENTS, 1, 1).expand(-1, -1, x_observed_norm.shape[2], -1)
    x_observed_norm = torch.cat([x_observed_norm, mask_channel_obs], dim=-1)

    print("Drawing posterior samples...")
    samples_norm = posterior.sample((NUM_SAMPLES,), x=x_observed_norm)
    samples_phys = unscale_theta_to_phys(samples_norm)

    mass_samples = samples_phys[:, 0]

    if USE_MASS_PRIOR:
        # Re-weight the posterior by the ratio of new/training priors
        p_train_mass = dist.Uniform(theta_min[0], theta_max[0])
        p_new_mass = dist.Normal(loc=sys_mass, scale=sig) # assume reported weight isn't precise
        log_weights = p_new_mass.log_prob(mass_samples) - p_train_mass.log_prob(mass_samples)
        w = torch.exp(log_weights - log_weights.max())
        w = (w / w.sum()).cpu().numpy().astype(np.float64)
        # print(f"ESS = {1.0 / np.sum(w**2):.0f} / {len(w)}")
    else:
        w = None

    samples_np = samples_phys.cpu().numpy().copy()
    samples_np[:, 1] = 10**samples_np[:, 1]
    samples_np[:, 2] = 100 * 10**samples_np[:, 2]
    samples_np[:, 3] = 1000 * 10**samples_np[:, 3]


    # Parameters
    labels = [r"$m$ (kg)", r"$C_d A$", r"$100 \times C_{rr}^{(h)}$", r"$1000 \times C_{rr}^{(i)}$"]
    k_idx = 3  # Column index of k_surf
    n_bins = 20  # or whatever bin count you use for the other axes

    axes_scale = ["linear", "linear", "linear", "log"]
    def weighted_quantile(x, q, w=None):
        if w is None:
            return np.quantile(x, q)
        idx = np.argsort(x)
        xs, cw = x[idx], np.cumsum(w[idx])
        cw /= cw[-1]
        return np.interp(q, cw, xs)

    qlo, qhi = 0.0005, 0.9995
    ranges = [tuple(weighted_quantile(samples_np[:, i], [qlo, qhi], w)) for i in range(3)]
    ranges.append((samples_np[:, k_idx].min(), samples_np[:, k_idx].max()))

    common = dict(
        labels=labels, bins=n_bins, axes_scale=axes_scale, range=ranges,
        label_kwargs={"fontsize": 14}, plot_datapoints=False,
        fill_contours=True, plot_density=True)

    fig = corner.corner(samples_np, weights=w, color="C0",
                        show_titles=True, title_kwargs={"fontsize": 12}, **common)

    axes = np.array(fig.axes).reshape(4, 4)
    clean_log_axis(axes[3, 3].xaxis)          # k_surf histogram (x axis)
    clean_log_axis(axes[3, 0].yaxis)      # k_surf on the y axis of the bottom row

    fig.align_ylabels(axes[:, 0])
    for ax in axes[3, :]:
        ax.tick_params(axis="x", labelrotation=45)

    # plt.show()
    return fig

def plot_pie(data, inf_theta, title, p_thresh=30, out_name=None):
    t, d, x, y, z, z_gps, z_bar, grad, power, speed, acceleration, wind, dhdt, rho = data
    mask, flags = compute_clean_mask(t, speed, acceleration, power, grad, dhdt, wind, rho)

    m, cda, crr, k = inf_theta

    accel = gaussian_filter(acceleration, 3)

    v_roll     = uniform(5.0, 8.0) # velocity of maximum impedance
    sigma_roll = uniform(0.7, 1.3) # smoothness of impedance transition
    amp        = np.clip(normal(1, 0.2), 0, np.inf) # amplitude of impedance maximum

    v_air = speed + wind
    F_climb = m * (G * np.sin(np.arctan(grad)))
    F_rolling = crr * m * G * ( np.cos(np.arctan(grad)) )
                            #    * (1-sigmoid(speed, v_roll+2*sigma_roll, sigma_roll) ) )
    F_drag = 0.5 * rho * cda * v_air * np.abs(v_air)
    F_impedance = m * F_imp_over_m(speed, grad, k, v_roll, sigma_roll, amp)
    F_accel = m * accel

    p_climb = np.clip(gaussian_filter(speed * F_climb, 3), 0, np.inf)
    # May be negative on descents, but we'll clip that for stability and because we don't care as much about downhills
    p_accel = np.clip(gaussian_filter(speed * F_accel, 3), 0, np.inf)
    # should only be negative if braking, but gaussian filter for noise and clip for stability
    p_drag  = np.clip(gaussian_filter(speed * F_drag, 3), 0, np.inf)
    # No sailing allowed... The rest are fundamentally bounded
    p_imp   = speed * F_impedance
    p_roll  = speed * F_rolling

    inferred_powers = np.array([np.sum(p_climb[mask]),
                                np.sum(p_accel[mask]),
                                np.sum(p_drag[mask]),
                                np.sum(p_roll[mask]),
                                np.sum(p_imp[mask])])
    
    print('Work required at inferred parameters:', np.sum(inferred_powers))
    print('Work recorded:', np.sum(power[mask]))
    
    percentages = 100 * inferred_powers / inferred_powers.sum()
    labels = ['Climbs', 'Accelerations', 'Aero', 'Tires', 'Surface']
    legend_labels = [f'{label} ('+f'{pct:.1f}'+r'%)' for label, pct in zip(labels, percentages)]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(inferred_powers, labels=None, autopct='%1.1f%%', pctdistance=0.8)
    
    param_text = "\n".join([
        rf"$m_{{sys}}$ = {m:.1f} kg",
        rf"$C_d A$ = {cda:.3f} $m^2$",
        rf"$C_{{rr}}^{{(h)}}$ = {crr:.4f}",
        rf"$C_{{rr}}^{{(i)}}$ = {1e3*k:.2g}$\times 10^{{-3}}$",
    ])
    ax.text(0.96, 0.3, param_text, transform=ax.transAxes, va='center', ha='left', fontsize=18)

    fig.tight_layout()
    ax.legend(legend_labels, loc=(0.93, 0.45))
    ax.set_title(title)
    if out_name is not None:
        plt.savefig(out_name, bbox_inches='tight')
    # plt.show()
    return inferred_powers, fig, (np.sum(inferred_powers), np.sum(power[mask]))

PARAM_NAMES = ['mass', 'CdA', 'Crr', 'k']

def inj_recov(processed_ride, params, use_prior=False, n_rep=10,
              accel_sigma=3, accel_mode='raw', sys_mass=None):
    """
    Injection-recovery on a real ride's covariates (speed, accel, grad, wind, rho).
 
    Power is generated from `params` = (m, cda, crr, k) using a SMOOTHED copy of the
    measured accel as the "true" accel.
      accel_mode='raw'    -> network sees the raw measured accel (real accel noise
                             enters as a power/accel mismatch, as in real data)
      accel_mode='smooth' -> network sees the smoothed accel (no accel noise at all)
    Repeats n_rep times with fresh noise/nuisance draws and reports pulls + coverage.
    """
    m, cda, crr, k = params
    processed_ride = torch.tensor(processed_ride, dtype=torch.float32)
    truth = np.array(params, dtype=float)
    meds, p25s, p75s = [], [], []
 
    for rep in range(n_rep):
        sim_ride = []
        for seg in processed_ride:                      # seg: (T, 6)
            _, speed, accel_raw, grad, wind, rho = seg.T
            accel_true = gaussian_filter(accel_raw, accel_sigma)
 
            # nuisance params: these must match the draws used in training
            v_roll = uniform(2.0, 5.0)
            sigma_roll = 2 * 10 ** uniform(-0.5, 0.0)
            amp = uniform(0.7, 1.3)
            eta = uniform(0.9, 1.05)          # match training convention
 
            theta = np.arctan(grad)
            v_air = speed - wind
            F = (m * (G * np.sin(theta) + accel_true)
                 + crr * m * G * np.cos(theta)
                 + 0.5 * rho * cda * v_air * np.abs(v_air)
                 + k * m * G * np.cos(theta)
                 * (speed ** p_law_index + amp * np.exp(-((speed - v_roll) / sigma_roll) ** 2)))
            p = speed * F / eta
            p = np.maximum(p, 0.0)                      # coasting reads 0 W; match training sims
            p = p * normal(1, 0.02, len(p))
            p = gaussian_filter(p, 1.5)
 
            accel_obs = accel_raw if accel_mode == 'raw' else accel_true
            sim_ride.append(np.stack([p, speed, accel_obs, grad, wind, rho], axis=1))
 
        sim_ride = np.array(sim_ride)                   # (N, T, 7)
        sim_ride = torch.tensor(sim_ride, dtype=torch.float32)

        sm = m if sys_mass is None else sys_mass
        (p25, med, p75), n_eff = analyze_ride(sim_ride, USE_MASS_PRIOR=use_prior,
                                              sys_mass=sm, widths=True)
        meds.append(np.asarray(med)); p25s.append(np.asarray(p25)); p75s.append(np.asarray(p75))

    truth[2] *= 100
    truth[3] *= 1000
 
    meds, p25s, p75s = map(np.array, (meds, p25s, p75s))   # (n_rep, 4)
    sigma = (p75s - p25s) / 1.349                           # rough Gaussian-equivalent width
    pull = (meds - truth) / sigma
    inside = (truth >= p25s) & (truth <= p75s)
 
    print(f'{"param":6s} {"truth":>9s} {"mean med":>9s} {"mean pull":>10s} {"std pull":>9s} {"in 25-75":>9s}')
    for i, name in enumerate(PARAM_NAMES):
        print(f'{name:6s} {truth[i]:9.4f} {meds[:, i].mean():9.4f} {pull[:, i].mean():10.2f} '
              f'{pull[:, i].std():9.2f} {inside[:, i].mean():9.2f}')
    print('Calibrated posterior: mean pull ~0, std pull ~1, fraction inside ~0.5')
    return dict(med=meds, p25=p25s, p75=p75s, pull=pull, inside=inside)

def ride_analysis(data, sys_mass=None, corner=True, pie=True, inj_rec=None, p_thresh=30, NUM_SAMPLES=100000):
    if sys_mass is not None:
        USE_MASS_PRIOR = True
    else:
        USE_MASS_PRIOR = False
    data = np.array(data)
    # data[8] *= 1.2  # testing PM scaling
    chunks = process_ride_to_chunks(f_name=None, data=data)
    ride_tensor = torch.tensor(chunks, dtype=torch.float32)
    num_segs = chunks.shape[0]
    (m, cda, crr, k), n_eff = analyze_ride(ride_tensor, USE_MASS_PRIOR=False, sys_mass=sys_mass)
    m0 = np.copy(m)
    if USE_MASS_PRIOR:
        (m, cda, crr, k), n_eff = analyze_ride(ride_tensor, USE_MASS_PRIOR=USE_MASS_PRIOR, sys_mass=sys_mass)
    crr /= 100
    k /= 1e3

    if corner:
        corner_fig = plot_corner(ride_tensor, sys_mass=sys_mass, NUM_SAMPLES=NUM_SAMPLES)
    if pie:
        powers, pie_fig, work = plot_pie(data, (m, cda, crr, k), title=None, p_thresh=p_thresh)
    if inj_rec is not None:
        test = inj_recov(chunks, (m, cda, crr, k), use_prior=USE_MASS_PRIOR, n_rep=inj_rec, accel_sigma=3, accel_mode='smooth', sys_mass=sys_mass)
    return (m, cda, crr, k), (corner_fig, pie_fig), m0, work, num_segs

sys.modules['__main__'].PhysicsEmbedding = PhysicsEmbedding

posterior = torch.load('posterior.pt', weights_only=False)
slim = torch.load('training_data.pt')

x_mean = slim['x_mean']
x_std = slim['x_std']
theta_min = slim['theta_min']
theta_max = slim['theta_max']

x_mean_b = x_mean.view(1, 1, 1, 6)
x_std_b = x_std.view(1, 1, 1, 6)

device = torch.device("cpu")
MAX_SEGMENTS = 100
NUM_SAMPLES = 100000

mean = x_mean_b.to(device)
std = x_std_b.to(device)
p_train_mass = dist.Uniform(theta_min[0], theta_max[0])

# data = import_track('Grey_in_the_morning.fit')
# data = import_track('gpx_files/Lena.gpx')
# params = ride_analysis(data, sys_mass=None, NUM_SAMPLES=NUM_SAMPLES)