#!/usr/bin/env python3
"""
Reads a CSV file with sky map, scrambles them to mimic the incorrect/correct calibration,
converts to galactic-like coordinates for KM3NeT, and plots the result.

Implements:
  • Extended maximum-likelihood estimate (per zoom box) with correct background normalization
  • ON/OFF significance in local coordinates (az/zen)
  • Deterministic scrambling + deterministic signal injection

The zoom panel is saved as: <outfile_base>_zoom.png
"""

import sys
import math
import random
import hashlib
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.patches import Circle

# Extra imports
import pandas as pd
from scipy.stats import poisson, norm
from scipy.stats import vonmises_fisher

# --- Constants ---
KM3NET_LAT_DEG = 36.0 + 16.0 / 60.0
KM3NET_LAT_RAD = math.radians(KM3NET_LAT_DEG)

R_EQ_TO_GAL = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [ 0.4941094279, -0.4448296300,  0.7469822445],
    [-0.8676661490, -0.1980763734,  0.4559837762]
], dtype=float)

# --- Robust CSV readers (NEW) -------------------------------------------------

def read_ra_dec_csv_robust(path: str):
    """
    Robust CSV reader for columns 'ra_rad' and 'dec_rad'.

    Handles:
      - Windows newline oddities (open with newline="")
      - UTF-8 BOM (utf-8-sig)
      - comma/semicolon/tab delimiter (auto-sniff)
      - empty lines (skipped)
      - whitespace around headers/values
    """
    ra_vals, dec_vals = [], []

    # newline="" is critical for csv module correctness across platforms.
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        except csv.Error:
            dialect = csv.excel
            dialect.delimiter = ","

        reader = csv.DictReader(f, dialect=dialect, skipinitialspace=True)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header row.")

        field_map = {name.strip(): name for name in reader.fieldnames}
        if "ra_rad" not in field_map or "dec_rad" not in field_map:
            raise ValueError(
                f"{path}: expected columns 'ra_rad' and 'dec_rad', got {list(field_map.keys())}"
            )

        ra_key = field_map["ra_rad"]
        dec_key = field_map["dec_rad"]

        for row in reader:
            if not row or all((v is None or str(v).strip() == "") for v in row.values()):
                continue

            try:
                ra_vals.append(float(str(row[ra_key]).strip()))
                dec_vals.append(float(str(row[dec_key]).strip()))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path}: bad row {row}") from e

    return np.array(ra_vals, dtype=float), np.array(dec_vals, dtype=float)


def read_two_col_csv_int_float(path: str, key_int: str, key_float: str):
    """
    Robustly reads a 2-column CSV with header. Returns dict[int] -> float.

    Handles:
      - Windows newline oddities (newline="")
      - UTF-8 BOM (utf-8-sig)
      - delimiter sniffing
      - empty rows
      - whitespace around headers/values
    """
    out = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        except csv.Error:
            dialect = csv.excel
            dialect.delimiter = ","

        reader = csv.DictReader(f, dialect=dialect, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header row.")

        field_map = {name.strip(): name for name in reader.fieldnames}
        if key_int not in field_map or key_float not in field_map:
            raise ValueError(
                f"{path}: expected columns '{key_int}' and '{key_float}', got {list(field_map.keys())}"
            )

        k_int = field_map[key_int]
        k_float = field_map[key_float]

        for row in reader:
            if not row or all((v is None or str(v).strip() == "") for v in row.values()):
                continue
            try:
                out[int(str(row[k_int]).strip())] = float(str(row[k_float]).strip())
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path}: bad row {row}") from e

    return out

# --- Utility helpers (new) ----------------------------------------------------

def _uniform01_from_hash(*values):
    """
    Deterministic uniform [0,1) from a SHA-256 hash of the provided values.
    """
    h = hashlib.sha256("|".join(f"{v:.16e}" if isinstance(v, float) else str(v)
                                for v in values).encode("utf-8")).digest()
    u64 = int.from_bytes(h[:8], "big", signed=False)
    return (u64 / 2**64)

def _rng_from_seed(seed_int):
    """
    Deterministic NumPy Generator from a 64-bit seed.
    """
    return np.random.default_rng(seed_int & ((1 << 64) - 1))

def modular_delta_deg(a_deg, b_deg):
    """
    Smallest signed difference a-b in degrees, mapped to (-180, 180].
    """
    return ((a_deg - b_deg + 180.0) % 360.0) - 180.0

# --- Systematics --------------------------------------------------------------

def get_systematics(seed, offsets_file, calibration_file):
    systematics = {}

    offsets = read_two_col_csv_int_float(offsets_file, key_int="du_id", key_float="offset_ns")
    calibration = read_two_col_csv_int_float(calibration_file, key_int="du_id", key_float="calib_ns")

    systematics['miscal'] = sum(abs(offsets[du] + calibration[du]) for du in calibration.keys())
    systematics['avgcal'] = systematics['miscal'] / len(offsets)

    x_anchor = np.array([0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 3.0, 4.0, 5.0, 20.0])
    y_anchor = np.array([0.0, 0.05, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 180.0])
    systematics['radius'] = math.radians(
        np.interp(min(20.0, systematics['avgcal']), x_anchor, y_anchor)
    )

    systematics['seed'] = seed + int(hashlib.sha256(str(calibration).encode('utf-8')).hexdigest(), 16) % (2**32)
    return systematics

def apply_systematics(radius, calseed, day_fraction):
    """
    Deterministic variation around radius (±10%) and a rotation phase.
    Now fully deterministic given (radius,calseed,day_fraction).
    """
    local_seed = int((calseed ^ int(day_fraction * 1e12)) & 0xFFFFFFFF)
    rng = random.Random(local_seed)
    alpha_scale = max(0.1, min(10.0, rng.gauss(1.0, 0.10)))
    alpha = max(0.0, min(math.pi, radius * alpha_scale))
    phi = rng.uniform(0, 2*math.pi)
    return alpha, phi

def deterministic_day_fraction(ra, dec, calseed):
    """
    Deterministic 'pseudo-time' in [0,1), derived from (ra, dec, calseed).
    """
    return _uniform01_from_hash(calseed, ra, dec)

def scramble_point(ra, dec, radius, calseed, apply_systematics):
    """
    Deterministically move an event according to systematics.
    """
    day_fraction = deterministic_day_fraction(ra, dec, calseed)
    alpha, phi = apply_systematics(radius, calseed, day_fraction)

    sin_dec = math.sin(dec)
    cos_dec = math.cos(dec)
    x0 = cos_dec * math.cos(ra)
    y0 = cos_dec * math.sin(ra)
    z0 = sin_dec
    v0 = np.array([x0, y0, z0])

    e_theta = np.array([-sin_dec * math.cos(ra),
                        -sin_dec * math.sin(ra),
                         cos_dec])
    e_phi = np.array([-math.sin(ra), math.cos(ra), 0.0])

    e_theta_norm = np.linalg.norm(e_theta)
    e_phi_norm = np.linalg.norm(e_phi)
    if e_theta_norm == 0 or e_phi_norm == 0:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, v0)) > 0.9:
            tmp = np.array([0.0,1.0,0.0])
        e_theta = np.cross(v0, tmp); e_theta /= np.linalg.norm(e_theta)
        e_phi = np.cross(e_theta, v0); e_phi /= np.linalg.norm(e_phi)
    else:
        e_theta /= e_theta_norm
        e_phi   /= e_phi_norm

    disp = math.sin(alpha) * (math.cos(phi) * e_theta + math.sin(phi) * e_phi)
    new_vec = math.cos(alpha) * v0 + disp
    new_vec /= np.linalg.norm(new_vec)

    x, y, z = new_vec
    new_dec = math.asin(np.clip(z, -1.0, 1.0))
    new_ra  = math.atan2(y, x)
    if new_ra > math.pi: new_ra -= 2 * math.pi
    elif new_ra < -math.pi: new_ra += 2 * math.pi
    return new_ra, new_dec

def gaussian_smear_tangent(ra, dec, sigma_rad, rng=None):
    """
    Gaussian on tangent plane (small-angle) around (ra, dec).
    Accepts a deterministic NumPy Generator 'rng'; falls back to global RNG if None.
    """
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)
    v = np.array([x, y, z])

    tmp = np.array([1.0,0.0,0.0]) if abs(v[0]) < 0.9 else np.array([0.0,1.0,0.0])
    u_vec = np.cross(v, tmp); u_vec /= np.linalg.norm(u_vec)
    w_vec = np.cross(v, u_vec)

    if rng is None:
        dx, dy = np.random.normal(0, sigma_rad, 2)
    else:
        dx, dy = rng.normal(0.0, sigma_rad, 2)

    new_vec = v + dx*u_vec + dy*w_vec
    new_vec /= np.linalg.norm(new_vec)

    new_ra = math.atan2(new_vec[1], new_vec[0])
    new_dec = math.asin(np.clip(new_vec[2], -1, 1))
    return new_ra, new_dec

# --- Frame conversions --------------------------------------------------------

def local_horizontal_to_equatorial_vector(az, zen, lat_rad):
    alt = 0.5 * math.pi - zen
    x_loc = math.cos(alt) * math.cos(az)
    y_loc = math.cos(alt) * math.sin(az)
    z_loc = math.sin(alt)
    v_loc = np.array([x_loc, y_loc, z_loc])

    rot_angle = math.radians(90.0) - lat_rad
    Rx = np.array([
        [1,0,0],
        [0,math.cos(rot_angle),-math.sin(rot_angle)],
        [0,math.sin(rot_angle), math.cos(rot_angle)]
    ])
    v_eq = Rx @ v_loc
    v_eq /= np.linalg.norm(v_eq)
    return v_eq

def equatorial_vector_to_ra_dec(v_eq):
    x, y, z = v_eq
    ra = math.atan2(y, x)
    dec = math.asin(np.clip(z, -1, 1))
    return ra, dec

def equatorial_to_galactic_angles(ra_eq, dec_eq):
    x = math.cos(dec_eq) * math.cos(ra_eq)
    y = math.cos(dec_eq) * math.sin(ra_eq)
    z = math.sin(dec_eq)

    v_gal = R_EQ_TO_GAL @ np.array([x, y, z])
    v_gal /= np.linalg.norm(v_gal)

    lon_gal = math.atan2(v_gal[1], v_gal[0])
    lat_gal = math.asin(np.clip(v_gal[2], -1, 1))
    return lon_gal, lat_gal

def local_to_galactic(az, zen):
    v_eq = local_horizontal_to_equatorial_vector(az, zen, KM3NET_LAT_RAD)
    ra_eq, dec_eq = equatorial_vector_to_ra_dec(v_eq)
    return equatorial_to_galactic_angles(ra_eq, dec_eq)

# --- Likelihood utilities -----------------------------------------------------

def angular_separation_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    r1 = math.radians(ra1_deg); d1 = math.radians(dec1_deg)
    r2 = math.radians(ra2_deg); d2 = math.radians(dec2_deg)
    sin_d = math.sin((d2 - d1) / 2.0)
    sin_r = math.sin((r2 - r1) / 2.0)
    a = sin_d**2 + math.cos(d1)*math.cos(d2)*sin_r**2
    a = max(0.0, min(1.0, a))
    return 2.0 * math.degrees(math.asin(math.sqrt(a)))

def maximize_extended_likelihood_fixed_background(S_vals, B_val, n_b_fixed, N_events):
    if N_events == 0:
        return 0.0, 0.0, 0.0

    eps = 1e-300
    ngrid = 2001
    n_s_grid = np.linspace(0.0, float(N_events), ngrid)

    def logL_ext(n_s):
        model_vals = n_s * S_vals + n_b_fixed * B_val
        return np.sum(np.log(np.clip(model_vals, eps, None))) - (n_s + n_b_fixed)

    logL_vals = np.array([logL_ext(n) for n in n_s_grid])
    imax = int(np.argmax(logL_vals))
    n_s_peak = n_s_grid[imax]

    half = max(0.5, float(N_events) / 10.0)
    low = max(0.0, n_s_peak - half)
    high = min(float(N_events), n_s_peak + half)

    n_s_fine = np.linspace(low, high, 2001)
    logL_fine = np.array([logL_ext(n) for n in n_s_fine])
    jmax = int(np.argmax(logL_fine))

    n_s_hat = float(n_s_fine[jmax])
    logL_hat = float(logL_fine[jmax])
    logL_null = float(logL_ext(0.0))
    return n_s_hat, logL_hat, logL_null

# --- ON/OFF and helpers (LOCAL COORDS: az/zen, all angles in radians) ---------

def solid_angle_roi(roi_angle, degrees=True):
    roi_angle = np.asarray(roi_angle)
    if degrees:
        if np.any(roi_angle < 0) or np.any(roi_angle > 180):
            raise ValueError("roi_angle must be in the range [0, 180] degrees.")
        theta = np.deg2rad(roi_angle)
    else:
        if np.any(roi_angle < 0) or np.any(roi_angle > np.pi):
            raise ValueError("roi_angle must be in the range [0, π] radians.")
        theta = roi_angle
    omega = 2 * np.pi * (1 - np.cos(theta))
    if np.isscalar(roi_angle) or np.ndim(roi_angle) == 0:
        return float(omega)
    return omega

def solid_angle_band(theta_min, theta_max, degrees=True):
    theta_min = np.asarray(theta_min)
    theta_max = np.asarray(theta_max)
    if degrees:
        if np.any(theta_min < 0) or np.any(theta_max > 180):
            raise ValueError("Angles must be in the range [0, 180] degrees.")
        theta_min = np.deg2rad(theta_min)
        theta_max = np.deg2rad(theta_max)
    else:
        if np.any(theta_min < 0) or np.any(theta_max > np.pi):
            raise ValueError("Angles must be in the range [0, π] radians.")
    if np.any(theta_min > theta_max):
        raise ValueError("theta_min must be less than or equal to theta_max.")
    omega = 2 * np.pi * (np.cos(theta_min) - np.cos(theta_max))
    if (np.isscalar(theta_min) and np.isscalar(theta_max)) or \
       (np.ndim(theta_min) == 0 and np.ndim(theta_max) == 0):
        return float(omega)
    return omega

def poisson_pvalue(n_obs, mu):
    n_obs = np.asarray(n_obs)
    mu = np.asarray(mu)
    if np.any(n_obs < 0):
        raise ValueError("n_obs must contain non-negative integers.")
    if np.any(mu < 0):
        raise ValueError("mu must be non-negative.")
    if np.any(n_obs != np.floor(n_obs)):
        raise ValueError("n_obs must contain integer values.")
    p_value = 1 - poisson.cdf(n_obs - 1, mu)
    p_value = np.where(n_obs == 0, 1.0, p_value)
    if (np.isscalar(n_obs) and np.isscalar(mu)) or (np.ndim(p_value) == 0):
        return float(p_value)
    return p_value

def pvalue_to_sigma(pvalue, one_sided=True):
    pvalue = np.asarray(pvalue)
    if np.any((pvalue <= 0) | (pvalue >= 1)):
        raise ValueError("All p-values must be between 0 and 1 (exclusive).")
    sigma = norm.ppf(1 - pvalue) if one_sided else norm.ppf(1 - pvalue/2)
    if np.isscalar(pvalue) or np.ndim(pvalue) == 0:
        return float(sigma)
    return sigma

def signal_events_generator(ra, dec, smearing_angle, n_events=1, degrees=True):
    if degrees:
        ra = np.deg2rad(ra); dec = np.deg2rad(dec); smearing_angle = np.deg2rad(smearing_angle)
    mean = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    kappa = 1.0 / smearing_angle**2
    samples = vonmises_fisher(mean, kappa).rvs(n_events)
    x, y, z = samples.T
    ra_new = np.arctan2(y, x) % (2*np.pi)
    dec_new = np.arcsin(z)
    if degrees:
        ra_new = np.rad2deg(ra_new); dec_new = np.rad2deg(dec_new)
    if n_events == 1:
        return float(ra_new[0]), float(dec_new[0])
    return ra_new, dec_new

def poisson_significance(events_df_local: pd.DataFrame,
                         source_az, source_zen, roi_angle, source_name=None,
                         verbose: bool = True):
    theta_min = max(source_zen - roi_angle, 0.0)
    theta_max = min(source_zen + roi_angle, np.pi)

    omega_ON = solid_angle_roi(roi_angle, degrees=False)
    omega_band = solid_angle_band(theta_min, theta_max, degrees=False)
    omega_OFF = omega_band - omega_ON
    if omega_OFF <= 0:
        raise ValueError("OFF solid angle is non-positive; adjust ROI/band.")

    df = events_df_local[
        (events_df_local["zen_rad"] >= theta_min) &
        (events_df_local["zen_rad"] <= theta_max)
    ].copy()
    if df.empty:
        raise ValueError("No events in local zenith band; cannot estimate background.")

    dphi = (df["az_rad"] - source_az + np.pi) % (2*np.pi) - np.pi
    cos_g = np.cos(df["zen_rad"]) * np.cos(source_zen) + \
            np.sin(df["zen_rad"]) * np.sin(source_zen) * np.cos(dphi)
    cos_g = np.clip(cos_g, -1.0, 1.0)
    df["angular_distance"] = np.arccos(cos_g)

    on_events_df  = df[df["angular_distance"] <= roi_angle]
    off_events_df = df[df["angular_distance"] >  roi_angle]
    if off_events_df.empty:
        raise ValueError("OFF region empty in local coords; cannot estimate background.")

    n_ON  = len(on_events_df)
    n_OFF = len(off_events_df)

    mu_bkg = omega_ON * n_OFF / omega_OFF
    pvalue = poisson_pvalue(n_ON, mu_bkg)
    sigma  = pvalue_to_sigma(pvalue, one_sided=True) if (0.0 < pvalue < 1.0) else 0.0

    if verbose:
        if source_name is None:
            print(f'[LOCAL ON/OFF] Z = {sigma:.2f}σ ; n_ON={n_ON} ; n_BKG={mu_bkg:.3f} ; p={pvalue:.3e}')
        else:
            print(f'[LOCAL ON/OFF] {source_name}: Z = {sigma:.2f}σ ; n_ON={n_ON} ; n_BKG={mu_bkg:.3f} ; p={pvalue:.3e}')

    return float(theta_min), float(theta_max), float(sigma)

# --- Plotting -----------------------------------------------------------------

def plot_aitoff_and_zoom(az_array, zen_array, gal_coords,
                         extra_sources=None, extra_smeared=None,
                         outfile_base="aitoff",
                         verbose: bool = True):

    lon_local = (az_array + math.pi) % (2*math.pi) - math.pi
    lat_local = np.pi/2 - zen_array

    gal_lon = np.array([c[0] for c in gal_coords])
    gal_lat = np.array([c[1] for c in gal_coords])
    gal_lon_deg = np.degrees(gal_lon)
    gal_lat_deg = np.degrees(gal_lat)

    events_df_local = pd.DataFrame({
        "az_rad":  az_array.astype(float),
        "zen_rad": zen_array.astype(float)
    })

    fig_all, ax_all = plt.subplots(figsize=(10,5), subplot_kw={'projection':'aitoff'})
    ax_all.set_title("Reconstructed events (Aitoff projection, LOCAL az/alt)")
    ax_all.grid(True)

    sc_main = ax_all.scatter(lon_local, lat_local, s=8, alpha=0.7, marker='o', label="Background")

    cursor = mplcursors.cursor(sc_main, hover=True)
    @cursor.connect("add")
    def on_hover(sel):
        i = sel.index
        az_deg = math.degrees((az_array[i] + 2*math.pi) % (2*math.pi))
        alt_deg = math.degrees(lat_local[i])
        sel.annotation.set_text(
            f"Local az={az_deg:.2f}°\nLocal alt={alt_deg:.2f}°\n"
            f"Gal lon={gal_lon_deg[i]:.2f}°\nGal lat={gal_lat_deg[i]:.2f}°"
        )
        sel.annotation.get_bbox_patch().set(alpha=0.85, facecolor='white')

    if extra_smeared:
        extra_az = np.array([c[0] for c in extra_smeared])
        extra_zen = np.array([c[1] for c in extra_smeared])
        extra_lon_local = (extra_az + math.pi) % (2*math.pi) - math.pi
        extra_lat_local = np.pi/2 - extra_zen
        ax_all.scatter(extra_lon_local, extra_lat_local, s=15, color='green',
                       alpha=0.9, marker='o', label="Cosmic neutrino")

    if extra_sources:
        src_az = np.array([c[0] for c in extra_sources])
        src_zen = np.array([c[1] for c in extra_sources])
        src_lon_local = (src_az + math.pi) % (2*math.pi) - math.pi
        src_lat_local = np.pi/2 - src_zen
        ax_all.scatter(src_lon_local, src_lat_local, s=120, color='red', alpha=1.0,
                       marker='*', edgecolor='black', linewidths=0.5,
                       label="Sources")

    ax_all.legend(loc='upper left', bbox_to_anchor=(-0.05, 1))
    plt.savefig(outfile_base + "_aitoff.png", dpi=300, bbox_inches='tight')

    fig_zoom, axes_zoom = plt.subplots(2, 2, figsize=(18, 12))
    axes_zoom = axes_zoom.flatten()
    plt.subplots_adjust(hspace=0.22)

    if extra_sources:
        ra_all = gal_lon_deg
        dec_all = gal_lat_deg

        sigma_deg = 0.1
        area_box_deg2 = 10.0 * 10.0
        src_names = ["PKS 0239+108", "TXS 0506+056", "Vela X", "Markarian 421"]

        r_core_deg = 0.5
        area_core = math.pi * (r_core_deg ** 2)

        roi_onoff_deg = 0.3
        roi_onoff_rad = math.radians(roi_onoff_deg)

        for i, (gal_lon_src, gal_lat_src) in enumerate([(c[2], c[3]) for c in extra_sources]):
            ax = axes_zoom[i]
            ax.set_title(src_names[i], fontsize=14)
            ax.set_xlabel("Galactic Lon (deg)", fontsize=12)
            ax.set_ylabel("Galactic Lat (deg)", fontsize=12)
            ax.grid(alpha=0.25)
            ax.set_aspect('equal', adjustable='box')

            ra_src_deg = math.degrees(gal_lon_src)
            dec_src_deg = math.degrees(gal_lat_src)

            delta_ra = ((ra_all - ra_src_deg + 180.0) % 360.0) - 180.0
            in_box_mask = (np.abs(delta_ra) <= 5.0) & (np.abs(dec_all - dec_src_deg) <= 5.0)

            ra_box = ra_all[in_box_mask]
            dec_box = dec_all[in_box_mask]
            N_events = len(ra_box)

            if N_events > 0:
                dists = np.array([angular_separation_deg(ra_box[j], dec_box[j], ra_src_deg, dec_src_deg)
                                  for j in range(N_events)])
                N_core = int(np.sum(dists <= r_core_deg))
            else:
                dists = np.array([])
                N_core = 0

            area_box = area_box_deg2
            area_outside_core = max(area_box - area_core, 1e-6)

            N_outside_core = max(N_events - N_core, 0)

            if N_outside_core > 0:
                background_density = N_outside_core / area_outside_core
            else:
                background_density = (N_events / area_box) if N_events > 0 else 1e-6

            n_b_expected = background_density * area_box

            ax.scatter(ra_box, dec_box, s=10, alpha=0.6, color='gray',
                       label="Background", zorder=2)

            if extra_smeared:
                ra_sm = np.degrees([c[2] for c in extra_smeared])
                dec_sm = np.degrees([c[3] for c in extra_smeared])
                delta_sm = ((ra_sm - ra_src_deg + 180.0) % 360.0) - 180.0
                mask_sm = (np.abs(delta_sm) <= 5.0) & (np.abs(dec_sm - dec_src_deg) <= 5.0)

                ax.scatter(ra_sm[mask_sm], dec_sm[mask_sm], s=30,
                           color='green', alpha=0.9,
                           label="Cosmic neutrino", zorder=3)

            if N_events > 0:
                S_vals = np.zeros(N_events, dtype=float)
                for j in range(N_events):
                    ang_deg = angular_separation_deg(ra_box[j], dec_box[j], ra_src_deg, dec_src_deg)
                    S_vals[j] = (1.0 / (2.0 * math.pi * sigma_deg**2)) * math.exp(-0.5 * (ang_deg / sigma_deg)**2)
            else:
                S_vals = np.zeros(0, dtype=float)

            circle_small = Circle((ra_src_deg, dec_src_deg), 0.1,
                                  fill=False, edgecolor='red',
                                  linewidth=1.5, zorder=1)
            circle_large = Circle((ra_src_deg, dec_src_deg), 0.3,
                                  fill=False, edgecolor='black',
                                  linewidth=1.5, zorder=1)
            ax.add_patch(circle_small)
            ax.add_patch(circle_large)

            ax.set_xlim(ra_src_deg - 5, ra_src_deg + 5)
            ax.set_ylim(dec_src_deg - 5, dec_src_deg + 5)

            B_val = 1.0 / area_box_deg2
            n_b_fixed = n_b_expected

            n_s_best, logL_best, logL_null = maximize_extended_likelihood_fixed_background(
                S_vals, B_val, n_b_fixed, N_events)

            TS = max(0.0, 2.0 * (logL_best - logL_null))
            significance_ml = math.sqrt(TS) if TS > 0 else 0.0

            _, _, significance_onoff = poisson_significance(
                events_df_local,
                source_az=src_az[i],
                source_zen=src_zen[i],
                roi_angle=roi_onoff_rad,
                source_name=src_names[i],
                verbose=verbose
            )

            legend = ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02),
                               fontsize=10, frameon=True)
            legend.set_title(f"Z_ML = {significance_ml:.2f}σ\nZ_ON/OFF = {significance_onoff:.2f}σ",
                             prop={'size': 10})
            legend.get_frame().set_alpha(0.85)
            legend.get_frame().set_facecolor("white")

            ax.text(
                0.02, 0.98,
                f"nₛ = {n_s_best:.1f}",
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', linewidth=0.5)
            )

            if verbose:
                print(f"{src_names[i]}: N_box={N_events}, n_b_exp={n_b_expected:.2f}, "
                      f"n_s={n_s_best:.2f}, Z_ML={significance_ml:.3f}σ, Z_ON/OFF={significance_onoff:.3f}σ")

    plt.tight_layout()

    zoom_png = outfile_base + "_zoom.png"
    try:
        fig_zoom.savefig(zoom_png, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Zoom panel saved: {zoom_png}")
    except Exception as e:
        if verbose:
            print(f"Warning: could not save zoom PNG: {e}")

    plt.show()

# --- Progress helpers ---------------------------------------------------------

def _safe_call_progress(progress_callback, percent: int):
    if progress_callback is None:
        return
    try:
        progress_callback(int(percent))
    except Exception:
        # Never fail analysis because UI callback failed
        return

# --- Main API -----------------------------------------------------------------

def run_analysis(infile: str,
                 outfile: str,
                 calibration_file: str = "config/calibration_nearly_perfect.csv",
                 progress_callback=None,
                 verbose: bool = True) -> str:
    """
    Run the full analysis pipeline.

    Parameters
    ----------
    infile : str
        Path to the input sky-map CSV (with columns 'ra_rad' and 'dec_rad').
    outfile : str
        Path to the output CSV that will contain the scrambled events.
    calibration_file : str
        Path to the calibration CSV.
    progress_callback : callable | None
        Function that accepts an integer percent (0..100). Used by Voilà UI.
        Progress reflects the main scrambling loop over the input events.
    verbose : bool
        If False, suppresses print() outputs (recommended for Voilà).

    Returns
    -------
    str
        The path to the output CSV (outfile).
    """

    if verbose:
        print(f"Reading CSV input: {infile}")

    ra_vals, dec_vals = read_ra_dec_csv_robust(infile)

    az_list = []
    zen_list = []
    gal_coords = []

    out_f = open(outfile, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow(["azimuth", "zenith", "ra_gal", "dec_gal"])

    config_path = "config/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = int(config.get("global_seed"))
    offsets_file = "config/offsets.csv"

    systematics = get_systematics(seed, offsets_file, calibration_file)
    if verbose:
        print(systematics)
    radius = systematics["radius"]
    calseed = systematics["seed"]
    if verbose:
        radius_deg = math.degrees(radius)
        print(f"AVG radius: {radius_deg:.3f}°; CALSEED: {calseed}")
        print("Scrambling events...")

    n_total = int(len(ra_vals))
    _safe_call_progress(progress_callback, 0)
    last_percent = -1

    for idx, (ra, dec) in enumerate(zip(ra_vals, dec_vals), start=1):
        ra_new, dec_new = scramble_point(ra, dec, radius, calseed, apply_systematics)
        az = ra_new
        zen = math.pi / 2 - dec_new
        gal_lon, gal_lat = local_to_galactic(az, zen)

        writer.writerow([az, zen, gal_lon, gal_lat])
        az_list.append(az)
        zen_list.append(zen)
        gal_coords.append((gal_lon, gal_lat))

        # progress update (integer percent, throttled)
        if n_total > 0:
            percent = int((idx * 100) / n_total)
            if percent != last_percent:
                last_percent = percent
                _safe_call_progress(progress_callback, percent)

    _safe_call_progress(progress_callback, 100)

    extra_points_hardcoded = [
        (0.6201256910664598, 1.9847895668488024),
        (1.2495508304639656, 2.3803453248156247),
        (3.6512412533630015, 2.6106357722471365),
        (5.4421659242800465, 0.8360177966839565)
    ]
    neutrinos_per_source = [5, 2, 0, 2]

    extra_sources, extra_smeared = [], []
    sigma_rad = math.radians(0.1)

    for i, ((az_hc, zen_hc), n_nu) in enumerate(zip(extra_points_hardcoded, neutrinos_per_source)):
        gal_lon_src, gal_lat_src = local_to_galactic(az_hc, zen_hc)
        writer.writerow([az_hc, zen_hc, gal_lon_src, gal_lat_src])
        extra_sources.append((az_hc, zen_hc, gal_lon_src, gal_lat_src))

        for j in range(n_nu):
            inj_seed = (calseed + (i << 16) + j) & 0xFFFFFFFFFFFFFFFF
            rng = _rng_from_seed(inj_seed)

            ra_loc, dec_loc = az_hc, math.pi / 2 - zen_hc
            ra_gauss, dec_gauss = gaussian_smear_tangent(ra_loc, dec_loc, sigma_rad, rng=rng)

            ra_final, dec_final = scramble_point(ra_gauss, dec_gauss, radius, calseed, apply_systematics)
            az_final, zen_final = ra_final, math.pi / 2 - dec_final
            gal_lon_ev, gal_lat_ev = local_to_galactic(az_final, zen_final)

            writer.writerow([az_final, zen_final, gal_lon_ev, gal_lat_ev])
            extra_smeared.append((az_final, zen_final, gal_lon_ev, gal_lat_ev))

            az_list.append(az_final)
            zen_list.append(zen_final)
            gal_coords.append((gal_lon_ev, gal_lat_ev))

    out_f.close()
    if verbose:
        print(f"Scrambled CSV written: {outfile}")

    plot_aitoff_and_zoom(
        np.array(az_list), np.array(zen_list), gal_coords,
        extra_sources=extra_sources,
        extra_smeared=extra_smeared,
        outfile_base=outfile.replace(".csv", ""),
        verbose=verbose
    )

    return outfile

def main():
    """
    CLI entry point.

    Usage:
        python analysis.py input.csv output.csv
        python analysis.py input.csv output.csv calibration.csv
    """
    if len(sys.argv) not in (3, 4):
        print(
            "Usage:\n"
            "  python analysis.py input.csv output.csv\n"
            "  python analysis.py input.csv output.csv calibration.csv"
        )
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]

    if len(sys.argv) == 4:
        calibration_file = sys.argv[3]
    else:
        calibration_file = "config/calibration_nearly_perfect.csv"

    # CLI should remain verbose
    run_analysis(infile, outfile, calibration_file, progress_callback=None, verbose=True)

if __name__ == "__main__":
    main()
