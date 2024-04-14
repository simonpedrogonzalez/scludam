# scludam, Star CLUster Detection And Membership estimation package
# Copyright (C) 2022  Simón Pedro González

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Cli utils module.

For cli utils functions.
"""

import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from scludam import Query, search_object, search_table
from scludam import CountPeakDetector, DEP, DBME, RipleysKTest, HopkinsTest, HKDE, SHDBSCAN, PluginSelector, RuleOfThumbSelector
from astropy.table.table import Table
import math


# add lines to mark a point in a graph
def mark_point(x,y):
    plt.axhline(y=y, color="black", linestyle="dashed", linewidth=.5)
    plt.axvline(x=x, color="black", linestyle="dashed", linewidth=.5)
    return

# extract star source_id
def get_star_dr3_source_id_from_simbad_data(simbad_star_data):
  pattern = r"\|Gaia DR3 ([A-Za-z0-9]+)(?:\||$)"
  string = str(simbad_star_data.table['IDS'])
  match = re.search(pattern, string)
  if match:
    code = match.group(1)
    return int(code)
  else:
    print("No match found.")

# create file name from cluster identifier
def iden2filename(iden):
    return iden.replace(" ", "_").lower()

# get best bin shape for maximum sum of scores in detection
def best_bin_shape(
    data,
    pm=[.3, .4, .5, .6, .7, .8, .9, 1],
    plx=[.04, .05, .06, .07, .08, .09, .1, .11],
    min_score=1,
    max_n_peaks=25,
    **kwargs,
    ):
    # create arange of possible bin shapes
    pm = np.array(pm)
    plx = np.array(plx)
    # create all combinations of bin shapes
    bin_shapes = np.array(np.meshgrid(pm, plx)).T.reshape(-1, 2)
    # replicate pmra bin_shape for pmdec
    bin_shapes = np.repeat(bin_shapes, 2, axis=1)[:,:-1].tolist()

    last_scoresum = 0
    last_bin_shape = bin_shapes[0]
    last_result = None
    scores = []
    # iterate and find the one that is better
    for j, bin_shape in enumerate(bin_shapes):
        try:
            detector = CountPeakDetector(
                bin_shape=bin_shape,
                min_score=min_score,
                max_n_peaks=max_n_peaks,
                **kwargs,
            )
            detector.detect(data)
            if (detector._last_result.counts.size != 0):
                current_scoresum = detector._last_result.scores.sum()
            else:
                current_scoresum = 0
        except:
            # Exception: no bin passed min_count density check
            current_scoresum = 0
        scores.append(current_scoresum)
        if current_scoresum > last_scoresum:
            last_scoresum = current_scoresum
            last_bin_shape = bin_shape
            last_result = detector
        # print(f"bin_shape: {bin_shape}, scoresum: {current_scoresum}")
        # print(f"last_scoresum: {last_scoresum}, last_bin_shape: {last_bin_shape}")
    if last_scoresum == 0:
        raise Exception("Check input data for bin selection.")
    dfbins = pd.DataFrame({"pmra": np.array(bin_shapes)[:,0], "pmdec":np.array(bin_shapes)[:,1], "parallax": np.array(bin_shapes)[:,2], "score": scores})

    return last_bin_shape, last_result, dfbins

def closest_cluster(det_res, coords):
    all_distances = []
    for i, center in enumerate(det_res.centers):
        # calculate euclidean distance between vectors coords and center
        # we dont know if coords and center are 2,3,... dimensional
        all_distances.append(np.linalg.norm(np.array(coords) - center))
    current_scoresum = np.min(all_distances)
    # index of the closest cluster
    closest_cluster = np.argmin(all_distances)
    current_nstars = det_res.counts[closest_cluster]
    return closest_cluster, current_scoresum, current_nstars

def add_aen_to_variable_errors(df):
  error_names = [
      "ra_error", "dec_error", "pmra_error",
      "pmdec_error", "parallax_error"]
  aen2 = np.square(df["astrometric_excess_noise"].values / 2)
  for en in error_names:
    if en in df.columns:
      df[en] = np.sqrt(np.square(df[en].values) + aen2)
  return df


# get best bin shape for maximum proximity to known cluster
def best_bin_shape_for_known_cluster(data,
    coords,
    pm=[.3, .4, .5, .6, .7, .8, .9, 1],
    plx=[.04, .05, .06, .07, .08, .09, .1, .11],
    min_score=1,
    max_n_peaks=25,
    **kwargs,
    ):
    # create arange of possible bin shapes
    pm = np.array(pm)
    plx = np.array(plx)
    # create all combinations of bin shapes
    bin_shapes = np.array(np.meshgrid(pm, plx)).T.reshape(-1, 2)
    # replicate pmra bin_shape for pmdec
    bin_shapes = np.repeat(bin_shapes, 2, axis=1)[:,:-1].tolist()

    last_scoresum = np.inf
    last_bin_shape = bin_shapes[0]
    last_result = None
    scores = []
    found_index = None
    found_nstars = None
    # iterate and find the one that is better
    for j, bin_shape in enumerate(tqdm(bin_shapes)):
        try:
            detector = CountPeakDetector(
                bin_shape=bin_shape,
                min_score=min_score,
                max_n_peaks=max_n_peaks,
                **kwargs,
            )
            detector.detect(data)
            if (detector._last_result.counts.size != 0):
                # get the cluster centers and compare to the coords
                # take the closest one and save the distance
                # as current_scoresum
                all_distances = []
                for i, center in enumerate(detector._last_result.centers):
                    # calculate euclidean distance between vectors coords and center
                    # we dont know if coords and center are 2,3,... dimensional
                    all_distances.append(np.linalg.norm(np.array(coords) - center))
                current_scoresum = np.min(all_distances)
                # index of the closest cluster
                closest_cluster = np.argmin(all_distances)
                current_nstars = detector._last_result.counts[closest_cluster]
            else:
                current_scoresum = np.inf
        except:
            # Exception: no bin passed min_count density check
            current_scoresum = np.inf
        scores.append(current_scoresum)
        if current_scoresum < last_scoresum:
            last_scoresum = current_scoresum
            last_bin_shape = bin_shape
            last_result = detector
            found_index = closest_cluster
            found_nstars = current_nstars
            #print(f"\nnew best bin_shape: {bin_shape}, distance to target: {current_scoresum}")
            #print(f"number of stars in cluster: {found_nstars}")
            #print(f"index: {found_index}")
        #tqdm.set_postfix_str(f"progress: {j/len(bin_shapes)*100:.2f}")
        #tqdm.update(1)
    if last_scoresum == np.inf:
        raise Exception("Check input data for bin selection.")
    dfbins = pd.DataFrame({"pmra": np.array(bin_shapes)[:,0], "pmdec":np.array(bin_shapes)[:,1], "parallax": np.array(bin_shapes)[:,2], "score": scores})

    return last_bin_shape, last_result, dfbins, found_index, found_nstars


def calculate_antonio(df):
    df['e_G']=df['phot_g_mean_flux_error'] / df['phot_g_mean_flux'] * 2.5 * (1/math.log(10))
    df['e_BP-RP']=2.5 * (1/math.log(10)) * np.sqrt((df['phot_bp_mean_flux_error']/df['phot_bp_mean_flux'])**2 + (df['phot_rp_mean_flux_error']/df['phot_rp_mean_flux'])**2)
    df['Fe/H']=df['phot_g_mean_flux']
    return df

def change_column_names_antonio(df):
    headerList = ['Ra_J2000', 'Dec_J2000', 'Plx_mas','e_plx','pm_RA','e_pmRA','pm_DEC','e_pmDEC','G_mag','BP_RP_mag','RV','e_RV','l','b','astrometric_excess_noise','astrometric_excess_noise_sig','Fe/H','e_G','e_BP-RP'] # agregar col de probabilidad de pertenencia
    name_mapping = {
        'ra': 'Ra_J2000',
        'dec': 'Dec_J2000',
        'parallax': 'Plx_mas',
        'parallax_error': 'e_plx',
        'pmra': 'pm_RA',
        'pmra_error': 'e_pmRA',
        'pmdec': 'pm_DEC',
        'pmdec_error': 'e_pmDEC',
        'phot_g_mean_mag': 'G_mag',
        'bp_rp': 'BP_RP_mag',
        'radial_velocity': 'RV',
        'radial_velocity_error': 'e_RV',
        'l': 'l',
        'b': 'b',
        'astrometric_excess_noise': 'astrometric_excess_noise',
        'astrometric_excess_noise_sig': 'astrometric_excess_noise_sig',
        'mh_gspphot': 'Fe/H',
        'phot_g_mean_flux': 'e_G',
        'phot_g_mean_flux_error': 'e_BP-RP',
        'phot_bp_mean_flux': 'Unused_1',
        'phot_bp_mean_flux_error': 'Unused_2',
        'phot_rp_mean_flux': 'Unused_3',
        'phot_rp_mean_flux_error': 'Unused_4'
    }
    # todo
    return df

def prompt_cli_int_input(prompt, default=None):
    full_prompt = prompt + f"{' (default: ' + str(default) + ')' if default is not None else ''}\n>"
    value = input(full_prompt)
    # validate
    try:
        return int(value)
    except:
        if default is not None and value == "":
            return default
        print("Invalid input, try again.\n")
        return prompt_cli_int_input(prompt)

def prompt_cli_float_input(prompt, default=None):
    full_prompt = prompt + f"{' (default: ' + str(default) + ')' if default is not None else ''}\n>"
    value = input(full_prompt)
    # validate
    try:
        return float(value)
    except:
        if default is not None and value == "":
            return default
        print("Invalid input, try again.\n")
        return prompt_cli_float_input(prompt)

def prompt_cli_string_input(prompt):
    value = input(prompt + "\n> ")
    return value


def promp_selector_with_custom_option_and_default(
        prompt, options, values, custom_option_prompt, default,
):
    options = options + ["Custom"]
    values = values + [None]

    selected = prompt_cli_selector(prompt, options, values)

    if selected is None:
        custom_value = prompt_cli_string_input(custom_option_prompt)
        return custom_value
    else:
        return selected
    

def prompt_cli_selector(
        prompt,
        options,
        values,
        default_index=None,
        custom_prompt=None,
    ):

    if custom_prompt is not None:
        options = options + ["Custom"]
        values = values + [None]

    print(prompt + "\n")
    for i, option in enumerate(options):
        print(f"{i+1}. {option}{'' if i != default_index else ' (default)'}")
    selection = input("> ")

    if default_index is not None and selection == "":
        if custom_prompt is not None and default_index == len(options)-1:
            custom_value = prompt_cli_string_input(custom_prompt)
            return custom_value
        else:
            return values[default_index]
    
    # validate
    if selection.isdigit():
        selection = int(selection)

        if custom_prompt is not None and selection == len(options):
            custom_value = prompt_cli_string_input(custom_prompt)
            return custom_value
        
        if selection > 0 and selection <= len(options):
            return values[selection-1]
        else:
            print("Invalid selection, try again.\n")
            return prompt_cli_selector(prompt, options, values, default_index, custom_prompt)
    else:
        print("Invalid selection, try again.\n")
        return prompt_cli_selector(prompt, options, values, default_index, custom_prompt)