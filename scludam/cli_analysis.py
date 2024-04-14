import datetime
import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from scludam import Query, search_object, search_table
from scludam import CountPeakDetector, DEP, DBME, RipleysKTest, HopkinsTest, HKDE, SHDBSCAN, PluginSelector, RuleOfThumbSelector
from astropy.table.table import Table
from scludam.cli_utils import *
from copy import deepcopy

gaia5params = {
    "label": "Gaia 5 parameters (ra,dec,parallax,pmra,pmdec)",
    "columns": ["ra", "dec", "parallax", "pmra", "pmdec"]
}

gaia3params = {
    "label": "Gaia 3 parameters (pmra,pmdec,parallax)",
    "columns": ["pmra", "pmdec", "parallax"]
}

gaia2params = {
    "label": "Gaia 2 parameters (pmra,pmdec)",
    "columns": ["pmra", "pmdec"]
}

def automatic_bin_shape_selection(df, det_cols, method):

    def select_auto_bin_shape_to_try(var, default):
        options = [f"{default}"]
        values = [default]
        selected = prompt_cli_selector(
            f"Select the bin shape estimation method for {var}:",
            options,
            values,
            default_index=0,
            custom_prompt="Enter custom bandwidth as comma separated numbers (e.g. 0.1,0.2,0.3):",
        )
        if type(selected) == str:
            return selected.split(",").remove(" ")
        else:
            return selected

    pm_bin_shapes_to_try = select_auto_bin_shape_to_try("pm", [.3, .4, .5, .6, .7, .8, .9, 1])
    parallax_bin_shapes_to_try = select_auto_bin_shape_to_try("parallax", [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1])

    min_score = prompt_cli_int_input("Enter the minimum score to consider a peak:", default=1)
    max_n_peaks = prompt_cli_int_input("Enter the maximum number of peaks to consider:", default=25)
    min_count = prompt_cli_int_input("Enter the minimum number of stars to consider a peak:", default=5)

    if method == "max":
        best_shape, det_res, dfbins, cl_index, cl_star_n = best_bin_shape(
            # datos. El ".values" es para pasarlo a numpy array
            df[det_cols].values,
            # todos los tamaños de pm a probar
            pm_bin_shapes_to_try,
            # todos los tamaños de plx a probar
            parallax_bin_shapes_to_try,
            min_score=min_score,
            # cuántos picos de densidad buscar
            max_n_peaks=max_n_peaks,
            min_count=min_count,
        )
        print(f"Best bin shape for max sum score: {best_shape}")
        detector = CountPeakDetector(
            bin_shape=best_shape,
            min_score=min_score,
            max_n_peaks=max_n_peaks,
            min_count=min_count,
        )
        cl_index = "unknown"

    elif method == "known":
        
        def select_know_values():
            def get_values_from_simbad():
                cluster_name = prompt_cli_string_input("Enter the name of the cluster:")
                simbad_result = search_object(cluster_name)
                if simbad_result.table is not None:
                    cl_pmra = simbad_result.table['PMRA'][0]
                    cl_pmdec = simbad_result.table['PMDEC'][0]
                    cl_parallax = simbad_result.table['PLX_VALUE'][0]
                    print(f"Found in PMRA: {cl_pmra}, PMDEC: {cl_pmdec}, PLX: {cl_parallax}")
                    return cl_pmra, cl_pmdec, cl_parallax
                else:
                    print("Object not found. Try again.")
                    return select_know_values()

            selected = prompt_cli_selector(
                "Select the known values of the cluster in pmra, pmdec and parallax:",
                ["Find in Simbad"],
                ["simbad"],
                default_index=0,
                custom_prompt="Enter the values separated by commas (e.g. 0.1,0.2,0.3):",
            )
            if selected == "simbad":
                return get_values_from_simbad()
            else:
                return [float(x) for x in selected.split(",")]

        cl_pmra, cl_pmdec, cl_parallax = select_know_values()
        best_shape, det_res, dfbins, cl_index, cl_star_n = best_bin_shape_for_known_cluster(
            # datos. El ".values" es para pasarlo a numpy array
            df[det_cols].values,
            # ubicación del cúmulo según Simbad
            [cl_pmra, cl_pmdec, cl_parallax],
            # todos los tamaños de pm a probar
            pm_bin_shapes_to_try,
            # todos los tamaños de plx a probar
            parallax_bin_shapes_to_try,
            min_score=min_score,
            # cuántos picos de densidad buscar
            max_n_peaks=max_n_peaks,
            min_count=min_count,
        )
        distance_to_simbad_data = dfbins[(dfbins.pmra == best_shape[0]) & (dfbins.parallax == best_shape[2])].score.values[0]
        print(f"Best bin shape for cluster: {best_shape}")
        print(f"Initial No stars: {int(cl_star_n)}")
        print(f"Cluster index: {cl_index}")
        print(f"Distance to Cluster values: {distance_to_simbad_data}")
    
        detector = CountPeakDetector(
            bin_shape=best_shape,
            min_score=min_score,
            max_n_peaks=max_n_peaks,
            min_count=min_count,
        )

    return detector, cl_index

def configure_detection(df):   
    # columns
    def select_detection_columns():
        selected = prompt_cli_selector(
            "Select the columns to be used for detection:",
            [gaia3params['label'], gaia2params['label']],
            [gaia3params['columns'], gaia2params['columns']],
            default_index=0,
            custom_prompt="Enter columns separated by commas:",
        )
        if type(selected) == str:
            return selected.split(",").remove(" ")
        else:
            return selected
    det_cols = select_detection_columns()

    # parameters
    selected = prompt_cli_selector(
        "Select the bin shape estimation method:",
        [
            "Estimate bin shape to get max sum of scores on all peaks found (only works for 3 Params solution).",
            "Estimate bin shape to get good peak near the expected values (only works for 3 Params solution).",
        ],
        ["max", "known"],
        default_index=2, #default is custom
        custom_prompt="Enter custom bandwidth as comma separated numbers (e.g. 0.5,0.5,0.05):",
    )
    if selected == "max" or selected == "known":
        detector, cl_index = automatic_bin_shape_selection(df, det_cols, selected)
    else:
        bin_shape = [float(x) for x in selected.split(",")]
        min_score = prompt_cli_int_input("Enter the minimum score to consider a peak:", default=1)
        max_n_peaks = prompt_cli_int_input("Enter the maximum number of peaks to consider:", default=25)
        min_count = prompt_cli_int_input("Enter the minimum number of stars to consider a peak:", default=5)
        detector = CountPeakDetector(
            bin_shape=bin_shape,
            min_score=min_score,
            max_n_peaks=max_n_peaks,
            min_count=min_count,
        )
        cl_index = "unknown"
    
    return det_cols, detector, cl_index

def configure_membership():
    def select_membership_columns():
        selected = prompt_cli_selector(
            "Select the columns to be used for membership:",
            [gaia3params['label'], gaia5params['label']],
            [gaia3params['columns'], gaia5params['columns']],
            default_index=1,
            custom_prompt="Enter columns separated by commas:",
        )
        if type(selected) == str:
            return selected.split(",").remove(" ")
        else:
            return selected

    mem_cols = select_membership_columns()

    error_convolution = bool(
        prompt_cli_selector(
            "Convolve star error matrices? (Very slow)",
            ["Yes", "No"],
            [True, False],
            default_index=1,
        )
    )
    estimator = DBME(
        pdf_estimator=HKDE(
           bw=RuleOfThumbSelector(rule='scott'),
           error_convolution=error_convolution,
        ),
    )
    return mem_cols, estimator

def print_results(dep):
    proba_df = dep.proba_df()
    print(f"Detected Clusters: {dep.n_detected}")
    print(f"Estimated Clusters: {dep.n_estimated}")
    n_stars = proba_df[
        proba_df.columns[proba_df.columns.str.startswith('proba')]
    ].sum()
    n_stars_field = n_stars[0]
    n_stars_cl = n_stars[1:]
    print(f"Field stars: {round(n_stars_field)}")
    for i in range(1, len(n_stars_cl)+1):
        n_stars_cl = n_stars[i]
        print(f"Cluster {i} stars: {round(n_stars_cl)}")

def pipeline_run(df, dep):
    dep.fit(df)
    return dep

def plot_results(df, dep):
    
    def select_plot_type():
        selected = prompt_cli_selector(
            "Select the plot type:",
            ["Position", "Proper Motions", "Color-Magnitude Diagram", "G-Parallax"],
            ["pos", "pm", "cmd", "gplx"],
            default_index=1,
        )
        return selected
    
    plot_type = select_plot_type()
    if plot_type == "pos":
        dep.radec_plot()
    elif plot_type == "pm":
        dep.scatterplot(['pmra', 'pmdec'])
    elif plot_type == "cmd":
        dep.cm_diagram()
    elif plot_type == "gplx":
        dep.scatterplot(['phot_g_mean_mag', 'parallax'])

    plt.show()

def print_config(dep):
    print("Detection:")
    print(f"Columns: {dep.det_cols}")
    print(f"Detector: {dep.detector}")
    print(f"Sample sigma factor: {dep.sample_sigma_factor}")
    print("Membership:")
    print(f"Columns: {dep.mem_cols}")
    print(f"Estimator: {dep.estimator}")

def analyze(df, last_result_config=None):
    
    def new_config(df):
        # Detection
        det_cols, detector, cl_index = configure_detection(df)
        # subsample size config
        sample_sigma_factor = prompt_cli_float_input(
            f"Enter the sample sigma factor:\n(how big is the region taken to isolate the cluster\nin comparison with bin_shape {detector.bin_shape}). Example: one and a half the size of the bin will be 1.5.",
            default=1.5)
        # Membership
        mem_cols, estimator = configure_membership()
        dep = DEP(
            det_cols=det_cols,
            detector=detector,
            sample_sigma_factor=sample_sigma_factor,
            mem_cols=mem_cols,
            estimator=estimator,
        )
        return dep, cl_index

    def configure(last_result_config):
        if last_result_config is not None:
            selected = prompt_cli_selector(
                "Use last result configuration?",
                ["Yes", "No"],
                [True, False],
                default_index=0,
            )
            if selected:
                cl_index = last_result_config.detector.select_index
                if cl_index is None:
                    cl_index = "unknown"
                return last_result_config, cl_index
            else:
                return new_config(df)
        else:
            return new_config(df)
        
    def select_index(cl_index):
        options = ["No index"]
        values = ["none"]
        if cl_index != "unknown":
            options.append(f"Automatic index: {cl_index}")
            values.append(cl_index)
        selected = prompt_cli_selector(
            "Select the index to be used for peak selection:",
            options,
            values,
            default_index=0,
            custom_prompt="Enter the index:",
        )
        if selected == "none":
            return None
        else:
            return selected

        # Setup peak selection
        if selected_index == "auto" and cl_index != "unknown":
            print(f"Cluster index: {cl_index}")
            detector.selected_index = cl_index
            detector.max_n_peaks = cl_index + 1
        elif selected_index is not None:
            detector.selected_index = selected_index
            detector.max_n_peaks = selected_index + 1
    
    dep, cl_index = configure(last_result_config)

    selected_index = select_index(cl_index)

    if selected_index is not None:
        dep.detector.select_index = selected_index
        dep.detector.max_n_peaks = selected_index + 1
        
    dep_config_copy = deepcopy(dep)

    # Run
    dep = pipeline_run(df, dep)
    return dep, dep_config_copy

def save_results(dep, di=None):

    # select path
    def select_output_format():
        selected = prompt_cli_selector(
            "Select the output file format:",
            ['FITS', 'CSV'],
            ["fits", "csv"],
            default_index=0,
            )
        return selected

    def select_output_file_name(di=None):
        if di is not None and di.file_path is not None:
            # last part of di.filepath
            name = "result_" + di.file_path.split("/")[-1].split(".")[0]
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            name = f"result_{timestamp}"
        name2 = input(f"Name of the output file (press enter for '{name}'):")
        if name2 == "":
            return name
        return name2
    
    data = dep.proba_df()
    data = Table.from_pandas(data)

    output_format = select_output_format()
    output_file = select_output_file_name(di)
    file_path = f"{output_file}.{output_format}"
    
    if output_format == "csv":
        data.write(file_path, format="csv", overwrite=True)
    elif output_format == "fits":
        data.write(file_path, format="fits", overwrite=True)
    
    print(f"Results saved in {file_path}")

dep = None
config = None

def main(di):
    global dep
    global config
    print("Analysis menu")
    print("----------------------")
    print(di)
    print("----------------------")
    options = ["New analysis", "Back"]
    values = ["new", "exit"]
    if dep is not None:
        print("Last result available:")
        print("Config:")
        print_config(config)
        print("Results:")
        print_results(dep)
        print("----------------------")
        options = ["New analysis", "Plot available results", "Save results", "Back"]
        values = ["new", "plot", "save", "Back"]
    selected = prompt_cli_selector(
        "Options:",
        options,
        values,
    )
    if selected == "exit":
        return
    if selected == "new":
        dep, config = analyze(di.df, config)
        main(di)
    if selected == "plot":
        plot_results(di.df, dep)
        main(di)
    if selected == "save":
        save_results(dep, di)
        main(di)
    return