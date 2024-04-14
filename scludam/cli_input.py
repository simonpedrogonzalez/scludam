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

class DataInput:
    def __init__(self, df, file_path, query):
        self.df = df
        self.file_path = file_path
        self.query = query

    def df_str(self):
        cols = self.df.columns
        shape = self.df.shape
        return f"Columns: {cols}\nShape: {shape}"
    
    def __str__(self):
        return "Data:" + \
            f"\nFile path: {self.file_path}" + \
            f"\n{self.df_str()}"


def select_input_type():
    selected = prompt_cli_selector(
        "Select the input type",
        ['Local file', 'Gaia catalog'],
        ["file", "catalog"], 
        default_index=0)
    return selected

def select_columns():
    default = [
        "source_id", "l", "b",
        "ra", "dec", "ra_error", "dec_error", "ra_dec_corr",
        "pmra", "pmra_error", "ra_pmra_corr", "dec_pmra_corr",
        "pmdec", "pmdec_error", "ra_pmdec_corr", "dec_pmdec_corr", "pmra_pmdec_corr",
        "parallax", "parallax_error", "parallax_pmra_corr", "parallax_pmdec_corr",
        "ra_parallax_corr", "dec_parallax_corr", "parallax_over_error",
        "phot_g_mean_mag","bp_rp",'astrometric_excess_noise'
        ]
    new_columns = ["ra", "dec", "parallax","parallax_error","pmra","pmra_error","pmdec","pmdec_error","phot_g_mean_mag","bp_rp","radial_velocity","radial_velocity_error","l","b","mh_gspphot","phot_g_mean_flux","phot_g_mean_flux_error","phot_bp_mean_flux","phot_bp_mean_flux_error","phot_rp_mean_flux","phot_rp_mean_flux_error"]
    
    final_colums = list(set(default + new_columns))
    print("Default columns are:")
    print(final_colums)
    # todo ask for more cols
    return final_colums

def select_location():
    selected = prompt_cli_selector(
        "Seach by Simbad name or coordinates?",
        ['Name', 'Coordinates'],
        ["name", "coordinates"])
    if selected == "name":
        name = prompt_cli_string_input("Enter the Simbad name of the object:")
        simbad_result = search_object(name)
        if simbad_result.table is not None:
            ra = simbad_result.coords.ra.deg
            dec = simbad_result.coords.dec.deg
            print(f"Found in RA: {ra}, DEC: {dec}")
        else:
            print("Object not found. Try again.")
            return select_location()
        return name
    else:
        ra = float(input("Enter the RA of the object in deg (e.g. 121.24):"))
        dec = float(input("Enter the DEC of the object in deg (e.g. -28.14):"))
        return (ra, dec)

def select_radius():
    radius = prompt_cli_float_input("Enter the search radius in deg (e.g. 0.5):")
    return radius   

def select_catalog():
    selected = prompt_cli_selector(
        "Select the catalog to search",
        ['Gaia DR3', 'Gaia DR2'],
        ["gaiadr3.gaia_source", "gaiadr2.gaia_source"],
        default_index=0
        )
    return selected

def select_criteria():
    default = [
        ("Paralax search condition. Format is operator, whitespace and value (e.g.'> 0.2').", "parallax"),
        ("G magnitude search condition. Format is operator, whitespace and value (e.g.'< 18').", "phot_g_mean_mag"),
    ]
    # todo add some checking
    print("Please define the search criteria.")
    conditions = []
    for i in range(len(default)):
        cond = input(default[i][0] + ":\n")
        new_cond = (default[i][1], cond.split()[0], cond.split()[1])
        conditions.append(new_cond)
    return conditions

def select_output_format():
    selected = prompt_cli_selector(
        "Select the output file format:",
        ['FITS', 'CSV'],
        ["fits", "csv"],
        default_index=0
        )
    return selected


def select_output_file_name(location):
    if type(location) == str:
        name = iden2filename(location)
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name = f"catalog_download_{timestamp}"
    name2 = input(f"Name of the output file (press enter for '{name}'):")
    if name2 == "":
        return name
    return name2

def dowload_from_catalog():
    location = select_location()
    radius = select_radius()
    catalog = select_catalog()
    criteria = select_criteria()
    columns = select_columns()
    query = Query().select(*columns).where_in_circle(location, radius).where(criteria)
    query.table = catalog
    # Descargar los datos
    data = query.get()
    nrows = len(data)

    # Borrar las filas con datos faltantes
    # print(f"Data downloaded. No of rows: {nrows}")
    # print("Warning: dropping rows with missing data")
    # data = Table.from_pandas(data.to_pandas().dropna())
    # print(f"Dropped: {nrows - len(data)} rows")

    # Guardar los datos
    output_format = select_output_format()
    output_file = select_output_file_name(location)

    file_path = f"{output_file}.{output_format}"
    if output_format == "csv":
        data.write(file_path, format="csv", overwrite=True)
    elif output_format == "fits":
        data.write(file_path, format="fits", overwrite=True)
    
    # write query.build() to a txt file
    with open(f"{output_file}_query.txt", "w") as f:
        f.write(query.build())

    print(f"Data downloaded in {file_path}")
    print(f"Query saved in {output_file}_query.txt")
    di = DataInput(data.to_pandas(), file_path, query)
    return di

def get_from_file():
    # get current dir
    import os

    currentdir = os.getcwd()
    print("Current directory is: " + currentdir)

    file_path = prompt_cli_string_input("Enter the file path (with extension):")
    data = Table.read(file_path)
    di = DataInput(data.to_pandas(), file_path, None)
    return di

def select_input():
    input_type = select_input_type()
    if input_type == "file":
        di = get_from_file()
    else:
        di = dowload_from_catalog()
    return di
    
