# <Opencluster, a package for open star cluster probabilities calculations>
# Copyright (C) 2020  González Simón Pedro

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

# =============================================================================
# DOCS
# =============================================================================

"""Package for membership probability calculation from remote or local data."""

# =============================================================================
# IMPORTS
# =============================================================================

from numbers import Number
from typing import List, Tuple, Union

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from astropy.units.quantity import Quantity
from astroquery.simbad import Simbad
from astroquery.utils.commons import coord_to_radec, radius_to_unit
from attrs import Factory, define
from beartype import beartype
from ordered_set import OrderedSet

Coord = Tuple[Number, Number]
Condition = Tuple[str, str, Union[str, Number]]
LogicalExpression = Tuple[str, str, str, Union[str, Number]]


@define
class Config:
    MAIN_GAIA_TABLE: str = "gaiaedr3.gaia_source"
    MAIN_GAIA_RA: str = "ra"
    MAIN_GAIA_DEC: str = "dec"
    ROW_LIMIT: int = -1
    MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE: str = "astrometric_excess_noise"
    MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE_SIG: str = "astrometric_excess_noise_sig"
    MAIN_GAIA_BP_RP: str = "bp_rp"
    MAIN_GAIA_BP_RP_EXCESS_FACTOR: str = "phot_bp_rp_excess_factor"


config = Config()


@define
class SimbadResult:
    coords: SkyCoord = None
    table: Table = None


@beartype
def search_object(
    identifier: str,
    cols: List[str] = [
        "coordinates",
        "parallax",
        "propermotions",
        "velocity",
        "dimensions",
        "diameter",
    ],
    **kwargs,
):
    """Search an identifier in Simbad catalogues.

    Parameters
    ----------
    identifier : str
    fields : list of strings optional, default: ['coordinates',
    'parallax','propermotions','velocity']
        Fields to be included in the result.
    dump_to_file: bool optional, default False
    output_file: string, optional.
        Name of the file, default is the object identifier.

    Returns
    -------
    coordinates : astropy.coordinates.SkyCoord
        Coordinates of object if found, None otherwise
    result: votable
        Full result table

    Warns
    ------
    Identifier not found
        If the identifier has not been found in Simbad Catalogues.
    """
    simbad = Simbad()
    simbad.add_votable_fields(*cols)
    table = simbad.query_object(identifier, **kwargs)

    if table is None:
        return SimbadResult()

    try:
        coord = " ".join(np.array(table[["RA", "DEC"]])[0])
    except Exception:
        coord = None

    return SimbadResult(coords=SkyCoord(coord, unit=(u.hourangle, u.deg)), table=table)


@define
class TableInfo:
    name: str
    description: str
    columns: Table


@beartype
def search_table(search_query: str = None, only_names: bool = False, **kwargs):
    """List available tables in Gaia catalogues.

    Parameters
    ----------
    only_names: bool, optional, return only table names as list
        default False

    search_query: str, optional, return only results
        that match pattern.

    Returns
    -------
    tables : list of str if only_names=True
        Available tables names

    tables: vot table if only_names=False
    """
    from astroquery.gaia import Gaia

    available_tables = Gaia.load_tables(only_names=only_names, **kwargs)

    if search_query:
        available_tables = [
            table for table in available_tables if search_query in table.name
        ]

    tables = []
    colnames = [
        "TAP Column name",
        "Description",
        "Unit",
        "Ucd",
        "Utype",
        "DataType",
        "ArraySize",
        "Flag",
    ]
    for table in available_tables:
        name = table.name
        desc = table.description
        if only_names:
            cols = None
        else:
            colvalues = [
                [
                    c.name,
                    c.description,
                    c.unit,
                    c.ucd,
                    c.utype,
                    c.data_type,
                    c.arraysize,
                    c.flag,
                ]
                for c in table.columns
            ]
            df = pd.DataFrame(colvalues)
            df.columns = colnames
            cols = Table.from_pandas(df)
        tables.append(TableInfo(name=name, description=desc, columns=cols))

    return tables


@define
class Query:
    QUERY_TEMPLATE = """SELECT {row_limit} {columns} \nFROM {table_name} {conditions} {orderby}"""
    COUNT_QUERY_TEMPLATE = """SELECT COUNT(*) FROM {table_name} {conditions}"""
    table: str = config.MAIN_GAIA_TABLE
    row_limit: int = config.ROW_LIMIT
    columns: list = Factory(list)
    extra_columns: list = Factory(list)
    conditions: List[LogicalExpression] = Factory(list)
    orderby: str = None

    @beartype
    def select(self, *args: str):
        if "*" not in args:
            self.columns = list(args)
        return self

    @beartype
    def top(self, row_limit: int):
        self.row_limit = row_limit
        return self

    @beartype
    def _validate_column(self, column: str):
        if self.columns:
            if column not in self.columns and column not in self.extra_columns:
                raise KeyError(f"Invalid column name: {column}")

    @beartype
    def _validate_operator(self, operator: str):
        if operator not in ["<", ">", "=", ">=", "<=", "LIKE", "like"]:
            raise ValueError(f"Invalid operator {operator}")

    @beartype
    def where(self, condition: Union[Condition, List[Condition]]):
        if isinstance(condition, list):
            for i, cond in enumerate(condition):
                column, operator, value = cond
                self._validate_column(column)
                self._validate_operator(operator)
                self.conditions.append(
                    ("AND", column, operator, value)
                )
        else:
            column, operator, value = condition
            self._validate_column(column)
            self._validate_operator(operator)
            self.conditions.append(
                ("AND", column, operator, value)
            )
        return self

    @beartype
    def where_or(self, condition: Union[Condition, List[Condition]]):
        if isinstance(condition, list):
            first = condition.pop(0)
        else:
            first = condition
        column, operator, value = first
        self._validate_column(column)
        self._validate_operator(operator)
        self.conditions.append(
            ("AND (", column, operator, value)
        )

        if isinstance(condition, list):
            for i, cond in enumerate(condition):
                column, operator, value = cond
                self._validate_column(column)
                self._validate_operator(operator)
                self.conditions.append(
                    ("OR", column, operator, value)
                )

        last = self.conditions.pop(-1)
        new_last = (last[0], last[1], last[2], str(last[3]) + ")")
        self.conditions.append(new_last)
        return self

    @beartype
    def where_in_circle(
        self,
        coords_or_name: Union[Coord, SkyCoord, str],
        radius: Union[int, float, Quantity],
        ra_name=config.MAIN_GAIA_RA,
        dec_name=config.MAIN_GAIA_DEC,
    ):
        if isinstance(radius, Quantity):
            radius = radius_to_unit(radius, unit="deg")
        if isinstance(coords_or_name, str):
            coords = search_object(coords_or_name).coords
        elif isinstance(coords_or_name, tuple):
            coords = SkyCoord(
                coords_or_name[0],
                coords_or_name[1],
                unit=(u.degree, u.degree),
                frame="icrs",
            )
        else:
            coords = coords_or_name

        ra_hours, dec = coord_to_radec(coords)
        ra = ra_hours * 15.0

        self.conditions.append(
            (
                "AND",
                "1",
                "=",
                f"CONTAINS( POINT('ICRS', {ra_name}, {dec_name}),\nCIRCLE('ICRS', {ra},"
                f" {dec}, {radius}))",
            )
        )
        self.extra_columns.append(ra_name)
        self.extra_columns.append(dec_name)
        self.extra_columns.append(
            f"DISTANCE( POINT('ICRS', {ra_name}, {dec_name}), POINT('ICRS', {ra},"
            f" {dec})) AS dist"
        )
        self.orderby = "dist ASC"
        return self


    @beartype
    def where_aen_criterion(self, aen_value:Number=2, aen_sig_value:Number=2, aen_name:str=config.MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE, aen_sig_name:str=config.MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE_SIG):
        self.extra_columns.append(aen_name)
        self.extra_columns.append(aen_sig_name)
        return self.where_or([
            (aen_sig_name, '<=', aen_sig_value),
            (aen_name, '<', aen_value),
        ])


    @beartype
    def where_arenou_criterion(self, bp_rp_name:str=config.MAIN_GAIA_BP_RP, bp_rp_ef_name:str=config.MAIN_GAIA_BP_RP_EXCESS_FACTOR):
        self.extra_columns.append(bp_rp_name)
        self.extra_columns.append(bp_rp_ef_name)
        return self.where([
            (bp_rp_ef_name, '>', f"1 + 0.015 * POWER({bp_rp_name}, 2)"),
            (bp_rp_ef_name, '<', f"1.3 + 0.006 * POWER({bp_rp_name}, 2)"),
        ])


    def _build_columns(self):
        if not self.columns:
            columns = ["*"]
        else:
            columns = self.columns
        if self.extra_columns:
            columns = list(OrderedSet(columns).union(OrderedSet(self.extra_columns)))
        columns = ", \n".join(columns)
        return columns

    def build_count(self):
        
        query = self.COUNT_QUERY_TEMPLATE.format(
            table_name=self.table, conditions=self._build_conditions()
        )
        return query

    def _build_conditions(self):
        if self.conditions:
            conditions = "".join(
                [
                    f"\n{logical_op}{expr1 if '(' in logical_op else ' ' + expr1} {comp_op} {expr2}"
                    for logical_op, expr1, comp_op, expr2 in self.conditions
                ]
            )
            conditions = f"\nWHERE {conditions.replace('AND ', '', 1)}"
        else:
            conditions = ""
        return conditions

    def build(self):
        
        query = self.QUERY_TEMPLATE.format(
            row_limit=f"TOP {self.row_limit}" if self.row_limit > 0 else "",
            columns=self._build_columns(),
            table_name=self.table,
            conditions=self._build_conditions(),
            orderby=f" \nORDER BY {self.orderby}" if self.orderby else "",
        )
        return query

    def get(self, **kwargs):
        """Build and perform query.

        Parameters
        ----------
        Parameters that are passed through **kwargs to
        astroquery.gaia.Gaia.launch_job_async
        For example:
        dump_to_file : bool
            If True, results will be stored in file
            (default is False).
        output_file : str
            Name of the output file

        Returns
        -------
        octable : opencluster.OCTable
            Instance with query results,
            None if dump_to_file is True
        """
        query = self.build()

        from astroquery.gaia import Gaia

        print("Launching query")
        print(query)
        print("This may take some time...")

        job = Gaia.launch_job_async(query=query, **kwargs)
        if not kwargs.get("dump_to_file"):
            table = job.get_results()
            return table

    def count(self, **kwargs):
        query = self.build_count()
        from astroquery.gaia import Gaia

        print("Launching query")
        print(query)
        print("This may take some time...")
        job = Gaia.launch_job_async(query=query, **kwargs)

        if not kwargs.get("dump_to_file"):
            table = job.get_results()
            return table
