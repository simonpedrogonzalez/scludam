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

"""Module for remote catalog data fetching.

The module provides functions for searching objects and tables, and to
fetch the data from the remote catalog. Currently the data fetching
supports the GAIA catalogues. Object searching is done using the
Simbad service.

Examples
--------
.. literalinclude:: ../../examples/fetcher/get_data_from_gaia.py
    :language: python
    :linenos:

Options for each function and class in the example are described
in the documentation below.

"""

from numbers import Number
from typing import List, Union

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

from scludam.type_utils import Condition, Coord, LogicalExpression


@define
class Config:
    """Class to hold defaults for a query."""

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
    """Class to hold the result of a search_object query.

    Attributes
    ----------
    table : astropy.table.Table
        The table with the results of the query.
    coords : astropy.coordinates.SkyCoord
        The coordinates of the object in ICRS system.

    """

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
        Simbad identifier.
    cols : List[str], optional
        Columns to be included in the result, by default
        [ "coordinates", "parallax", "propermotions",
        "velocity", "dimensions", "diameter", ]

    Returns
    -------
    SimbadResult
        Object with the search results

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
    """Class to hold the result of a search_table query.

    Attributes
    ----------
    name : str
        Name of the table.
    columns : astropy.table.Table
        Table with the information of the columns of the table.
    description : str
        Description of the table.

    """

    name: str
    description: str
    columns: Table


@beartype
def search_table(search_query: str = None, only_names: bool = False, **kwargs):
    """List available tables in gaia catalogue matching a search query.

    Parameters
    ----------
    search_query : str, optional
        query to match, by default ``None``
    only_names : bool, optional
        return only table names and descriptions, by default ``False``

    Returns
    -------
    List[TableInfo]
        List of tables found.

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
    """Class to hold an ADQL query to be executed.

    Attributes
    ----------
    table : str
        Name of the table to be queried, by default given by Config.MAIN_GAIA_TABLE
    row_limit: int
        Maximum number of rows to be returned, by default given by Config.ROW_LIMIT
    conditions: List[LogicalExpression]
        List of conditions to be applied to the query, by default []
    columns: List[str]
        List of columns to be returned, by default [], meaning all.
    extra_columns: List[str]
        List of extra columns to be included in the query given by custom
        conditions, by default []
    orderby: str
        Column to be used for ordering.

    Notes
    -----
    It is recommended to not manually set the attributes of this class,
    except for table: str.

    """

    QUERY_TEMPLATE = (
        """SELECT {row_limit} {columns} \nFROM {table_name} {conditions} {orderby}"""
    )
    COUNT_QUERY_TEMPLATE = """SELECT COUNT(*) FROM {table_name} {conditions}"""
    table: str = config.MAIN_GAIA_TABLE
    row_limit: int = config.ROW_LIMIT
    columns: list = Factory(list)
    extra_columns: list = Factory(list)
    conditions: List[LogicalExpression] = Factory(list)
    orderby: str = None

    @beartype
    def select(self, *args: str):
        """Add columns to query.

        Parameters
        ----------
        *args: str
            Columns to be included in the query.

        Returns
        -------
        Query
            instance of query.

        """
        if "*" not in args:
            self.columns = list(args)
        return self

    @beartype
    def top(self, row_limit: int):
        """Set the number of rows to be returned.

        Parameters
        ----------
        row_limit : int
            number of rows to be returned.

        Returns
        -------
        Query
            instance of query.

        """
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
            raise ValueError(f"Invalid operator: {operator}")

    @beartype
    def where(self, condition: Union[Condition, List[Condition]]):
        """Add a condition to the query.

        Parameters
        ----------
        condition : Union[Condition, List[Condition]]
            Condition or list of Conditions to be added to the query.
            Each Condition is a tuple of the form
            (expression1, operator, expression2): (str, str, Union[str, Number])

        Returns
        -------
        Query
            instance of query.

        """
        if isinstance(condition, list):
            for i, cond in enumerate(condition):
                column, operator, value = cond
                self._validate_column(column)
                self._validate_operator(operator)
                self.conditions.append(("AND", column, operator, value))
        else:
            column, operator, value = condition
            self._validate_column(column)
            self._validate_operator(operator)
            self.conditions.append(("AND", column, operator, value))
        return self

    @beartype
    def where_or(self, condition: Union[Condition, List[Condition]]):
        """Add conditions to the query following in CNF.

        CNF is conjunctive normal form: AND (c1 OR c2 OR ...).

        Parameters
        ----------
        condition : Union[Condition, List[Condition]]
            Condition or list of Conditions to be added to the query.

        Returns
        -------
        Query
            instance of query.

        """
        if isinstance(condition, list):
            first = condition.pop(0)
        else:
            first = condition
        column, operator, value = first
        self._validate_column(column)
        self._validate_operator(operator)
        self.conditions.append(("AND (", column, operator, value))

        if isinstance(condition, list):
            for i, cond in enumerate(condition):
                column, operator, value = cond
                self._validate_column(column)
                self._validate_operator(operator)
                self.conditions.append(("OR", column, operator, value))

        last = self.conditions.pop(-1)
        new_last = (last[0], last[1], last[2], str(last[3]) + ")")
        self.conditions.append(new_last)
        return self

    @beartype
    def where_in_circle(
        self,
        coords_or_name: Union[Coord, SkyCoord, str],
        radius: Union[int, float, Quantity],
        ra_name: str = config.MAIN_GAIA_RA,
        dec_name: str = config.MAIN_GAIA_DEC,
    ):
        """Add a condition to the query to select objects within a circle.

        The circle is drawn in the spherical coordinates space (ra, dec). It also adds
        the dist (distance from the center) column to column list and adds orderby
        distance to the query.

        Parameters
        ----------
        coords_or_name : Union[Coord, SkyCoord, str]
            Coordinates of the center of the circle or name of the identifier to be
            searched using search_object.
        radius : Union[int, float, astropy.units.quantity.Quantity]
            value of the radius of the circle. If int or float, its taken as degrees.
        ra_name : str, optional
            ra column name, by default config.MAIN_GAIA_RA
        dec_name : str, optional
            dec column name, by default config.MAIN_GAIA_DEC

        Returns
        -------
        Query
            instance of query.

        """
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
    def where_aen_criterion(
        self,
        aen_value: Number = 2,
        aen_sig_value: Number = 2,
        aen_name: str = config.MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE,
        aen_sig_name: str = config.MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE_SIG,
    ):
        """Add astrometric excess noise rejection criterion based on GAIA criteria.

        It also adds the aen and aen_sig columns to column list.

        Parameters
        ----------
        aen_value : Number, optional
            astrometric excess noise threshold value, by default 2
        aen_sig_value : Number, optional
            astrometric excess noise score threshold value, by default 2
        aen_name : str, optional
            column name for astrometric excess noise, by default
            config.MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE
        aen_sig_name : str, optional
            column name for astrometric escess noise score, by default
            config.MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE_SIG

        Returns
        -------
        Query
            instance of query

        Notes
        -----
        The criteria [1]_ used is:
        *  exclude objects if
        *   ``astrometric_excess_noise_score > aen_sig_value AND``
        *   ``astrometric_excess_noise > aen_value``

        References
        ----------
        .. [1] GAIA Team (2021). GAIA EDR3 data model
            . https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html

        """  # noqa E501
        self.extra_columns.append(aen_name)
        self.extra_columns.append(aen_sig_name)
        return self.where_or(
            [
                (aen_sig_name, "<=", aen_sig_value),
                (aen_name, "<", aen_value),
            ]
        )

    @beartype
    def where_arenou_criterion(
        self,
        bp_rp_name: str = config.MAIN_GAIA_BP_RP,
        bp_rp_ef_name: str = config.MAIN_GAIA_BP_RP_EXCESS_FACTOR,
    ):
        """Add rejection criterion based on Arenou et al. (2018).

        It also adds the bp_rp and bp_rp_ef columns to column list.

        Parameters
        ----------
        bp_rp_name : str, optional
            bp_rp column name, by default config.MAIN_GAIA_BP_RP
        bp_rp_ef_name : str, optional
            bp_rp excess factor column name, by default
            config.MAIN_GAIA_BP_RP_EXCESS_FACTOR

        Returns
        -------
        Query
            instance of query

        Notes
        -----
        The criteria [2]_ used is:
        *  include objects if: ``1 + 0.015(BP-RP)^2 < E <1.3 + 0.006(BP-RP)^2``
        *   where ``E`` is photometric BP-RP excess factor.

        References
        ----------
        .. [2] Arenou et al. (2018).  Gaia Data Release 2.
            A&A 616, A17. https://doi.org/10.1051/0004-6361/201833234

        """  # noqa E501
        self.extra_columns.append(bp_rp_name)
        self.extra_columns.append(bp_rp_ef_name)
        return self.where(
            [
                (bp_rp_ef_name, ">", f"1 + 0.015 * POWER({bp_rp_name}, 2)"),
                (bp_rp_ef_name, "<", f"1.3 + 0.006 * POWER({bp_rp_name}, 2)"),
            ]
        )

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
        """Build the count query.

        It allows to preview a query without executing it.

        Returns
        -------
        str
            string query in ADQL.

        """
        query = self.COUNT_QUERY_TEMPLATE.format(
            table_name=self.table, conditions=self._build_conditions()
        )
        return query

    def _build_conditions(self):
        if self.conditions:
            conditions = "".join(
                [
                    f"\n{lop}{expr1 if '(' in lop else ' ' + expr1} {cop} {expr2}"
                    for lop, expr1, cop, expr2 in self.conditions
                ]
            )
            conditions = f"\nWHERE {conditions.replace('AND ', '', 1)}"
        else:
            conditions = ""
        return conditions

    def build(self):
        """Build the query.

        Returns
        -------
        str
            string query in ADQL.

        """
        query = self.QUERY_TEMPLATE.format(
            row_limit=f"TOP {self.row_limit}" if self.row_limit > 0 else "",
            columns=self._build_columns(),
            table_name=self.table,
            conditions=self._build_conditions(),
            orderby=f" \nORDER BY {self.orderby}" if self.orderby else "",
        )
        return query

    def get(self, **kwargs):
        """Execute the query.

        It launches an asynchronous gaia job. It takes some time to execute the
        query and parse the results. Parameters are passed through kwargs to
        astroquery.gaia.Gaia.launch_job_async.

        Parameters
        ----------
        dump_to_file : bool
            If ``True``, results will be stored in file, false by default.
        output_file : str
            Name of the output file if dump_to_file is True.

        Returns
        -------
        astroquery.table.table.Table
            Table with the results if dump_to_file is False.

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
        """Execute the count query.

        It launches an asynchronous gaia job. It takes some time
        to execute the query and parse the results. It only returns
        a table with a single count_all column. Parameters are
        passed through kwargs to astroquery.gaia.Gaia.launch_job_async.

        Parameters
        ----------
        dump_to_file : bool
            If ``True``, results will be stored in file, false by default.
        output_file : str
            Name of the output file if dump_to_file is True.

        Returns
        -------
        astroquery.table.table.Table
            table with the results if dump_to_file is False

        """
        query = self.build_count()
        from astroquery.gaia import Gaia

        print("Launching query")
        print(query)
        print("This may take some time...")
        job = Gaia.launch_job_async(query=query, **kwargs)

        if not kwargs.get("dump_to_file"):
            table = job.get_results()
            return table
