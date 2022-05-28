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

import inspect
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
from attr import attrib, attrs
from beartype import beartype

Coord = Tuple[Number, Number]
Condition = Tuple[str, str, Union[str, Number]]
LogicalExpression = Tuple[str, str, str, Union[str, Number]]


@attrs(auto_attribs=True)
class Conf:
    MAIN_GAIA_TABLE: str = "gaiaedr3.gaia_source"
    MAIN_GAIA_TABLE_RA: str = "ra"
    MAIN_GAIA_TABLE_DEC: str = "dec"
    ROW_LIMIT: int = -1


# DEPRECATED
def checkargs(function):
    """Check arguments match their annotated type.

    Parameters
    ----------
    function : function

    Returns
    -------
    function result

    Raises
    ------
    TypeError
    If argument does not match annotated type.
    """

    def wrapper(*args, **kwargs):
        argnames = inspect.getfullargspec(function).args
        arguments = dict(kwargs)
        for index, argvalue in enumerate(args):
            if index < len(argnames):
                arguments[argnames[index]] = argvalue
        annotations = function.__annotations__
        for name, value in arguments.items():
            annotation = annotations.get(name)
            if annotation:
                if not isinstance(value, annotation):
                    raise TypeError("{} is not of type {}".format(name, annotation))
        return function(*args, **kwargs)

    return wrapper


@attrs(auto_attribs=True)
class SimbadResult:
    coords: SkyCoord = None
    table: Table = None


@beartype
def simbad_search(
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


@attrs(auto_attribs=True)
class Query:
    """Query class to retrieve data from remote gaia catalogues.

    Attributes
    ----------
    table : str, optional
        Name of the Gaia catalogues table.
        (default is astroquery.gaia.Gaia.MAIN_GAIA_TABLE)
    ra_name : str, optional
        Name of the column of right ascension column in the selected table.
        (default is astroquery.gaia.Gaia.MAIN_GAIA_TABLE_RA)
    dec_name : str, optional
        Name of the declination column in the selected table.
        (default is astroquery.gaia.Gaia.MAIN_GAIA_TABLE_DEC)
    columns : str or list of str
        If str, must be '*', indicating all columns. If list of str,
        must be list of valid column names.
    column_filters : dict, optional
        Dictionary of filters: {'column_name' : 'column_filter'}
        column_name: str
        Valid column in the selected table
        column_filter: str
        Must start with '<', '>', '<=', '>=' and end with a numeric value.
    row_limit : int, optional
        Limit of rows to retrieve from the remote table.
        (default is -1, meaning all found rows)
    radius : u.quantity.Quantity
        Radius of the cone search.
    coords : astropy.coordinates.SkyCoord
        Coordinates of the center of the cone search.
    QUERY_TEMPLATE : multiline str template used for the query
    """

    QUERY_TEMPLATE = """SELECT {row_limit}
{columns},
DISTANCE(
    POINT('ICRS', {ra_column}, {dec_column}),
    POINT('ICRS', {ra}, {dec})
) AS dist
FROM {table_name}
WHERE 1 = CONTAINS(
    POINT('ICRS', {ra_column}, {dec_column}),
    CIRCLE('ICRS', {ra}, {dec}, {radius}))"""

    COUNT_QUERY_TEMPLATE = """SELECT COUNT(*)
FROM {table_name}
WHERE 1 = CONTAINS(
    POINT('ICRS', {ra_column}, {dec_column}),
    CIRCLE('ICRS', {ra}, {dec}, {radius}))"""

    radius: Number
    coords: SkyCoord
    table: str = Conf().MAIN_GAIA_TABLE
    column_filters: List[LogicalExpression] = attrib(factory=list)
    row_limit: int = Conf().ROW_LIMIT
    ra_name: str = Conf().MAIN_GAIA_TABLE_RA
    dec_name: str = Conf().MAIN_GAIA_TABLE_DEC
    columns: str = "*"

    @beartype
    def where(self, condition: Union[Condition, List[Condition]]):
        """Add filter or condition to the query.

        Parameters
        ----------
        column: str
            Valid column in the selected table
        operator: str
            Must be '<', '>', '<=', '>='
        value: str, int, float
            value that defines the condition

        Returns
        -------
        query : Query class instance

        Raises
        ------
        TypeError if an attribute does not match type.
        KeyError if columns is not '*' and filter has invalid column.
        ValueError if operator is invalid.
        """
        if isinstance(condition, list):
            for i, cond in enumerate(condition):
                column, operator, value = cond
                self.validate_column(column)
                self.validate_operator(operator)
                self.column_filters.append(("AND", column, operator, value))
        else:
            column, operator, value = condition
            self.validate_column(column)
            self.validate_operator(operator)
            self.column_filters.append(("AND", column, operator, value))
        return self

    @beartype
    def validate_column(self, column: str):
        if self.columns != "*":
            if column not in self.columns:
                raise KeyError(f"invalid column '{column}'")

    @beartype
    def validate_operator(self, operator: str):
        if operator not in ["<", ">", "=", ">=", "<=", "LIKE", "like"]:
            raise ValueError(f"invalid operator {operator}")

    @beartype
    def or_where(self, condition: Union[Condition, List[Condition]]):
        if isinstance(condition, list):
            first = condition.pop(0)
        else:
            first = condition
        column, operator, value = first
        self.validate_column(column)
        self.validate_operator(operator)
        self.column_filters.append(("AND (", column, operator, value))

        if isinstance(condition, list):
            for i, cond in enumerate(condition):
                column, operator, value = cond
                self.validate_column(column)
                self.validate_operator(operator)
                self.column_filters.append(("OR", column, operator, value))

        last = self.column_filters.pop(-1)
        new_last = (last[0], last[1], last[2], str(last[3]) + " )")
        self.column_filters.append(new_last)
        return self

    @beartype
    def select(self, *args: str):
        """Select table columns to be retrieved.

        Parameters
        ----------
        columns : str or list of str
            If str, must be '*', indicating all columns or one column
            If list of str, must be list of valid column names.

        Returns
        -------
        query : Query class instance

        Raises
        ------
        ValueError if an attribute does not match type.
        """
        self.columns = list(args)
        return self

    @beartype
    def from_table(self, table: str, ra_name: str = None, dec_name: str = None):
        """Select Gaia table.

        Parameters
        ----------
        table : str, optional
            Name of the Gaia catalogues table
            (default is 'gaiaedr3.gaia_source')
        ra_name : str, optional
            Name of the column of right ascension column in the selected table.
            (default is 'ra')
        dec_name : str, optional
            Name of the declination column in the selected table.
            (default is 'dec')

        Returns
        -------
        query : Query class instance

        Raises
        ------
        TypeError if an attribute does not match type.
        """
        if ra_name:
            self.ra_name = ra_name
        if dec_name:
            self.dec_name = dec_name
        self.table = table
        return self

    def build(self):
        """Build and perform query.

        Returns
        -------
        query : string with built query.
        """
        if isinstance(self.columns, list):
            columns = ", ".join(map(str, self.columns))
        else:
            columns = self.columns

        if self.radius is not None and self.coords is not None:
            ra_hours, dec = coord_to_radec(self.coords)
            ra = ra_hours * 15.0

        row_limit = f"\nTOP {self.row_limit}" if self.row_limit > 0 else ""

        query = self.QUERY_TEMPLATE.format(
            row_limit=row_limit,
            columns=columns,
            ra_column=self.ra_name,
            dec_column=self.dec_name,
            ra=ra,
            dec=dec,
            table_name=self.table,
            radius=self.radius,
        )

        if self.column_filters:
            query_filters = "".join(
                [
                    f"\n{logical_op} {column} {comp_op} {value}"
                    for logical_op, column, comp_op, value in self.column_filters
                ]
            )
            query += query_filters

        query += "\nORDER BY dist ASC"
        return query

    @beartype
    def top(self, row_limit: int):
        """Set row limit for the query.

        Attributes
        ----------
        row_limit : int, optional
            Limit of rows to retrieve from the remote table.
            (default is -1, meaning all found rows)

        Returns
        -------
        query : Query class instance

        Raises
        ------
        ValueError if an attribute does not match type.
        """
        self.row_limit = row_limit
        return self

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

    def build_count(self):
        if self.radius is not None and self.coords is not None:
            ra_hours, dec = coord_to_radec(self.coords)
            ra = ra_hours * 15.0

        query = self.COUNT_QUERY_TEMPLATE.format(
            ra_column=self.ra_name,
            dec_column=self.dec_name,
            ra=ra,
            dec=dec,
            table_name=self.table,
            radius=self.radius,
        )

        if self.column_filters:
            query_filters = "".join(
                [
                    f"\n{logical_op} {column} {comp_op} {value}"
                    for logical_op, column, comp_op, value in self.column_filters
                ]
            )
            query += query_filters
        return query

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


@beartype
def query_region(
    coords_or_name: Union[Coord, SkyCoord, str],
    radius: Union[int, float, Quantity],
):
    """Make a cone search type query for retrieving data.

    Parameters
    ----------
    ra : int, float, optional
        Right ascention of the center of the cone search.
    dec : int, float, optional
        Declination of the center of the cone search.
    name : str, optional
        Name of the Simbad identifier to set the center of the cone search.
    radius : astropy.units.quantity.Quantity
        Radius of the cone search.
    coord : astropy.coordinates.SkyCoord, optional
        Coords of center of cone search

    Returns
    -------
    query : Query class instance

    Raises
    ------
    ValueError if an attribute does not match type, or if
    both name and ra & dec are provided.
    """
    if isinstance(radius, Quantity):
        radius = radius_to_unit(radius, unit="deg")
    if isinstance(coords_or_name, str):
        coords = simbad_search(coords_or_name).coords
    elif isinstance(coords_or_name, tuple):
        coords = SkyCoord(
            coords_or_name[0],
            coords_or_name[1],
            unit=(u.degree, u.degree),
            frame="icrs",
        )
    else:
        coords = coords_or_name
    query = Query(radius=radius, coords=coords)
    return query


@attrs(auto_attribs=True)
class TableInfo:
    name: str
    description: str
    columns: Table


@beartype
def table_info(search_query: str = None, only_names: bool = False, **kwargs):
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
