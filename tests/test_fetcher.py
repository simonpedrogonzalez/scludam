"""Test data fetcher module"""

import re
from itertools import chain

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from astroquery.utils.tap.model.job import Job
from astroquery.utils.tap.model.tapcolumn import TapColumn
from astroquery.utils.tap.model.taptable import TapTableMeta

from scludam import query_region, simbad_search, table_info
from scludam.fetcher import SimbadResult


# TODO: change for utils test_if_raises_exception
class Ok:
    pass


def verify_result(test, func):
    if issubclass(test, Exception):
        with pytest.raises(test):
            func()
    else:
        func()


def simbad_query_object():
    return Table.read("tests/files/simbad_response.xml", format="votable")


def gaia_load_tables():
    c1 = TapColumn(None)
    c1.name = "ra"
    c1.description = "Right ascension"
    c1.unit = "deg"
    c1.ucd = "pos.eq.ra;meta.main"
    c1.utype = "Char.SpatialAxis.Coverage.Location.Coord.Position2D.Value2.C1"
    c1.datatype = "double"
    c1.data_type = None
    c1.flag = "primary"
    t1 = TapTableMeta()
    t1.name = "gaiaedr3.gaia_source"
    t1.description = (
        "This table has an entry for every Gaia observed source as listed in the\nMain"
        " Database accumulating catalogue version from which the catalogue\nrelease has"
        " been generated. It contains the basic source parameters,\nthat is only final"
        " data (no epoch data) and no spectra (neither final\nnor epoch)."
    )
    t1.columns = [c1]
    t2 = TapTableMeta()
    t2.name = "other.name"
    return [t1, t2]


def simbad_search_mock():
    coords = SkyCoord(
        "08 42 31.0", "-48 06 00", unit=(u.hourangle, u.deg), frame="icrs"
    )
    return SimbadResult(coords=coords)


def gaia_launch_job_async():
    return Job(None)


def gaia_job_get_results():
    return Table.read("tests/files/gaia_response.xml", format="votable")


def remove_last_digits(string, number, precision):
    nstring = str(round(number, precision))
    if nstring in string:
        return re.sub(r"(?<=" + nstring + r")\d*", "", string=string)
    return string


def multiline2singleline(string: str):
    return " ".join(
        list(
            filter(
                len,
                list(chain(*[line.split(" ") for line in string.splitlines()])),
            )
        )
    )


# remove unimportant differences between queries"
# 1. very last digit difference in coordinates due to
# precision in calculation.
# 2. linejumps and several spaces that do not affect
# query semantics
def format_query_string(string):
    sra = -48.1000
    sdec = 130.6291
    prec = 4
    string = remove_last_digits(string, sra, prec)
    string = remove_last_digits(string, sdec, prec)
    string = multiline2singleline(string)
    return string


@pytest.fixture
def mock_simbad_query_object(mocker):
    mocker.patch(
        "astroquery.simbad.Simbad.query_object",
        return_value=simbad_query_object(),
    )


@pytest.fixture
def mock_gaia_load_tables(mocker):
    return mocker.patch(
        "astroquery.gaia.Gaia.load_tables", return_value=gaia_load_tables()
    )


@pytest.fixture
def mock_simbad_search(mocker):
    return mocker.patch(
        "scludam.fetcher.simbad_search", return_value=simbad_search_mock()
    )


@pytest.fixture
def mock_gaia_launch_job_async(mocker):
    mocker.patch(
        "astroquery.utils.tap.model.job.Job.get_results",
        return_value=gaia_job_get_results(),
    )
    return mocker.patch(
        "astroquery.gaia.Gaia.launch_job_async",
        return_value=gaia_launch_job_async(),
    )


class TestFetcher:
    def test_simbad_search(self, mock_simbad_query_object):
        result = simbad_search("ic2395", cols=["coordinates", "parallax"])
        assert result.coords.to_string("hmsdms", precision=2) == SkyCoord(
            ra=130.62916667, dec=-48.1, frame="icrs", unit="deg"
        ).to_string("hmsdms", precision=2)
        assert isinstance(result.table, Table)
        assert sorted(list(result.table.columns)) == sorted(
            [
                "MAIN_ID",
                "RA",
                "DEC",
                "RA_PREC",
                "DEC_PREC",
                "COO_ERR_MAJA",
                "COO_ERR_MINA",
                "COO_ERR_ANGLE",
                "COO_QUAL",
                "COO_WAVELENGTH",
                "COO_BIBCODE",
                "RA_2",
                "DEC_2",
                "RA_PREC_2",
                "DEC_PREC_2",
                "COO_ERR_MAJA_2",
                "COO_ERR_MINA_2",
                "COO_ERR_ANGLE_2",
                "COO_QUAL_2",
                "COO_WAVELENGTH_2",
                "COO_BIBCODE_2",
                "PLX_VALUE",
                "PLX_PREC",
                "PLX_ERROR",
                "PLX_QUAL",
                "PLX_BIBCODE",
                "SCRIPT_NUMBER_ID",
            ]
        )
        empty_result = simbad_search("invalid_identifier")
        assert empty_result.table is None and empty_result.coords is None

    def test_table_info(self, mock_gaia_load_tables):
        result = table_info("gaiaedr3", only_names=True)
        assert len(result) == 1
        assert result[0].name == "gaiaedr3.gaia_source"
        assert isinstance(result[0].description, str)
        assert result[0].columns is None
        result = table_info("gaiaedr3")
        assert isinstance(result[0].columns, Table)

    def test_query_region(self, mock_simbad_search):
        name = "dummy_name"
        coords = simbad_search_mock().coords
        ra = coords.ra.value
        dec = coords.dec.value

        correct = (
            "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1)"
            " ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra,"
            " dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) ORDER BY dist ASC"
        )

        queries = [
            query_region(name, 0.5),
            query_region(coords, 0.5),
            query_region((ra, dec), 0.5),
            query_region((ra, dec), 30 * u.arcmin),
            query_region((ra, dec), 0.5 * u.deg),
        ]

        for q in queries:
            q = format_query_string(q.build())
            assert correct == q

    def test_select(self, mock_simbad_search):
        correct = (
            "SELECT parallax, pmra, pmdec, DISTANCE( POINT('ICRS', ra, dec),"
            " POINT('ICRS', 130.6291, -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE"
            " 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1,"
            " 0.5)) ORDER BY dist ASC"
        )
        q = query_region("dummy_name", 0.5).select("parallax", "pmra", "pmdec")
        assert format_query_string(q.build()) == correct

    def test_from_table(self, mock_simbad_search):
        correct = (
            "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1)"
            " ) AS dist FROM table WHERE 1 = CONTAINS( POINT('ICRS', ra, dec),"
            " CIRCLE('ICRS', 130.6291, -48.1, 0.5)) ORDER BY dist ASC"
        )
        q = query_region("dummy_name", 0.5).from_table("table")
        assert format_query_string(q.build()) == correct

    @pytest.mark.parametrize(
        "column, operator, value, test, correct",
        [
            ("parallax", ">>", 3, ValueError, None),
            (
                "string_column",
                "=",
                "'string'",
                Ok,
                "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291,"
                " -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS("
                " POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND"
                " string_column = 'string' ORDER BY dist ASC",
            ),
            (
                "string_column",
                "LIKE",
                "'string'",
                Ok,
                "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291,"
                " -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS("
                " POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND"
                " string_column LIKE 'string' ORDER BY dist ASC",
            ),
            (
                "parallax",
                ">=",
                5,
                Ok,
                "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291,"
                " -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS("
                " POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND"
                " parallax >= 5 ORDER BY dist ASC",
            ),
        ],
    )
    def test_single_where(
        self, column, operator, value, test, correct, mock_simbad_search
    ):
        q = query_region("dummy_name", 0.5)
        verify_result(test, lambda: q.where((column, operator, value)))
        if test == Ok:
            assert format_query_string(q.build()) == correct

    def test_multiple_where(self, mock_simbad_search):
        correct = (
            "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1)"
            " ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra,"
            " dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND column1 >= 5 AND column2"
            " <= 17 AND column3 LIKE 'string' ORDER BY dist ASC"
        )
        q = (
            query_region("dummy_name", 0.5)
            .where(("column1", ">=", 5))
            .where([("column2", "<=", 17), ("column3", "LIKE", "'string'")])
        )
        assert format_query_string(q.build()) == correct

    def test_or_where(self, mock_simbad_search):
        correct = (
            "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1)"
            " ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra,"
            " dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND column1 >= 5 AND ("
            " column2 <= 17 ) AND ( column3 <= 3 OR column4 LIKE 'string' ) ORDER BY"
            " dist ASC"
        )
        q = (
            query_region("dummy_name", 0.5)
            .where(("column1", ">=", 5))
            .or_where(("column2", "<=", 17))
            .or_where([("column3", "<=", 3), ("column4", "LIKE", "'string'")])
        )
        assert format_query_string(q.build()) == correct

    def test_top(self, mock_simbad_search):
        correct = (
            "SELECT TOP 31 *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291,"
            " -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS("
            " POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) ORDER BY"
            " dist ASC"
        )
        q = query_region("dummy_name", 0.5).top(10).top(31)
        assert format_query_string(q.build()) == correct

    def test_build_count(self, mock_simbad_search):
        correct = (
            "SELECT COUNT(*) FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS("
            " POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND column1"
            " >= 5 AND column2 <= 17 AND column3 LIKE 'string'"
        )
        q = (
            query_region("dummy_name", 0.5)
            .where(("column1", ">=", 5))
            .where(("column2", "<=", 17))
            .where(("column3", "LIKE", "'string'"))
        )
        assert format_query_string(q.build_count()) == correct

    def test_get(self, mock_simbad_search, mock_gaia_launch_job_async, mocker):
        q = (
            query_region("gaiaedr3.gaia_source", 0.5)
            .select("ra", "dec", "parallax", "pmra", "pmdec")
            .top(5)
        )
        query = q.build()
        result = q.get()
        mock_gaia_launch_job_async.assert_called_with(query=query)
        assert isinstance(result, Table)
        result = q.get(dump_to_file=True, output_file="test_file.xml")
        mock_gaia_launch_job_async.assert_called_with(
            query=query, dump_to_file=True, output_file="test_file.xml"
        )
        assert result is None

    def test_count(self, mock_simbad_search, mock_gaia_launch_job_async, mocker):
        q = (
            query_region("gaiaedr3.gaia_source", 0.5)
            .select("ra", "dec", "parallax", "pmra", "pmdec")
            .top(5)
        )
        query = q.build_count()
        result = q.count()
        mock_gaia_launch_job_async.assert_called_with(query=query)
        assert isinstance(result, Table)
        result = q.count(dump_to_file=True, output_file="test_file.xml")
        mock_gaia_launch_job_async.assert_called_with(
            query=query, dump_to_file=True, output_file="test_file.xml"
        )
        assert result is None
