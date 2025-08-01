"""Test data fetcher module."""

import re
from itertools import chain

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from astroquery.utils.tap.model.job import Job
from astroquery.utils.tap.model.tapcolumn import TapColumn
from astroquery.utils.tap.model.taptable import TapTableMeta

from scludam import Query, search_object, search_objects_near_data, search_table
from scludam.fetcher import Config, SimbadResult


def simbad_query_object(identifier):
    if identifier == "ic2395":
        return Table.read("tests/files/simbad_response.xml", format="votable")
    return None


def query_criteria_mock():
    return Table.read("tests/files/query_criteria.xml", format="votable")


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


def search_object_mock():
    return SimbadResult(
        coords=SkyCoord(
            "08 42 31.0", "-48 06 00", unit=(u.hourangle, u.deg), frame="icrs"
        )
    )


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
    return mocker.patch(
        "astroquery.simbad.SimbadClass.query_object",
        side_effect=simbad_query_object,
    )


@pytest.fixture
def mock_simbad_add_votable_columns(mocker):
    return mocker.patch("astroquery.simbad.SimbadClass.add_votable_fields")


@pytest.fixture
def mock_gaia_load_tables(mocker):
    return mocker.patch(
        "astroquery.gaia.Gaia.load_tables", return_value=gaia_load_tables()
    )


@pytest.fixture
def mock_search_object(mocker):
    return mocker.patch(
        "scludam.fetcher.search_object", return_value=search_object_mock()
    )


@pytest.fixture
def mock_simbad_query_criteria(mocker):
    return mocker.patch(
        "astroquery.simbad.SimbadClass.query_criteria",
        return_value=query_criteria_mock(),
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


def test_search_object_valid_id(
    mock_simbad_query_object, mock_simbad_add_votable_columns, mocker
):
    result = search_object("ic2395", cols=["coordinates", "parallax"])
    assert result.coords.to_string("hmsdms", precision=2) == SkyCoord(
        ra=130.62916667, dec=-48.1, frame="icrs", unit="deg"
    ).to_string("hmsdms", precision=2)
    assert isinstance(result.table, Table)
    mock_simbad_query_object.assert_called_with("ic2395")
    mock_simbad_add_votable_columns.assert_called_with("coordinates", "parallax")


def test_search_object_invalid_id(mock_simbad_query_object, mocker):
    empty_result = search_object("invalid_identifier")
    assert empty_result.table is None and empty_result.coords is None


def test_search_table_only_names(mock_gaia_load_tables):
    result = search_table("gaiaedr3", only_names=True)
    assert len(result) == 1
    assert result[0].name == "gaiaedr3.gaia_source"
    assert isinstance(result[0].description, str)
    assert result[0].columns is None


def test_search_table_full_table_data(mock_gaia_load_tables):
    result = search_table("gaiaedr3")
    assert isinstance(result[0].columns, Table)


class TestQuery:
    def test_constructor_from_table(self):
        correct = "SELECT * FROM table"
        assert format_query_string(Query("table").build()) == correct

    def test_constructor_from_default_table(self):
        correct = f"SELECT * FROM {Config().MAIN_GAIA_TABLE}"
        assert format_query_string(Query().build()) == correct

    def test_select_all(self):
        correct = "SELECT * FROM table"
        assert format_query_string(Query("table").select().build()) == correct
        assert format_query_string(Query("table").select("*").build()) == correct
        assert (
            format_query_string(Query("table").select("*", "col1").build()) == correct
        )

    def test_select_cols(self):
        correct = "SELECT col1, col2 FROM table"
        assert (
            format_query_string(Query("table").select("col1", "col2").build())
            == correct
        )

    def test_select_override_yields_last_value(self):
        correct = "SELECT col2 FROM table"
        assert (
            format_query_string(Query("table").select("col1").select("col2").build())
            == correct
        )
        correct = "SELECT * FROM table"
        assert format_query_string(Query("table").select("col1").select().build())

    def test_top_valid(self):
        correct = "SELECT TOP 31 * FROM table"
        assert format_query_string(Query("table").top(31).build()) == correct

    def test_top_invalid_yields_all(self):
        correct = "SELECT * FROM table"
        assert format_query_string(Query("table").top(0).build()) == correct
        assert format_query_string(Query("table").top(-1).build()) == correct

    def test_top_override_yields_last_value(self):
        correct = "SELECT TOP 31 * FROM table"
        assert format_query_string(Query("table").top(15).top(31).build()) == correct

    def test_where_valid_tuple_adds_where_clause(self):
        correct = "SELECT * FROM table WHERE col1 = 'value'"
        assert (
            format_query_string(Query("table").where(("col1", "=", "'value'")).build())
            == correct
        )
        correct = "SELECT * FROM table WHERE col1 LIKE '%value%'"
        assert (
            format_query_string(
                Query("table").where(("col1", "LIKE", "'%value%'")).build()
            )
            == correct
        )
        correct = "SELECT * FROM table WHERE col1 >= 3"
        assert (
            format_query_string(Query("table").where(("col1", ">=", 3)).build())
            == correct
        )
        correct = "SELECT * FROM table WHERE col1 <= 3.5"
        assert (
            format_query_string(Query("table").where(("col1", "<=", 3.5)).build())
            == correct
        )

    def test_where_invalid_column_raises_error(self):
        with pytest.raises(KeyError, match="Invalid column name: col3"):
            Query("table").select("col1", "col2").where(("col3", "=", "'value'"))

    def test_where_invalid_operator_raises_error(self):
        with pytest.raises(ValueError, match="Invalid operator: invalid"):
            Query("table").where(("col1", "invalid", "'value'"))

    def test_where_list_tuple_adds_and_expression(self):
        correct = "SELECT * FROM table WHERE col1 = 'value' AND col2 = 'value2'"
        assert (
            format_query_string(
                Query("table")
                .where([("col1", "=", "'value'"), ("col2", "=", "'value2'")])
                .build()
            )
            == correct
        )

    def test_where_repeat_adds_and_expression(self):
        correct = "SELECT * FROM table WHERE col1 = 'value' AND col2 = 'value2'"
        assert (
            format_query_string(
                Query("table")
                .where(("col1", "=", "'value'"))
                .where(("col2", "=", "'value2'"))
                .build()
            )
            == correct
        )

    def test_where_or(self):
        correct = (
            "SELECT * FROM table WHERE col1 >= 5 AND ("
            "col2 <= 17) AND (col3 <= 3 OR col4 LIKE 'value')"
        )
        assert (
            format_query_string(
                (
                    Query("table")
                    .where(("col1", ">=", 5))
                    .where_or(("col2", "<=", 17))
                    .where_or([("col3", "<=", 3), ("col4", "LIKE", "'value'")])
                    .build()
                )
            )
            == correct
        )

    @pytest.mark.parametrize(
        "center",
        [
            search_object_mock().coords,
            "some_identifier",
            (
                search_object_mock().coords.ra.value,
                search_object_mock().coords.dec.value,
            ),
        ],
    )
    @pytest.mark.parametrize("radius", [0.5, 0.5 * u.deg, 30 * u.arcmin])
    def test_where_in_circle(self, center, radius, mock_search_object, mocker):
        correct = (
            f"SELECT *, {Config().MAIN_GAIA_RA}, {Config().MAIN_GAIA_DEC}, DISTANCE("
            " POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1)) AS dist FROM"
            " table WHERE col1 <= 1 AND 1 = CONTAINS( POINT('ICRS', ra, dec),"
            " CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND col2 >= 5 ORDER BY dist ASC"
        )
        query = (
            Query("table")
            .where(("col1", "<=", 1))
            .where_in_circle(center, radius)
            .where(("col2", ">=", 5))
        )
        assert format_query_string(query.build()) == correct
        if isinstance(center, str):
            mock_search_object.assert_called_with("some_identifier")

    def test_where_astrometric_aen_criterion(self):
        correct = (
            f"SELECT *, {Config().MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE},"
            f" {Config().MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE_SIG} FROM table WHERE"
            f" ({Config().MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE_SIG} <= 2 OR"
            f" {Config().MAIN_GAIA_ASTROMETRIC_EXCESS_NOISE} < 2)"
        )
        assert (
            format_query_string(Query("table").where_aen_criterion().build()) == correct
        )
        correct = "SELECT *, col1, col2 FROM table WHERE (col2 <= 3 OR col1 < 3)"
        assert (
            format_query_string(
                Query("table").where_aen_criterion(3, 3, "col1", "col2").build()
            )
            == correct
        )

    def test_where_arenou_criterion(self):
        correct = (
            f"SELECT *, {Config().MAIN_GAIA_BP_RP},"
            f" {Config().MAIN_GAIA_BP_RP_EXCESS_FACTOR} FROM table WHERE"
            f" {Config().MAIN_GAIA_BP_RP_EXCESS_FACTOR} > 1 + 0.015 *"
            f" POWER({Config().MAIN_GAIA_BP_RP}, 2) AND"
            f" {Config().MAIN_GAIA_BP_RP_EXCESS_FACTOR} < 1.3 + 0.006 *"
            f" POWER({Config().MAIN_GAIA_BP_RP}, 2)"
        )
        assert (
            format_query_string(Query("table").where_arenou_criterion().build())
            == correct
        )
        correct = (
            "SELECT *, col1, col2 FROM table WHERE col2 > 1 + 0.015 * POWER(col1, 2)"
            " AND col2 < 1.3 + 0.006 * POWER(col1, 2)"
        )
        assert (
            format_query_string(
                Query("table").where_arenou_criterion("col1", "col2").build()
            )
            == correct
        )

    def test_build_count(self, mock_search_object):
        correct = (
            "SELECT COUNT(*) FROM table WHERE col1 = 'value' AND 1 = CONTAINS("
            " POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND col2"
            " >= 5"
        )
        assert (
            format_query_string(
                (
                    Query("table")
                    .where(("col1", "=", "'value'"))
                    .where_in_circle("dummy_name", 0.5)
                    .where(("col2", ">=", 5))
                    .build_count()
                )
            )
            == correct
        )

    def test_get(self, mock_search_object, mock_gaia_launch_job_async, mocker):
        q = (
            Query("gaiaedr3.gaia_source")
            .select("ra", "dec", "parallax", "pmra", "pmdec")
            .where(("parallax", ">=", 0.5))
            .where_in_circle("dummy_name", 0.5)
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

    def test_count(self, mock_search_object, mock_gaia_launch_job_async, mocker):
        q = (
            Query("gaiaedr3.gaia_source")
            .select("ra", "dec", "parallax", "pmra", "pmdec")
            .where(("parallax", ">=", 0.5))
            .where_in_circle("dummy_name", 0.5)
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

    def test_get_objects_near_data(
        self, mock_simbad_add_votable_columns, mock_simbad_query_criteria, mocker
    ):
        df = Table.read("tests/data/ngc2323_data.fits").to_pandas()
        result = search_objects_near_data(df).to_pandas()
        included_cols = sorted(
            [
                "fluxdata(B)",
                "fluxdata(R)",
                "parallax",
                "propermotions",
                "otype",
                "coordinates",
                "diameter",
                "fluxdata(G)",
                "typed_id",
            ]
        )
        criteria = (
            "ra >= 104.99113744429378 & dec >= -9.03711496168022 & ra <="
            " 106.40326142636158 & dec <= -7.6386969625395436"
        )
        mock_simbad_add_votable_columns.assert_called_with(*included_cols)
        mock_simbad_query_criteria.assert_called_with(criteria)
        correct_result = Table.read(
            "tests/files/object_near_data_result.xml"
        ).to_pandas()
        assert correct_result.shape == result.shape
        assert sorted(correct_result.columns) == sorted(result.columns)
