from scludam import Query, search_object, search_table


iden = "ngc2527"

object_info = search_object(iden)
print(f'Object found in {object_info.coords}')
object_info.table.write(f"examples/{iden}_object_metadata.txt", format="ascii", overwrite=True)

default_table = Query().table

tables = search_table(default_table)
first_table = tables[0]
print(f"name: {first_table.name}\n description: {first_table.description}\n")
first_table.columns.write(f"examples/{default_table}_columns_metadata.txt", format="ascii", overwrite=True)

query = (
    Query()
    .select(
        "ra",
        "dec",
        "ra_error",
        "dec_error",
        "ra_dec_corr",
        "pmra",
        "pmra_error",
        "ra_pmra_corr",
        "dec_pmra_corr",
        "pmdec",
        "pmdec_error",
        "ra_pmdec_corr",
        "dec_pmdec_corr",
        "pmra_pmdec_corr",
        "parallax",
        "parallax_error",
        "parallax_pmra_corr",
        "parallax_pmdec_corr",
        "ra_parallax_corr",
        "dec_parallax_corr",
        "phot_g_mean_mag",
    )
    .where_in_circle(iden, 2.5)
    .where(
        [
            ("parallax", ">", 0.2),
            ("phot_g_mean_mag", "<", 18),
        ]
    )
    # low noise_sig means "do not rely on excess noise, do not check it"
    # high noise_sig means "you should check that excess noise is small"
    .where_arenou_criterion()
    .where_aen_criterion()
)

count = query.count()
print(f'Stars found: {count["count_all"][0]}')

data = query.get()
data.write(f'examples/{iden}_data.xml', format="votable", overwrite=True)