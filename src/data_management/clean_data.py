import pandas as pd

from bld.project_paths import project_paths_join as ppj

# load the data.
hom_rate = pd.read_csv(ppj("IN_DATA", "homicide_rate_by_country.csv"), skiprows=4)

# set index to country name.
hom_rate.set_index(hom_rate["Country Name"], inplace=True)

# take the homicide rate for Colombia.
hom_rate_col = hom_rate.loc["Colombia"].copy()
hom_rate

# drop years where there is no information.
hom_rate_col.dropna(inplace=True)

# drop the unnecessary information
hom_rate_col.drop(hom_rate_col[:4].index, inplace=True)

# construct data frame from series.
hom_rate_col_df = pd.DataFrame(
    {"Year": hom_rate_col.index, "Homicide": hom_rate_col.values}
)

# save the data.
hom_rate_col_df.to_csv(ppj("OUT_DATA", "col_historic_homicide.csv"))
