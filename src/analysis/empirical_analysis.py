import pickle

import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.model_code.breakpoints import breakpoints

# load the data.
col_hom_rate = pd.read_csv(ppj("OUT_DATA", "col_historic_homicide.csv"))

# extract information from the data
years = col_hom_rate["Year"].tolist()
hom_rates = col_hom_rate["Homicide"].tolist()

# find optimal breakpoints.
breakpoints_col = breakpoints("Homicide ~ 1", col_hom_rate, h=0.1)

opt_breaks = breakpoints_col[0]
opt_obs = breakpoints_col[1]

# find date of breaks.
if opt_breaks == 0:
    opt_dates = np.nan
else:
    opt_dates = []
    for i in opt_obs:
        opt_dates.append(years[i])


# build dictionary with data.
breakpoint_info = {
    "Years": years,
    "Homicide Rate": hom_rates,
    "Optimal Number of Breaks": opt_breaks,
    "Observation of Breaks": opt_obs,
    "Optimal Break Dates": opt_dates,
}

# store the information of the breakpoint analysis.
with open(ppj("OUT_ANALYSIS", "colombia_hom_breaks.pickle"), "wb") as out_file:
    pickle.dump(breakpoint_info, out_file)
