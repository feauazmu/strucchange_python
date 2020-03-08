import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from bld.project_paths import project_paths_join as ppj

# parameters.
dpi = 600

# plot empirical analysis

# load plot information.
with open(ppj("OUT_ANALYSIS", "colombia_hom_breaks.pickle"), "rb") as in_file:
    breakpoint_info = pickle.load(in_file)

# plot the homicide rate per year
fig, ax = plt.subplots()
ax.plot(breakpoint_info["Years"], breakpoint_info["Homicide Rate"], color="blue")
ax.set(
    xlabel="Years",
    ylabel="Intentional homicides (per 100 000 people)",
    title="Historical homicide rate in Colombia",
)
fig.savefig(ppj("OUT_FIGURES", "colombia_hom.png"), dpi=dpi)

# plot the breakpoints
breakpoint_info
for xc in breakpoint_info["Optimal Break Dates"]:
    ax.axvline(x=xc, color="r", linestyle="--", lw=0.4)
fig.savefig(ppj("OUT_FIGURES", "colombia_hom_bp.png"), dpi=dpi)

breakpoint_info["Optimal Break Dates"]

# plot timing

# load information
with open(ppj("OUT_ANALYSIS", "timing_info.pickle"), "rb") as in_file:
    timing_info = pickle.load(in_file)

# plot the timing
runtimes_plot = sns.regplot(x=timing_info[0], y=timing_info[1], fit_reg=True)
runtimes_plot.set(
    title="Runtime of breakpoints function for number of observations.",
    xlabel="Number of observations",
    ylabel="Time",
)
runtimes_plot.get_figure().savefig(ppj("OUT_FIGURES", "runtimes_plot.png"), dpi=dpi)
