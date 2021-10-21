import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.load_mcs import load_domestic_hps

def plottable_dhps():
    dhps = load_domestic_hps()
    dhps["year"] = dhps.date.dt.year
    dhps_noexhaust = dhps[dhps.tech_type != "Exhaust Air Heat Pump"]
    
    return dhps_noexhaust

# median installation cost per year
def plot_median_costs(df):
    
    cost_data = (
        df
        .groupby(["year", "tech_type"])
        ["cost"]
        .median()
        .loc[2015:2021]
        .unstack()
    )

    fig, ax = plt.subplots()
    
    ax.plot(
        cost_data.index,
        cost_data['Ground/Water Source Heat Pump'].tolist(),
        label = 'Ground/Water Source Heat Pump',
        c = 'blue'
    )
    ax.plot(
        cost_data.index,
        cost_data['Air Source Heat Pump'].tolist(),
        label = 'Air Source Heat Pump',
        c = 'orange'
    )
    ax.set_ylim(0, 21000)
    ax.grid(axis='y', color='0.8')
    ax.set_xlabel('Year')
    ax.set_ylabel('Median cost of installation (£)')
    ax.legend(title = "Heat pump type")
    ax.set_title('Median cost of heat pump installations\n(years with >100 installations only)')

    plt.savefig(PROJECT_DIR / 'outputs/figures/median_costs.png')


# numbers of installations each year
def plot_installation_numbers(df):
    
    numbers_data = (
        df
        .groupby(["year", "tech_type"])
        ["cost"]
        .count()
        .unstack()
        .loc[2014:2021]
    )

    fig, ax = plt.subplots()
    
    numbers_data.plot(
        kind="bar",
        ax=ax,
        zorder=3,
        color=['orange', 'blue']
    )
    ax.set_xlabel('Year')
    ax.set_ylabel('Installations')
    ax.grid(axis='y', color='0.8', zorder=0)
    ax.legend(title = 'Heat pump type')
    ax.set_title('Numbers of heat pumps installed')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(PROJECT_DIR / 'outputs/figures/median_costs.png')





#### Geographic distribution - local authorities
def regional_plot(df):
    df.groupby('local_authority').agg(
            number = ('cost', lambda col: col.count()), 
            median_cost = ('cost', lambda col: col.median())
        ).sort_values('median_cost', ascending=False)
    # tbc...

# some LAs with not many installations
# some non-standard local authority names -
# e.g. "Belfast City Council, Lisburn & Castlereagh..."


def main():
    """Main function: Plots MCS data."""
    dhps_noexhaust = plottable_dhps()
    plot_median_costs(dhps_noexhaust)
    plot_installation_numbers(dhps_noexhaust)


if __name__ == "__main__":
    # Execute only if run as a script
    main()