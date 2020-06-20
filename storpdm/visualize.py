import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualise_sensor_correlation_all_engine(data_df):
    """Plot and save the correlation between sensors for all engines in the dataset.

    Parameters
    -----------
        data_df : pd.dataFrame)
            Dataframe with engine data to be plotted.

    Returns
    --------
    matplotlib.figure.Figure
        Heatmap representing correlation between features
    """

    # Remove the time column from the dataframe.
    modified_df = data_df.drop("cycle", axis=1)

    # Create correlation.
    sensor_corr = modified_df.corr()

    # Define and show correlation plot.
    corr_fig = sns.heatmap(
        sensor_corr,
        xticklabels=sensor_corr.columns.values,
        yticklabels=sensor_corr.columns.values,
        cmap="RdYlGn",
    )
    plt.title("Engine Data Correlation")

    return corr_fig


def visualise_sensor_data_distribution(dataset_df):
    """Plot and save the sensor data distributions for all engines.
    
    Parameters
    -----------
    data_df:  pd.dataframe
        Dataframe with engine data to be plotted.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with subplots, one distribution in each one.
    """

    # Prepare dataset for plot.
    plotted_dataset_df = dataset_df.copy()

    columns = plotted_dataset_df.columns

    figs_per_row = 4
    n_columns = len(columns)
    n_rows = (int(n_columns / figs_per_row)) + 1

    # Create plot.
    fig, axes = plt.subplots(n_rows, figs_per_row, figsize=(15, 15))
    axes = axes.flatten()
    fig.suptitle("Sensor Distributions - All Engines")

    for column, ax in zip(columns, axes):
        ax = sns.distplot(plotted_dataset_df[column], ax=ax, label=column)
        ax.legend(loc=1)

    # Save plot.
    return fig


def plot_time_history_all_engines(df):
    """Plot and save the complete engine time history (normalised to RUL) for 
    all engines in dataset.
   
    Parameters
    ----------
    dataset_df : pd.dataframe
        Dataframe with engine data to be plotted.
    
    Returns
    --------
    matplotlib.figure.Figure
        Figure with subplots, one distribution in each one.
    """
    
    columns = df.columns.drop("RUL")

    # Define and show plot.
    fig, axes = plt.subplots(len(columns) - 1, 1, figsize=(35, 17))

    for column, ax in zip(columns, axes):

        ax.set_title(column, loc="left", fontdict={"fontsize": 14})

        # Add data for each engine to axis.
        for engine in df["id"].unique():
            idx = df.id == engine
            ax.plot(df.RUL.loc[idx], df[column].loc[idx], label=column)
            ax.set(xlabel="RUL", title=column)

    # Add figure title.
    fig.suptitle("Run to Failure - All Engines")

    return fig
