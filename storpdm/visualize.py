import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualise_sensor_correlation_all_engine(data_df):
    """Plot and save the correlation between sensors for all engines in the dataset.

    Parameters
    -----------
        data_df : pd.dataFrame)
            Dataframe with engine data to be plotted.
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

