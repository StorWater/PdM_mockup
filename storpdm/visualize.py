import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px


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


def define_visibles(x):
    """Funcion usada para definir qué traces se muestran en un grafico de plotly

        Examples
        --------
        >>> define_visibles(x=[2,1,3])
        [[True, True, False, False, False, False],
        [False, False, True, False, False, False],
        [False, False, False, True, True, True]]

        Parameters
        -------
        x: list or np.array
                Contiene el numero de clases/traces por cada dropdown menu. [1,1,1] significa 1 trace, 3 dropdown menus
        
        Returns
        -------
        list
                Lista de listas, contiene solo True or False. 
        """
    if isinstance(x, list):
        x = np.array(x)

    visible_trace = []
    for i, a in enumerate(x):
        visible_trace.append(
            [False] * np.sum(x[0:i]) + [True] * x[i] + [False] * np.sum(x[i + 1 :])
        )

    return visible_trace



def interactive_rul_series(df, filename = None):
    """Create interative plot

    # TODO: document and make general
    """
    z_all = df.columns.drop("RUL")
    fig = go.Figure()
    buttons = []
    visible_start = [True] + [False] * (len(z_all) - 1)

    # Select which columns are visible for each button
    visible_trace = define_visibles([100] * len(z_all))

    # Loop over the selected columns and create trace
    for i, z in enumerate(z_all):

        # Generate figure and keep data and layout
        temp_fig  = px.line(df, x="RUL", y=z, color="id",hover_name="op1")
   
        # Add traces, one per boxplot
        for f in temp_fig.data:
            f.legendgroup= f.legendgroup[len(z)+1:]
            #f.name = f.name[len(z)+1:]
            fig.add_trace(go.Scattergl( f, visible=visible_start[i]))
            
        # First one is visible
        if visible_start[i] is True:
            fig.update_layout(temp_fig.layout) # need to provide for a layout on the first time
        

        # Crear botones
        buttons.append(
            dict(label=z_all[i], method="update", args=[{"visible": visible_trace[i]},
                                                        temp_fig.layout])
        )
        
    # Añadir botones
    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                direction="up",
                showactive=True,
                xanchor="center",
                yanchor="bottom",
                active=0,
                pad={"l": 150, "b": -420, "t": 0},
                # Update data
                buttons=buttons,
            )
        ]
    )

    fig.update_layout(width=900, height=500, title="Sensor and operating settings vs RUL")

    if filename:
       plotly.offline.plot(fig, filename=filename)
    
    return fig