import itertools

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "none"


def visualise_sensor_correlation_all_engine(df, title="Correlation Matrix"):
    """Plot and save the correlation between sensors for all engines in the dataset.

    Parameters
    -----------
        data_df : pd.dataFrame)
            Dataframe with engine data to be plotted.

    Returns
    --------
    plotly.Figure
        Heatmap representing correlation between features
    """

    df = df.drop(["id", "cycle"], axis=1)
    N = df.shape[1]
    corr = df.corr()
    labels = df.columns.to_list()

    # Show low triangular only
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i > j:
                corr.iloc[i, j] = None

    # Define labels and colors
    hovertext = [
        [f"corr({labels[i]}, {labels[j]})= {corr.iloc[i,j]:.2f}" for j in range(N)]
        for i in range(N)
    ]

    sns_colorscale = [
        [0.0, "#3f7f93"],  # cmap = sns.diverging_palette(220, 10, as_cmap = True)
        [0.071, "#5890a1"],
        [0.143, "#72a1b0"],
        [0.214, "#8cb3bf"],
        [0.286, "#a7c5cf"],
        [0.357, "#c0d6dd"],
        [0.429, "#dae8ec"],
        [0.5, "#f2f2f2"],
        [0.571, "#f7d7d9"],
        [0.643, "#f2bcc0"],
        [0.714, "#eda3a9"],
        [0.786, "#e8888f"],
        [0.857, "#e36e76"],
        [0.929, "#de535e"],
        [1.0, "#d93a46"],
    ]

    heat = go.Heatmap(
        z=corr,
        x=labels,
        y=labels,
        xgap=1,
        ygap=1,
        colorscale=sns_colorscale,
        colorbar_thickness=20,
        colorbar_ticklen=3,
        hovertext=hovertext,
        hoverinfo="text",
    )

    layout = go.Layout(
        title_text=title,
        title_x=0.5,
        width=400,
        height=400,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",
    )

    return go.Figure(data=[heat], layout=layout)


def visualise_sensor_correlation_double(df1, df2, subplot_titles=("", "")):
    """[summary]
    doc: TODO

    Parameters
    ----------
    df1 : [type]
        [description]
    df2 : [type]
        [description]
    subplot_titles : tuple of str
        ...
    """

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=subplot_titles,
        print_grid=False,
        horizontal_spacing=0.2,
    )

    fig_subplot1 = visualise_sensor_correlation_all_engine(df1)
    fig.add_trace(go.Heatmap(fig_subplot1.data[0]), row=1, col=1)

    fig_subplot2 = visualise_sensor_correlation_all_engine(df2)
    fig.add_trace(go.Heatmap(fig_subplot2.data[0]), row=1, col=2)

    fig.update_layout(
        height=600, width=1000, margin=dict(l=150, b=150), font=dict(size=10,)
    )
    return fig


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
                Contiene el numero de clases/traces por cada dropdown menu. 
                [1,1,1] significa 1 trace, 3 dropdown menus
        
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


def interactive_rul_series(df, id_filter=None, filename=None):
    """Create interative plot

    # TODO: document and make general
    """

    if id_filter:
        # Filter IDs
        df = df[df.id.isin(id_filter)]

    z_all = df.columns.drop("RUL")
    fig = go.Figure()
    buttons = []
    visible_start = [True] + [False] * (len(z_all) - 1)

    # Select which columns are visible for each button
    n_unique = df.id.nunique()
    visible_trace = define_visibles([n_unique] * len(z_all))

    # Loop over the selected columns and create trace
    for i, z in enumerate(z_all):

        # Generate figure and keep data and layout
        temp_fig = px.line(
            df, x="RUL", y=z, color="id", hover_name="op1", render_mode="webgl"
        )

        # Add traces, one per boxplot
        for f in temp_fig.data:
            f.legendgroup = f.legendgroup[len(z) + 1 :]
            # f.name = f.name[len(z)+1:]
            fig.add_trace(go.Scattergl(f, visible=visible_start[i]))

        # First one is visible
        if visible_start[i] is True:
            fig.update_layout(
                temp_fig.layout
            )  # need to provide for a layout on the first time

        # Crear botones
        buttons.append(
            dict(
                label=z_all[i],
                method="update",
                args=[{"visible": visible_trace[i]}, temp_fig.layout],
            )
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

    fig.update_layout(
        width=900, height=500, title="Sensor and operating settings vs RUL"
    )

    if filename:
        plotly.offline.plot(fig, filename=filename)

    return fig


def display_roc_pr(
    precision, recall, fpr, thres, title="", fig=None, color=("#1f77b4", "#ff7f0e")
):
    """[summary]

    TODO: documentation
    
    Parameters
    ----------
    precision : [type]
        [description]
    recall : [type]
        [description]
    fpr : [type]
        [description]
    thres : [type]
        [description]
    title : str, optional
        [description], by default ""
    fig : [type], optional
        [description], by default None
    i_col : int, optional
        [description], by default 0

    Returns
    -------
    [type]
        [description]
    """

    colors = [
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
    ]
    hovertext = [
        f"Recall: {r:.3f}<br>Precision: {p:.3f}<br>FPR: {f:.3f}<br>Thres: {t:.2f}"
        for r, p, f, t in zip(recall, precision, fpr, thres)
    ]

    # ROC curve
    lw = 2
    if fig is None:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC", "Precision-recall"))

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=recall,
            mode="lines",
            name=f"{title} [ROC]",
            hovertext=hovertext,
            hoverinfo="text",
            line=dict(color=color[0]),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="navy", width=lw, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Precision-recall curve
    fig.add_trace(
        go.Scatter(
            x=precision,
            y=recall,
            mode="lines",
            name=f"{title} [Precision-recall]",
            hovertext=hovertext,
            hoverinfo="text",
            line=dict(color=color[1]),
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="Recall (TPR)", row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)

    fig.update_yaxes(title_text="Recall (TPR)", row=1, col=2)
    fig.update_xaxes(title_text="Precision", row=1, col=2)

    fig.update_layout(height=500)

    return fig


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """[summary]

    TODO: docum
    Parameters
    ----------
    cm : [type]
        [description]
    classes : [type]
        [description]
    normalize : bool, optional
        [description], by default False
    title : str, optional
        [description], by default 'Confusion matrix'
    cmap : [type], optional
        [description], by default plt.cm.Blues
    """

    # General outline parameters
    font = {"family": "DejaVu Sans", "weight": "bold", "size": 15}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["axes.grid"] = False

    if normalize:
        print("Confusion matrix before normalization")
        print(cm)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("Actual label", fontsize=18)
    plt.xlabel("Predicted label", fontsize=18)
    plt.tight_layout()


def actual_vs_pred(model, X_test, y_test, df_train_proc2):
    """[summary]

    TODO: document and clean

    Returns
    -------
    [type]
        [description]
    """

    y_test_pred = model.predict(X_test)

    df_plot = pd.DataFrame(
        {"y_test": y_test, "y_test_pred": y_test_pred, "cycle": X_test.cycle}
    )
    df_plot = df_plot.sort_values("y_test")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Actual vs predicted RUL",
            "Actual vs predcted RUL, selected cycles",
        ),
        print_grid=False,
        horizontal_spacing=0.2,
    )

    # Create traces
    fig.add_trace(
        go.Scattergl(
            x=df_plot["y_test"],
            y=df_plot["y_test_pred"],
            mode="markers",
            marker=dict(
                color=np.abs(df_plot["y_test"] - df_plot["y_test_pred"]),
                colorscale="Viridis",
                line_width=0,
                opacity=0.7,
                size=6,
            ),
            name="Data Points",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=350,
        y1=350,
        line=dict(width=4, dash="dot", color="grey"),
        xref="x1",
        yref="y1",
    )
    fig.update_xaxes(title_text="Actual RUL", row=1, col=1)
    fig.update_yaxes(title_text="Predicted RUL", row=1, col=1)

    y_train_pred = model.predict(df_train_proc2.drop(["RUL", "id"], axis=1))
    df_plot = pd.DataFrame(
        {
            "y_train": df_train_proc2.RUL,
            "y_train_pred": y_train_pred,
            "id": df_train_proc2.id,
        }
    )

    # Create traces
    engine_list = [96, 39, 80, 71]

    for engine in engine_list:
        idx = df_plot.id == engine
        df_plot_filter = df_plot[idx]
        fig.add_trace(
            go.Scattergl(
                x=df_plot_filter["y_train"],
                y=df_plot_filter["y_train_pred"],
                # mode="markers",
                marker=dict(
                    color=np.abs(
                        df_plot_filter["y_train"] - df_plot_filter["y_train_pred"]
                    ),
                    colorscale="Viridis",
                    line_width=0,
                    opacity=0.7,
                    size=6,
                ),
                name=f"Enigne {engine}",
            ),
            row=1,
            col=2,
        )
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=350,
        y1=350,
        line=dict(width=4, dash="dot", color="grey"),
        xref="x2",
        yref="y2",
    )
    fig.update_xaxes(title_text="Actual RUL", row=1, col=2)
    fig.update_yaxes(title_text="Predicted RUL", row=1, col=2)

    return fig


def display_tp_fp(thres, tp, fp, title="", fig=None, i=0, name1="", name2=""):
    """Display true positives and false positive

    Parameters
    ----------
    thres : [type]
        [description]
    tp : [type]
        [description]
    fp : [type]
        [description]
    title : str, optional
        [description], by default ""
    fig : [type], optional
        [description], by default None
    i : int, optional
        [description], by default 0
    name1 : str, optional
        [description], by default ""
    name2 : str, optional
        [description], by default ""

    Returns
    -------
    [type]
        [description]
    """

    colors = [
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
    ]
    hovertext = [
        f"TP: {r}<br>FP: {p}<br>Thres: {t:.2f}" for r, p, t in zip(tp, fp, thres)
    ]

    # ROC curve Train
    lw = 2
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    opacity = 0.8

    ## PRecision
    fig.add_trace(
        go.Bar(
            x=thres[::10],
            y=tp[::10],
            name=name1,
            hovertext=hovertext[::10],
            hoverinfo="text",
            marker_color=colors[9],
            opacity=opacity,
            xaxis="x1",
            yaxis="y1",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=thres,
            y=tp,
            mode="lines",
            name=name1,
            hovertext=hovertext,
            hoverinfo="text",
            line=dict(color=colors[9]),
            showlegend=False,
            xaxis="x1",
            yaxis="y1",
        ),
        secondary_y=False,
    )

    ## recall curve
    fig.add_trace(
        go.Bar(
            x=thres[::10],
            y=fp[::10],
            name=name2,
            hovertext=hovertext[::10],
            hoverinfo="text",
            marker_color=colors[4],
            opacity=opacity,
            xaxis="x1",
            yaxis="y1",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=thres,
            y=fp,
            mode="lines",
            name=name2,
            hovertext=hovertext,
            hoverinfo="text",
            line=dict(color=colors[4]),
            showlegend=False,
            xaxis="x1",
            yaxis="y1",
        ),
        secondary_y=False,
    )

    fig.update_yaxes(title_text="")
    fig.update_xaxes(title_text="Probability threshold")
    fig.update_yaxes(range=[0, 210])
    fig.update_xaxes(range=[0.25, 0.95])

    fig.update_layout(height=500)

    return fig


def plot_prob_RUL(model, df_train_proc2, engine = 96):
    """Plot probability versus RUL

    TODO: doc
    """
    y_train_pred = model.predict_proba(
        df_train_proc2.drop(["RUL", "id", "RUL_thres"], axis=1)
    )[:, 1]

    df_plot = pd.DataFrame(
        {
            "y_train": df_train_proc2.RUL,
            "y_train_pred": y_train_pred,
            "id": df_train_proc2.id,
        }
    )

    # Create traces
    fig = go.Figure()

    idx = df_plot.id == engine
    df_plot_filter = df_plot[idx]

    fig.add_trace(
        go.Bar(
            x=df_plot_filter["y_train"],
            y=df_plot_filter["y_train_pred"],
            name=f"Enigne {engine}",
        )
    )
    fig.add_shape(
        type="line",
        x0=10,
        y0=0,
        x1=10,
        y1=1,
        line=dict(width=4, dash="dot", color="red"),
    )
    fig.update_layout(
        title=f"Predicted probabilty of failure vs RUL engine id = {engine}",
        xaxis_title="RUL",
        yaxis_title="Predicted probability",
    )
    fig.update_xaxes(range=[0, 60])

    return fig
