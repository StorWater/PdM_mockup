import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def find_correlated_data(data_df, correlation_threshold):
    """Identifies column values with strong correlation to other column values.

    Parameters
    -----------
    data_df : pd.DataFrame)
        Dataframe containing data to be analysed.
    correlation_threshold : float
        Thershold value above which correlation is assumed.

   Returns
   --------
    tuple
        Tuple of two-column data sets, with correlation.
    """

    # Remove time column.
    modified_df = data_df.drop("cycle", axis=1)

    # Compute pairwise correlation of columns.
    data_corr = modified_df.corr()

    # Analyse data correlations; Keep values above threshold.
    high_correlation = []

    for column_no, column in enumerate(data_corr):
        # Create slice to ignore symmetry duplicates.
        col_corr = data_corr[column].iloc[column_no + 1 :]

        # Use a bool mask to identify highly-correlated data.
        mask_pairs = col_corr.apply(lambda x: abs(x)) > correlation_threshold

        index_pairs = col_corr[mask_pairs].index

        # Create list of highly-correlated data sets.
        for index, correlation in zip(index_pairs, col_corr[mask_pairs].values):
            high_correlation.append((column, index, correlation))

    return high_correlation


def list_correlated_data(correlated_data):
    """Creates a list of data entities from correlated data tuple.

    Parameters
    -----------
	correlated_data : tuple
        Tuple of data columns with high correlation.

   Returns
   --------
    list
        List of data columns correlated at least once.
    """

    data_list = []

    # Iterate over correlated data, add second value to list if not present.
    for correlation in correlated_data:

        # Data item to be removed is 2nd item in tuple.
        data_item = correlation[1]

        if data_list.__contains__(data_item):
            pass

        else:
            data_list.append(data_item)

    # Return list.
    return data_list


def find_time_independent_columns(data_df, std_threshold = 0.0001):
    """Returns a list of columns that do not change with time (i.e. almost zero
    standard deviation)

    Parameters
    -----------
	data_df : DataFrame
        Dataframe containing time-series data.
    std_threshold : float, default 0.0001
        Filter columns with std lower than this threshold

   Returns
   --------
    list
        List of columns from dataframe which do not change with time.
    """

    unchanging_columns = []

    # Iterate over columns; Identify if std is close-coupled to mean.
    for column in data_df.columns:

        if data_df[column].std() <= std_threshold * data_df[column].mean():

            if unchanging_columns.__contains__(column):
                pass

            # Add tightly-coupled columns to list.
            else:
                unchanging_columns.append(column)

    return unchanging_columns


def add_calculated_rul(df):
    """Calculates Remaining Useful Life (RUL) and adds it to dataset, which is returned.
    
    Parameters
    -----------
	df : pd.DataFrame
        Dataframe containing time-series data.
        
    Returns
	pd.DataFrame
        dataset, inclusive of calculated RUL values.
    """

    # RUL is negative and trends to zero (end of life point)
    df["RUL"] = (
        df.groupby("id")
        .apply(lambda x: x["cycle"].max() - x["cycle"])
        .reset_index()["cycle"]
    )

    return df


def prepare_training_data(df, target_col, scaled=False, discard_cols = None, test_size=0.2
):
    """Prepare and return training and test data arrays from input dataframe, 
    normalising using 
    
    Parameters
    -----------
    df : pd.DataFrame
        Dataframe containing training dataset.
    target_col : str
        Target value for model training.
    discard_cols : str, list of str, default None
        Determines which column to drop
    
    Returns
    --------
    X_train : pd.DataFrame
        Training feature columns
    X_test : pd.DataFrame
        Testing features columns
    y_train : pd.DataFrame
        Training target column
    y_test  : pd.DataFrame 
        Testing target column
    """

    if discard_cols:
        df = df.drop(discard_cols, axis=1)


    # Split X and y
    X = df.drop(target_col, axis=1)
    y = df[target_col].values

    if scaled:
        scalar = StandardScaler()  # alternative: MinMaxScalar() TODO
        X = scalar.fit_transform(X)

    # Create split between training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    return X_train, X_test, y_train, y_test
