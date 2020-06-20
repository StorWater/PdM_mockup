import pandas as pd
import numpy as np


def find_correlated_data(data_df, correlation_threshold):
    """Identifies column values with strong correlation to other column values.

    Parameters
    -----------
    data_df : pd.dataFrame)
        Dataframe containing data to be analysed.
    correlation_threshold : float
        Thershold value above which correlation is assumed.

   Returns
   --------
	data_with_correlation : tuple
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
	data_entities : list
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


def find_time_independent_columns(data_df):
    """
    Returns a list of columns that do not change with time.

    Parameters
    -----------
	data_df : dataFrame
        Dataframe containing time-series data.
        
   Returns
   --------
	unchanging_columns : list
        List of columns from dataframe which do not change with time.
    """

    unchanging_columns = []
    std_threshold = 0.0001

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
	df : pd.dataFrame
        Dataframe containing time-series data.
        
    Returns
	pd.dataFrame
        dataset, inclusive of calculated RUL values.
    """

    # RUL is negative and trends to zero (end of life point)
    df['RUL'] = df.groupby('id').apply(
            lambda x: x["cycle"].max() - x["cycle"]
        ).reset_index()['cycle']

    return df
