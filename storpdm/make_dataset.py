# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from typing import Union, Tuple
import requests, zipfile, io
import pandas as pd


def download_dataset(file_location: Union[str, Path] = "data/raw/"):
    """Download and unzips raw data

    Parameters
    ----------
    file_location : str or Path, default: "data/raw/"
        Location of the folder where data is stored, by default "data/raw/"
    """

    zip_file_url = "https://ti.arc.nasa.gov/c/6/"

    print("Downloading raw data...")
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(file_location)
    print(f"Done. Data downloaded at {file_location}")


def import_dataset(
    filename: Union[str, Path] = "FD001"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Import the dataset as a dataframe, adding column names.

    Parameters
    ----------
    filename : str or Path
        Name suffix of dataset to load. Must be one of the following values:
        ['FD0001','FD0002','FD0003','FD0004']
    
    Returns
    --------
        df_rul: pd.DataFrame
            Test results
        df_train : pd.DataFrame
            Training dataset
        df_test : pd.DataFrame
            Testing dataset
    """

    if filename not in ["FD001", "FD002", "FD003", "FD004"]:
        raise ValueError("Wrong filename.")

    dataset_columns = [
        "id",
        "cycle",
        "op1",
        "op2",
        "op3",
        "FanInletTemp",
        "LPCOutletTemp",
        "HPCOutletTemp",
        "LPTOutletTemp",
        "FanInletPres",
        "BypassDuctPres",
        "TotalHPCOutletPres",
        "PhysFanSpeed",
        "PhysCoreSpeed",
        "EnginePresRatio",
        "StaticHPCOutletPres",
        "FuelFlowRatio",
        "CorrFanSpeed",
        "CorrCoreSpeed",
        "BypassRatio",
        "BurnerFuelAirRatio",
        "BleedEnthalpy",
        "DemandFanSpeed",
        "DemandCorrFanSpeed",
        "HPTCoolantBleed",
        "LPTCoolantBleed",
    ]

    # Import the raw data into series of dataframes.
    df_rul = pd.read_csv(
        "data/raw/RUL_" + filename + ".txt",
        header=None,
        names=["rul"],
        delim_whitespace=True,
    )
    df_train = pd.read_csv(
        "data/raw/train_" + filename + ".txt",
        header=None,
        names=dataset_columns,
        delim_whitespace=True,
    )
    df_test = pd.read_csv(
        "data/raw/test_" + filename + ".txt",
        header=None,
        names=dataset_columns,
        delim_whitespace=True,
    )

    return df_rul, df_train, df_test


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
