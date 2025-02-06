"""
Methylation Rate Analysis Module

This module provides functions for calculating and normalizing DNA methylation rates
from time-series methylation data stored in SQLite databases. It processes experimental
data across different samples, replicates, and genomic regions to compute both raw
and normalized methylation rates.

Key Features
-----------
- Calculate median methylation fractions for normalization
- Compute genome-wide methylation rates
- Normalize rates using specified genomic regions (e.g., mitochondrial DNA)
- Process data across multiple experiments and replicates
- Export results to both SQLite database and CSV formats

Main Functions
-------------
calc_median_for_normalization:
    Calculate median methylated fraction for normalization reference
calc_median_rate_for_normalization:
    Calculate median methylation rate for normalization
calc_rate_genome:
    Calculate genome-wide methylation rates normalized to median rates
calculate_rate:
    Main entry point for methylation rate calculation from config file

Dependencies
-----------
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- sqlite3: Database operations
- logging: Logging functionality
- os: File and directory operations

Database Schema
--------------
Required tables:
- SampleSheet: Contains experiment metadata
- {experiment}_{replicate}_combine: Raw methylation data
- Sample_Median_combine: Calculated median values
- Sample_Median_Rate_combine: Calculated median rates
- {experiment}_{replicate}_rate: Final normalized rates

Example Usage
------------
>>> config_path = "path/to/config.toml"
>>> calculate_rate(config_path)

Notes
-----
- Input data should be properly formatted in SQLite database
- Normalization can be performed on whole genome or specific regions
- Default normalization uses mitochondrial DNA (on='Mito')
- Uses 0.99 as default cutoff for rate calculations

See Also
--------
load_config_file : Function for parsing configuration files
calc_methylation_rate : Function for calculating methylation rates

Author: Zhuwei Xu
Version: 1.0.0
Date: 25-02-03
"""

import logging
import sqlite3
import os
from sinekit.utility import (calc_methylation_rate, load_config_file)

import numpy as np
import pandas as pd

logger = logging.getLogger("sinekit")

def calc_median_for_normalization(
    sample_pd: pd.DataFrame,
    data_con: sqlite3.Connection,
    on: str = 'Mito'
) -> pd.DataFrame:
    """
    Calculate median methylated fraction for normalization.
    
    Parameters
    ----------
    sample_pd : pd.DataFrame
        DataFrame containing sample information with columns:
        ['experiment', 'rep', 'time']
    data_con : sqlite3.Connection
        SQLite database connection containing combined data tables
    on : str, default='Mito'
        Chromosome/region to calculate median on. Use 'Genome' for whole genome
        or specific chromosome name (e.g., 'Mito' for mitochondrial DNA)
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing median methylation values with columns:
        ['Experiment', 'Rep', 'On'] + timepoints
        Also saves results to 'Sample_Median_combine' table in database
    
    Raises
    ------
    RuntimeError
        If specified chromosome is not found in the database
    """
    median_pd: list = []
    
    if on == 'Genome':
        for (exp_, rep), tmp_pd in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
            data_points = np.sort(tmp_pd['time'].values)
            data_pd = pd.read_sql_query(
                f'SELECT * FROM "{exp_}_{rep}_combine"',
                data_con
            )
            tmp = [exp_, rep, 'genome']
            for da in data_points:
                med = np.nanmedian(data_pd[str(da)])
                tmp.append(med)
            median_pd.append(tmp)
    else:
        for (exp_, rep), tmp_pd in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
            data_points = np.sort(tmp_pd['time'].values)
            data_pd = pd.read_sql_query(
                f'SELECT * FROM "{exp_}_{rep}_combine" WHERE CHROM = "{on}"',
                data_con
            )
            if len(data_pd) == 0:
                msg = f"{on} is not found in Table {exp_}_{rep}_combine"
                logger.error(msg)
                raise RuntimeError(msg)
            tmp = [exp_, rep, on]
            for da in data_points:
                med = np.nanmedian(data_pd[str(da)])
                tmp.append(med)
            median_pd.append(tmp)
            
    median_pd = pd.DataFrame(
        median_pd,
        columns=['Experiment', 'Rep', 'On'] + [str(da) for da in data_points]
    )
    logger.info("Median Methylated Fraction of %s is calculated", on)
    cursor = data_con.cursor()
    cursor.execute("DROP TABLE IF EXISTS Sample_Median_combine")
    data_con.commit()
    cursor.close()
    median_pd.to_sql("Sample_Median_combine", data_con, if_exists='replace', index=False)
    return median_pd


def calc_median_rate_for_normalization(
    sample_pd: pd.DataFrame,
    data_con: sqlite3.Connection,
    on: str = 'Mito',
    cutoff: float = 0.99
) -> pd.DataFrame:
    """
    Calculate median methylation rate for normalization.
    
    Parameters
    ----------
    sample_pd : pd.DataFrame
        DataFrame containing sample information
    data_con : sqlite3.Connection
        SQLite database connection
    on : str, default='Mito'
        Chromosome/region to calculate median on
    cutoff : float, default=0.99
        Cutoff value for rate calculation
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing median methylation rates
        Also saves results to 'Sample_Median_Rate_combine' table in database
    """
    median_pd = calc_median_for_normalization(
        sample_pd=sample_pd,
        data_con=data_con,
        on=on
    )
    data_points = np.array([int(x) for x in (median_pd.columns[3:])])
    median_pd.rename(columns={'On': 'Pos', 'Rep': 'Chrom'}, inplace=True)
    
    rate_pd = calc_methylation_rate(
        data_pd=median_pd,
        data_points=data_points,
        cutoff=cutoff
    )
    
    rate_pd.rename(columns={'Chrom': 'Rep', 'Pos': 'On'}, inplace=True)
    rate_pd['Experiment'] = median_pd['Experiment']
    columns = list(rate_pd.columns)
    rate_pd = rate_pd[[columns[-1]] + columns[:-1]]
    
    logger.info("Median Methylation Rate of %s is calculated", on)
    rate_pd.to_sql("Sample_Median_Rate_combine", data_con, if_exists='replace', index=False)
    return rate_pd


def calc_rate_genome(
    data_con: sqlite3.Connection,
    cutoff: float = 0.99
) -> None:
    """
    Calculate genome-wide methylation rates normalized to median rates.
    
    Parameters
    ----------
    data_con : sqlite3.Connection
        SQLite database connection
    cutoff : float, default=0.99
        Cutoff value for rate calculation
    
    Returns
    -------
    None
        Creates tables named '{experiment}_{replicate}_rate' in database
        containing normalized methylation rates
    """
    norm_rate_pd = pd.read_sql_query('SELECT * from "Sample_Median_Rate_combine"', data_con)
    
    for _, row in norm_rate_pd.iterrows():
        exp_ = row['Experiment']
        rep = row['Rep']
        norm_rate = row["Rate"]
        col = list(norm_rate_pd.columns)
        data_points = np.array([int(x) for x in col[6:]])
        
        tmp_pd = pd.read_sql_query(f'SELECT * from "{exp_}_{rep}_combine"', data_con)
        tmp_rate_pd = calc_methylation_rate(
            data_pd=tmp_pd,
            data_points=data_points,
            cutoff=cutoff
        )
        
        tmp_rate_pd['Relative Rate'] = tmp_rate_pd['Rate'] / norm_rate
        cols = list(tmp_rate_pd.columns)
        cols = cols[:2] + ['Relative Rate'] + cols[2:-1]
        tmp_rate_pd = tmp_rate_pd[cols]
        
        tmp_rate_pd.to_sql(f"{exp_}_{rep}_rate", data_con, if_exists='replace', index=False)
        logger.info("Methylation rate of %s %s is calculated.", exp_, rep)


def calculate_rate(config: str) -> None:
    """
    Calculate methylation rates from configuration file.
    
    This function serves as the main entry point for methylation rate calculation.
    It reads configuration, sets up output directories, and orchestrates the
    calculation of methylation rates across the genome.
    
    Parameters
    ----------
    config : str
        Path to configuration file in TOML format
    
    Returns
    -------
    None
        Creates output files and database tables with methylation rate calculations
    
    Raises
    ------
    RuntimeError
        If required SQLite database is not found
    """
    config_dict = load_config_file(config)
    output_folder = config_dict["output"]["output_folder"]
    os.makedirs(output_folder, exist_ok=True)
    prefix = config_dict["output"]["prefix"]
    data_db = os.path.join(output_folder, f'{prefix}.SQLite.db')
    normalize_on = config_dict['parameter']["normalization"]
    normalize_fn = os.path.join(output_folder, f'{prefix}.Rate_Norm.on={normalize_on}.csv')
    
    if not os.path.isfile(data_db):
        msg = f'Cannot found the SQLite database {data_db}\nPlease run the load command first.'
        logger.error(msg)
        raise RuntimeError(msg)

    data_con = sqlite3.connect(data_db)
    sample_pd = pd.read_sql_query('select * from SampleSheet', data_con)

    rate_pd = calc_median_rate_for_normalization(
        sample_pd=sample_pd,
        data_con=data_con,
        on=normalize_on,
        cutoff=0.99
    )
    rate_pd.to_csv(normalize_fn, index=False)
    logger.info('Median Rate for Normalization saved in %s', normalize_fn)
    
    calc_rate_genome(data_con)
    logger.info('Calculation of Methylation Rates is successful.')
    data_con.close()