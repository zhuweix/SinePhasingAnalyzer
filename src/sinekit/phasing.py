"""
SineKit Analysis Module

A Python module for analyzing gene expression and methylation data using sine wave fitting.
This module provides tools to process and analyze genomic data, particularly focusing on
periodic patterns in gene expression and methylation levels.

Key Features:
-------------
- Sine wave fitting for gene expression data
- Automated and user-defined quantile analysis
- Support for both rate and methylation fraction analyses
- SQLite database integration for result storage
- CSV export functionality for analysis results

Main Components:
---------------
- prepare_results_for_csv: Formats sine wave fitting results for CSV export
- sinefit_gene_average: Performs sine wave fitting on averaged gene data
- calc_adj_gene_with_quantile: Calculates adjusted gene levels and assigns quantiles
- sine_fit_gene_quantile: Performs quantile-based sine wave analysis
- sinekit_analysis: Main function orchestrating the complete analysis pipeline

Dependencies:
------------
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- sqlite3: Database operations
- logging: Logging functionality
- pickle: Object serialization
- gc: Garbage collection for memory management

Typical Usage:
-------------
>>> config_file = "analysis_config.yaml"
>>> sinekit_analysis(config_file)

For custom quantile analysis:
>>> sinekit_analysis(config_file, user_quantile_fn="custom_quantiles.csv")

Data Requirements:
----------------
- Gene expression/methylation measurements
- Configuration file with analysis parameters
- Optional: User-defined quantile assignments

Output:
-------
- CSV files with analysis results
- SQLite database with processed data
- Pickle files containing raw fitting results
- Directory structure organizing different output types

Notes:
-----
This module is designed for genomic data analysis with a focus on periodic patterns.
It assumes input data follows specific formats and provides comprehensive error checking
for data validity. Memory management is handled through strategic garbage collection
during intensive computations.

Author: [Your Name]
Version: 1.0.0
License: [License Type]
"""

import logging
import sqlite3
import os
import gc
import pickle
from sinekit.utility import (
    load_sample_sheet, load_config_file, 
    get_chrom_sizes, calc_sine_fit, calculate_adj_gene_level)
import pandas as pd
import numpy as np

logger = logging.getLogger("sinekit")


def prepare_results_for_csv(result_dict: dict, analysis: str) -> pd.DataFrame:
    """
    Converts a dictionary of sine fitting results into a formatted pandas DataFrame suitable for CSV export.
    
    Parameters
    ----------
    result_dict : dict
        Dictionary containing sine fitting results. Each key is a tuple of (experiment, replicate, 
        [extract Time column for methylation fraction]), and each value is a dictionary with the 
        following required keys:
            - Spacing: float, calculated spacing between peaks
            - Error_spacing: float, error in spacing calculation
            - Amplitude: float, amplitude of fitted sine wave
            - Error_Amp: float, error in amplitude calculation
            - Slope: float, linear slope component
            - Error_Slope: float, error in slope calculation
            - Adj.R2: float, adjusted R-squared value of fit
            - Decay: float, decay rate per period
            - theta0: float, phase offset
            - b0: float, baseline offset
    
    analysis : str
        Type of analysis being performed. Must be one of:
            - 'Rate': Basic rate analysis
            - 'Frac': Methylation fraction analysis
            - 'RateQuantile': Rate analysis with quantiles
            - 'FracQuantile': Methylation fraction analysis with quantiles
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
            - Experiment: experiment identifier (str)
            - Rep: replicate identifier (str)
            [- Time: Only for methylated fraction analysis]
            [- Quantile: Only for quantile analyses]
            - Spacing (bp): peak spacing in base pairs (float, 1 decimal)
            - Spacing Error (bp): error in spacing (float, 1 decimal)
            - Amplitude: sine wave amplitude (float, 3 decimals)
            - Amplitude Error: error in amplitude (float, 3 decimals)
            - Slope (per kb): linear slope per kilobase (float, 2 decimals)
            - Slope Error (per kb): error in slope (float, 2 decimals)
            - Adjusted R2: goodness of fit (float, 3 decimals)
            - Decay per Period: decay rate (float, 3 decimals)
            - Phase (theta0): phase offset (float, 2 decimals)
            - Baseline (b0): baseline offset (float, 3 decimals)
    
    Raises
    ------
    ValueError
        If the analysis parameter is not one of the valid analysis types.
    
    Notes
    -----
    The function formats numeric values with specific decimal precision for readability
    and consistency in the output CSV file. Different column sets are used depending
    on the analysis type specified.
    """

    results_pd = []
    valid_analyses = {'Rate', 'Frac', 'RateQuantile', 'FracQuantile', 'RateUserQuantile', 'FracUserQuantile'}
    if analysis not in valid_analyses:
        logger.error('Invalid analysis type: %s. Must be within %s', analysis, valid_analyses)
        raise ValueError
    for ent, tmp_dict in result_dict.items():
        row = [str(e) for e in ent]
        da = [
            f"{tmp_dict['Spacing']:.1f}", 
            f"{tmp_dict['Error_spacing']:.1f}",
            f"{tmp_dict['Amplitude']:.3f}",
            f"{tmp_dict['Error_Amp']:.3f}",
            f"{tmp_dict['Slope']:.2f}",
            f"{tmp_dict['Error_Slope']:.2f}",
            f"{tmp_dict['Adj.R2']:.3f}",
            f"{tmp_dict['Decay']:.3f}",
            f"{tmp_dict['theta0']:.2f}",
            f"{tmp_dict['b0']:.3f}"
        ]
        row.extend(da)
        results_pd.append(row)
        
    da_columns = [
            "Spacing (bp)",
            "Spacing Error (bp)",
            "Amplitude",
            "Amplitude Error",
            "Slope (per kb)",
            "Slope Error (per kb)",
            "Adjusted R2",
            "Decay per Period",
            "Phase (theta0)",
            "Baseline (b0)"
        ]
    columns = []
    if analysis == 'Rate':
        columns = ['Experiment', 'Rep'] + da_columns
    elif analysis == 'Frac':
        columns = ['Experiment', 'Rep', 'Time'] + da_columns
    elif analysis == 'RateQuantile':
        columns = ['Experiment', 'Rep', 'Quantile'] + da_columns
    elif analysis == 'FracQuantile':
        columns = ['Experiment', 'Rep', 'Time', 'Quantile'] + da_columns
    elif analysis == 'RateUserQuantile':
        columns = ['GroupName', 'Experiment', 'Rep', 'Quantile'] + da_columns        
    elif analysis == 'FracUserQuantile':
        columns = ['GroupName', 'Experiment', 'Rep', 'Time', 'Quantile'] + da_columns        
    results_pd = pd.DataFrame(results_pd, columns=columns)
    return results_pd


def sinefit_gene_average(gene_raw_pd: pd.DataFrame,
                        analysis: str = "Rate",
                        xmin: int = -50,
                        xmax: int = 1000) -> tuple:
    """
    Performs sine wave fitting analysis on gene methylation data with flexible grouping options.
    
    This function processes methylation data by filtering positions within a specified range
    and performs sine wave fitting on the values. It supports two analysis modes:
    'Rate' for standard experiment-replicate grouping, and 'Frac' for including time points.
    
    Parameters
    ----------
    gene_raw_pd : pd.DataFrame
        Input DataFrame containing methylation data.
        Required columns for 'Rate' analysis:
            - 'Pos': Position values (will be converted to integers)
            - 'Value': Methylation values
            - 'Experiment': Experiment identifiers
            - 'Rep': Replicate identifiers
        Additional required column for 'Frac' analysis:
            - 'Time': Time point identifiers
    
    analysis : str, optional
        Type of analysis to perform (default: "Rate")
        Valid values:
            - "Rate": Groups by experiment and replicate
            - "Frac": Groups by experiment, replicate, and time point
    
    xmin : int, optional
        Minimum position value to include in analysis (default: -50)
    
    xmax : int, optional
        Maximum position value to include in analysis (default: 1000)
    
    Returns
    -------
    tuple
        A tuple containing two elements:
        - result_pd : pd.DataFrame
            Processed results formatted for CSV export
        - result_dict : dict
            For 'Rate' analysis: Dictionary keyed by (experiment, replicate) tuples
            For 'Frac' analysis: Dictionary keyed by (experiment, replicate, time) tuples
            Each value contains the sine wave fitting parameters and statistics
    
    Notes
    -----
    The function filters positions within the specified range and performs sine wave
    fitting using the `calc_sine_fit` function. The grouping structure depends on
    the selected analysis type. An error is logged if an invalid analysis type is provided.
    """
    valid_analyses = {'Rate', 'Frac'}
    if analysis not in valid_analyses:
        logger.error('Invalid analysis type: %s. Must be either "Rate" or "Frac"', analysis)
        raise ValueError
    result_dict = {}
    if analysis == 'Rate':
        for (exp_, rep), tmp_pd in gene_raw_pd.groupby(
                by=['Experiment', 'Rep'], sort=False):
            tmp_pd['Pos'] = tmp_pd['Pos'].values.astype(int)
            tmp_pd = tmp_pd.loc[(tmp_pd['Pos'] >= xmin) & (tmp_pd['Pos'] <= xmax)].copy()
            tmp_pd.sort_values(by='Pos', inplace=True)
            
            # Perform fitting
            ydata = tmp_pd['Value'].values
            xdata = tmp_pd['Pos'].values
            tmp_dict = calc_sine_fit(ydata, xdata)
            result_dict[(exp_, rep)] = tmp_dict
    elif analysis == 'Frac':
        for (exp_, rep, time_), tmp_pd in gene_raw_pd.groupby(
                by=['Experiment', 'Rep', 'Time'], sort=False):
            tmp_pd['Pos'] = tmp_pd['Pos'].values.astype(int)
            tmp_pd = tmp_pd.loc[(tmp_pd['Pos'] >= xmin) & (tmp_pd['Pos'] <= xmax)].copy()
            tmp_pd.sort_values(by='Pos', inplace=True)
            
            # Perform fitting
            ydata = tmp_pd['Value'].values
            xdata = tmp_pd['Pos'].values
            tmp_dict = calc_sine_fit(ydata, xdata)
            result_dict[(exp_, rep, time_)] = tmp_dict
    result_pd = prepare_results_for_csv(result_dict, analysis)
    return result_pd, result_dict
    
    
def calc_adj_gene_with_quantile(
        gene_ind_pd: pd.DataFrame,
        fit_dict: dict,
        quantile: int,
        analysis: str,
        xmin: int,
        xmax: int) -> pd.DataFrame:
    """
    Calculates adjusted gene expression levels and assigns them to quantiles based on fitted parameters,
    supporting both standard rate analysis and time-series fraction analysis.
    
    Parameters
    ----------
    gene_ind_pd : pd.DataFrame
        Input DataFrame containing gene expression data.
        Required columns for all analyses:
            - Experiment: Experiment identifiers
            - Rep: Replicate identifiers
            - Gene: Gene identifiers
            - Pos: Position values
            - Value: Expression values
        Additional required column for 'Frac' analysis:
            - Time: Time point identifiers
    
    fit_dict : dict
        Dictionary containing fitted parameters.
        For 'Rate' analysis: Keys are (experiment, replicate) tuples
        For 'Frac' analysis: Keys are (experiment, replicate, time) tuples
        Each value must contain a 'fit_params' key with the fitted parameters.
    
    quantile : int
        Number of quantiles to divide the adjusted gene levels into.
        Must be a positive integer.
    
    analysis : str
        Type of analysis to perform.
        Valid values:
            - "Rate": Standard analysis grouping by experiment and replicate
            - "Frac": Time-series analysis grouping by experiment, replicate, and time point
    
    xmin : int
        Minimum position value to include in calculations.
    
    xmax : int
        Maximum position value to include in calculations.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing adjusted gene expression levels and quantile assignments.
        For 'Rate' analysis, columns include:
            - Experiment: experiment identifier
            - Rep: replicate identifier
            - Gene: gene identifier
            - Adj.Average: adjusted expression level
            - R2: R-squared value for the adjustment
            - Quantile: assigned quantile (Q1, Q2, etc.)
        For 'Frac' analysis, includes all above columns plus:
            - Time: time point identifier
    
    Notes
    -----
    The function processes data through several steps:
    1. Validates input parameters and analysis type
    2. Filters positions within the specified range
    3. Groups data according to the analysis type
    4. Calculates adjusted gene levels using fitted parameters
    5. Assigns genes to quantiles based on their adjusted levels within each group
    
    The grouping structure and output format adapt automatically based on the selected
    analysis type, allowing for both standard rate analysis and time-series studies.
    """
    if not isinstance(quantile, int) or quantile <= 0:
        raise ValueError("quantile must be a positive integer")
    valid_analyses = {'Rate', 'Frac'}
    if analysis not in valid_analyses:
        logger.error('Invalid analysis type: %s. Must be either "Rate" or "Frac"', analysis)
        raise ValueError    
    gene_ind_pd = gene_ind_pd.loc[(gene_ind_pd['Pos'] >= xmin) & 
                                 (gene_ind_pd['Pos'] <= xmax)]
    gene_adj_pd = []
    if analysis == 'Rate':
        for (exp_, rep), tmp_pd in gene_ind_pd.groupby(by=['Experiment', 'Rep'], sort=False):
            tmp_rate_pd = []
            fit_params = fit_dict[(exp_, rep)]['fit_params']
            
            for gene, tmp_pd2 in tmp_pd.groupby(by='Gene', sort=False):
                y = tmp_pd2['Value'].values
                xpos = tmp_pd2['Pos'].values
                adj_rate, r2 = calculate_adj_gene_level(y, xpos, fit_params)
                tmp_rate_pd.append((exp_, rep, gene, adj_rate, r2))
                
            tmp_rate_pd = pd.DataFrame(tmp_rate_pd, columns=[
                'Experiment', 'Rep', 'Gene', 'Adj.Average', 'R2'
            ])
            
            # Create quantile labels and assign them
            q = pd.qcut(tmp_rate_pd['Adj.Average'], 
                    quantile, 
                    labels=[f'Q{i+1}' for i in range(quantile)])
            tmp_rate_pd['Quantile'] = q
            gene_adj_pd.append(tmp_rate_pd)
    elif analysis == 'Frac':
        for (exp_, rep, time_), tmp_pd in gene_ind_pd.groupby(by=['Experiment', 'Rep', 'Time'], sort=False):
            tmp_rate_pd = []
            fit_params = fit_dict[(exp_, rep)]['fit_params']
            
            for gene, tmp_pd2 in tmp_pd.groupby(by='Gene', sort=False):
                y = tmp_pd2['Value'].values
                xpos = tmp_pd2['Pos'].values
                adj_rate, r2 = calculate_adj_gene_level(y, xpos, fit_params)
                tmp_rate_pd.append((exp_, rep, time_, gene, adj_rate, r2))
                
            tmp_rate_pd = pd.DataFrame(tmp_rate_pd, columns=[
                'Experiment', 'Rep', 'Time', 'Gene', 'Adj.Average', 'R2'
            ])
            
            # Create quantile labels and assign them
            q = pd.qcut(tmp_rate_pd['Adj.Average'], 
                    quantile, 
                    labels=[f'Q{i+1}' for i in range(quantile)])
            tmp_rate_pd['Quantile'] = q
            gene_adj_pd.append(tmp_rate_pd)        

    gene_adj_pd = pd.concat(gene_adj_pd, ignore_index=True)
    
    return gene_adj_pd


def sine_fit_gene_quantile(
    gene_ind_pd: pd.DataFrame,
    gene_adj_pd: pd.DataFrame,
    quantile: int = 5,
    analysis: str = 'Rate',
    xmin: int = -50,
    xmax: int = 1000,
    is_user: bool = False,
) -> tuple:
    """
    Performs sine wave fitting analysis on gene expression data grouped by quantiles.
    
    Parameters
    ----------
    gene_ind_pd : pd.DataFrame
        DataFrame containing individual gene measurements with columns:
        - Experiment: experiment identifier
        - Rep: replicate identifier
        - Gene: gene identifier
        - Pos: position relative to gene
        - Value: measurement value
        - Time: time point (only required for Frac analysis)
        
    gene_adj_pd : pd.DataFrame
        DataFrame containing gene assignments to quantiles with columns:
        - Experiment: experiment identifier
        - Rep: replicate identifier
        - Gene: gene identifier
        - Quantile: quantile assignment
        - Time: time point (only required for Frac analysis)
        
    quantile : int, optional
        Number of quantiles to analyze. Must be a positive integer.
        Default is 5.
        
    analysis : str, optional
        Type of analysis to perform. Must be either 'Rate' or 'Frac'.
        - 'Rate': Analyzes rate measurements
        - 'Frac': Analyzes fraction measurements with time points
        Default is 'Rate'.
        
    xmin : int, optional
        Minimum position to include in analysis, relative to gene.
        Default is -50.
        
    xmax : int, optional
        Maximum position to include in analysis, relative to gene.
        Default is 1000.
        
    Returns
    -------
    tuple
        Contains three elements:
        1. pd.DataFrame: Processed quantile data with columns:
           - Experiment, Rep, Quantile, Pos, Value
           - Time (only for Frac analysis)
        2. pd.DataFrame: Fitting results formatted for CSV export
        3. dict: Raw fitting results dictionary
    
    Raises
    ------
    ValueError
        - If quantile is not a positive integer
        - If analysis is not 'Rate' or 'Frac'
    
    Notes
    -----
    The function processes gene expression data by:
    1. Filtering data to specified position range
    2. Grouping genes by quantiles
    3. Calculating average values within each quantile
    4. Performing sine wave fitting on the averaged data
    5. Preparing results in multiple formats
    
    The fitting is performed using the calc_sine_fit function and results
    are formatted using prepare_results_for_csv.
    """
    if not isinstance(quantile, int) or quantile <= 0:
        raise ValueError("quantile must be a positive integer")
    
    valid_analyses = {'Rate', 'Frac'}
    if analysis not in valid_analyses:
        logger.error('Invalid analysis type: %s. Must be one of %s', analysis, valid_analyses)
        raise ValueError(f"Analysis type must be one of {valid_analyses}")

    gene_ind_pd = gene_ind_pd.loc[(gene_ind_pd['Pos'] >= xmin) &
                                 (gene_ind_pd['Pos'] <= xmax)]
    # extract quantile average data
    gene_quantile_pd = []
    result_dict = {}
    if not is_user:
        if analysis == 'Rate':
            for (exp_, rep, q), tmp_gene_pd in gene_adj_pd.groupby(
                by=['Experiment', 'Rep', 'Quantile'], sort=False):
                gene_set = set(tmp_gene_pd['Gene'].values)
                tmp_data_pd = gene_ind_pd.loc[(gene_ind_pd['Experiment'] == exp_) & 
                    (gene_ind_pd['Rep'] == rep) & (gene_ind_pd['Gene'].isin(gene_set))]
                tmp_data = np.zeros(xmax-xmin+1)
                tmp_cov = np.zeros(xmax-xmin+1)
                for p, v in zip(tmp_data_pd['Pos'], tmp_data_pd['Value']):
                    tmp_data[p - xmin] += v
                    tmp_cov[p - xmin] += 1
                valid = tmp_cov > 0
                tmp_data[valid] /= tmp_cov[valid]
                pos = np.where(valid)[0] + xmin
                value = tmp_data[valid]
                tmp_ent = pd.DataFrame({
                    'Experiment': exp_,
                    'Rep': rep,
                    'Quantile': q,
                    'Pos': pos,
                    'Value': value
                })
                # Perform fitting
                ydata = tmp_ent['Value'].values
                xdata = tmp_ent['Pos'].values
                tmp_dict = calc_sine_fit(ydata, xdata)
                result_dict[(exp_, rep, q)] = tmp_dict
                gene_quantile_pd.append(tmp_ent)
                gc.collect()
        elif analysis == 'Frac':
            for (exp_, rep, time_, q), tmp_gene_pd in gene_adj_pd.groupby(
                by=['Experiment', 'Rep', 'Time', 'Quantile'], sort=False):
                gene_set = set(tmp_gene_pd['Gene'].values)
                tmp_data_pd = gene_ind_pd.loc[(gene_ind_pd['Experiment'] == exp_) & 
                    (gene_ind_pd['Rep'] == rep) & (gene_ind_pd['Gene'].isin(gene_set)) & 
                    (gene_ind_pd['Time'] == time_)]
                tmp_data = np.zeros(xmax-xmin+1)
                tmp_cov = np.zeros(xmax-xmin+1)
                for p, v in zip(tmp_data_pd['Pos'], tmp_data_pd['Value']):
                    tmp_data[p - xmin] += v
                    tmp_cov[p - xmin] += 1
                valid = tmp_cov > 0
                tmp_data[valid] /= tmp_cov[valid]
                pos = np.where(valid)[0] + xmin
                value = tmp_data[valid]
                tmp_ent = pd.DataFrame({
                    'Experiment': exp_,
                    'Rep': rep,
                    'Time': time_,
                    'Quantile': q,
                    'Pos': pos,
                    'Value': value
                })

                # Perform fitting
                ydata = tmp_ent['Value'].values
                xdata = tmp_ent['Pos'].values
                tmp_dict = calc_sine_fit(ydata, xdata)
                result_dict[(exp_, rep, time_, q)] = tmp_dict
                gene_quantile_pd.append(tmp_ent)
                gc.collect()
    else:
        if analysis == 'Rate':
            for (name, q), tmp_gene_pd in gene_adj_pd.groupby(
                by=['GroupName', 'Quantile'], sort=False):
                gene_set = set(tmp_gene_pd['Gene'].values)
                tmp_data_pd = gene_ind_pd.loc[gene_ind_pd['Gene'].isin(gene_set)]
                for (exp_, rep), tmp_data_pd2 in tmp_data_pd.groupby(by=['Experiment', 'Rep'], sort=False):
                    tmp_data = np.zeros(xmax-xmin+1)
                    tmp_cov = np.zeros(xmax-xmin+1)
                    for p, v in zip(tmp_data_pd2['Pos'], tmp_data_pd2['Value']):
                        tmp_data[p - xmin] += v
                        tmp_cov[p - xmin] += 1
                    valid = tmp_cov > 0
                    tmp_data[valid] /= tmp_cov[valid]
                    pos = np.where(valid)[0] + xmin
                    value = tmp_data[valid]
                    tmp_ent = pd.DataFrame({
                        'GroupName': name,
                        'Experiment': exp_,
                        'Rep': rep,
                        'Quantile': q,
                        'Pos': pos,
                        'Value': value
                    })
                    # Perform fitting
                    ydata = tmp_ent['Value'].values
                    xdata = tmp_ent['Pos'].values
                    tmp_dict = calc_sine_fit(ydata, xdata)
                    result_dict[(name, exp_, rep, q)] = tmp_dict
                    gene_quantile_pd.append(tmp_ent)
                    gc.collect()
        elif analysis == 'Frac':
            for (name, q), tmp_gene_pd in gene_adj_pd.groupby(
                by=['GroupName', 'Quantile'], sort=False):
                gene_set = set(tmp_gene_pd['Gene'].values)
                tmp_data_pd = gene_ind_pd.loc[gene_ind_pd['Gene'].isin(gene_set)]
                for (exp_, rep), tmp_data_pd2 in tmp_data_pd.groupby(by=['Experiment', 'Rep'], sort=False):
                    tmp_data = np.zeros(xmax-xmin+1)
                    tmp_cov = np.zeros(xmax-xmin+1)
                    for p, v in zip(tmp_data_pd['Pos'], tmp_data_pd['Value']):
                        tmp_data[p - xmin] += v
                        tmp_cov[p - xmin] += 1
                    valid = tmp_cov > 0
                    tmp_data[valid] /= tmp_cov[valid]
                    pos = np.where(valid)[0] + xmin
                    value = tmp_data[valid]
                    tmp_ent = pd.DataFrame({
                        'GroupName': name,
                        'Experiment': exp_,
                        'Rep': rep,
                        'Time': time_,
                        'Quantile': q,
                        'Pos': pos,
                        'Value': value
                    })

                    # Perform fitting
                    ydata = tmp_ent['Value'].values
                    xdata = tmp_ent['Pos'].values
                    tmp_dict = calc_sine_fit(ydata, xdata)
                    result_dict[(name, exp_, rep, time_, q)] = tmp_dict
                    gene_quantile_pd.append(tmp_ent)
                    gc.collect()        
    gene_quantile_pd = pd.concat(gene_quantile_pd, ignore_index=True)
    if not is_user:
        result_pd = prepare_results_for_csv(result_dict, analysis + 'Quantile')
    else:
        result_pd = prepare_results_for_csv(result_dict, analysis + 'UserQuantile')

    return gene_quantile_pd, result_pd, result_dict

def sinekit_analysis(config: str, user_quantile_fn=None):
    """
    Performs sine wave analysis on gene expression data with optional quantile-based grouping.
    
    Args:
        config (str): Path to configuration file containing analysis parameters
        user_quantile_fn (str, optional): Path to user-provided quantile assignments file
        
    Raises:
        RuntimeError: If required database files or tables are missing, or if user quantile
                     file is missing required columns
    """
    # Load configuration and input files
    config_dict = load_config_file(config)

    # Set up output directory structure
    prefix = config_dict["output"]["prefix"]
    output_folder = config_dict["output"]["output_folder"]    
    csv_folder = os.path.join(output_folder, 'Spreadsheet')
    os.makedirs(csv_folder, exist_ok=True)
    fit_folder = os.path.join(output_folder, 'FitResult')
    os.makedirs(fit_folder, exist_ok=True)    

    # Extract analysis parameters
    num_quantile = int(config_dict["parameter"]["num_quantile"])
    xmin = int(config_dict["parameter"]["xmin"])
    xmax = int(config_dict["parameter"]["xmax"])

    # Connect to SQLite database
    data_db = os.path.join(output_folder, f'{prefix}.SQLite.db')
    if not os.path.isfile(data_db):
        logger.error('Cannot find database file %s, please run load command first', data_db)
        raise RuntimeError

    data_con = sqlite3.connect(data_db)

    # Process average gene rates
    try:
        gene_raw_pd = pd.read_sql_query('select * from "gene_1010bp_average_rate"', data_con)
    except:
        logger.error('Error loading gene average rate from gene_1010bp_average_rate')
        raise RuntimeError

    # Fit sine wave to average gene rates
    gene_average_pd, gene_average_dict = sinefit_gene_average(
        gene_raw_pd, "Rate", xmin, xmax
    )
    logger.info('sine fit done')
    # Save average gene rate results
    csv_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageRate.SineWave.result.csv')
    pkl_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageRate.SineWave.pkl')    
    gene_average_pd.to_csv(csv_fn, index=False)
    gene_average_pd.to_sql('SineWave_Gene_AverageRate_Fit', data_con, index=False, if_exists='replace')
    with open(pkl_fn, 'wb') as filep:
        pickle.dump(gene_average_dict, filep)
    logger.info('Sine Wave analysis of average gene phasing is successful.')        

    # Process individual gene rates
    try:
        gene_ind_pd = pd.read_sql_query('select * from "gene_1010bp_individual_rate"', data_con)
    except:
        logger.error('Error loading gene average rate from gene_1010bp_individual_rate')
        raise RuntimeError    

    # Handle quantile assignments (either calculated or user-provided)
    if user_quantile_fn is None:
        # Calculate quantiles based on data
        gene_adj_pd = calc_adj_gene_with_quantile(
            gene_ind_pd, 
            gene_average_dict,
            num_quantile,
            "Rate", xmin, xmax
        )
        csv_fn = os.path.join(csv_folder, f'{prefix}.Gene.Adjusted_Rate.csv')
        gene_adj_pd.to_csv(csv_fn, index=False)
        gene_adj_pd.to_sql('SineWave_Gene_Ajusted_Rate', data_con, index=False, if_exists='replace')
    else:
        # Use user-provided quantile assignments
        gene_adj_pd = pd.read_csv(user_quantile_fn, index_col=False)
        required_cols = {'GroupName', 'Gene', 'Quantile'}
        missing_cols = required_cols - set(gene_adj_pd.columns)
        if missing_cols:
            logger.error('Manual Gene Quantile File missing required columns: %s', missing_cols)
            raise RuntimeError(f'Missing required columns: {missing_cols}')
        gene_adj_pd.to_sql('SineWave_Gene_Ajusted_Rate_UserQuantile', data_con, index=False, if_exists='replace')

    # Process quantile-specific analyses
    if user_quantile_fn is None:
        # Analysis with calculated quantiles
        gene_quantile_ave_pd, gene_quantile_ave_fit_pd, gene_quantile_ave_dict = sine_fit_gene_quantile(
            gene_ind_pd, gene_adj_pd, num_quantile, 
            'Rate', xmin, xmax)
        
        # Save calculated quantile results
        csv_fn = os.path.join(csv_folder, f'{prefix}.Gene.AverageRate_1kb.Quantile.csv')
        gene_quantile_ave_pd.to_csv(csv_fn, index=False)
        gene_quantile_ave_pd.to_sql('SineWave_Gene_AverageRate_1kb_Quantile', data_con, index=False, if_exists='replace')

        csv_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageRate.Quantile.SineWave.result.csv')
        pkl_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageRate.Quantile.SineWave.pkl')    
        gene_quantile_ave_fit_pd.to_csv(csv_fn, index=False)
        gene_quantile_ave_fit_pd.to_sql('SineWave_Gene_AverageRate_Quantile_Fit', data_con, index=False, if_exists='replace')
        with open(pkl_fn, 'wb') as filep:
            pickle.dump(gene_quantile_ave_dict, filep)
    else:
        # Analysis with user-provided quantiles
        gene_quantile_ave_pd, gene_quantile_ave_fit_pd, gene_quantile_ave_dict = sine_fit_gene_quantile(
            gene_ind_pd, gene_adj_pd, num_quantile, 
            'Rate', xmin, xmax, is_user=True)
        
        # Save user quantile results
        csv_fn = os.path.join(csv_folder, f'{prefix}.Gene.AverageRate_1kb.UserQuantile.csv')
        gene_quantile_ave_pd.to_csv(csv_fn, index=False)
        gene_quantile_ave_pd.to_sql('SineWave_Gene_AverageRate_1kb_Quantile', data_con, index=False, if_exists='replace')

        csv_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageRate.UserQuantile.SineWave.result.csv')
        pkl_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageRate.UserQuantile.SineWave.pkl')    
        gene_quantile_ave_fit_pd.to_csv(csv_fn, index=False)
        gene_quantile_ave_fit_pd.to_sql('SineWave_Gene_AverageRate_UserQuantile_Fit', data_con, index=False, if_exists='replace')
        with open(pkl_fn, 'wb') as filep:
            pickle.dump(gene_quantile_ave_dict, filep)        
    logger.info('Sine Wave analysis of average gene phasing in quantiles is successful.')
    data_con.close()
