"""
A comprehensive toolkit for analyzing genomic methylation data with a focus on various genomic features.

This module provides functions for extracting, processing, and analyzing methylation data 
from SQLite databases, with particular emphasis on different genomic regions including 
mitochondrial DNA, gene bodies, nucleosome-depleted regions (NDRs), tRNA genes, and 
other genomic features. It supports time course analysis and statistical calculations
for methylation patterns.

Key Features:
    - Extraction and analysis of mitochondrial DNA methylation data
    - Processing of gene body and NDR methylation patterns
    - Analysis of tRNA genes and their methylation profiles
    - Calculation of methylation rates and fractions around transcription start sites
    - Statistical analysis including percentile calculations and smoothing functions
    - Support for time course experiments and replicate handling
    - Memory-efficient processing of large genomic datasets

Requirements:
    - pandas: For data manipulation and analysis
    - numpy: For numerical computations
    - sqlite3: For database operations
    - sinekit.utility: Custom utility functions for methylation calculations
    - logging: For operation logging
    - gc: For memory management

The module expects input data to be organized in SQLite databases with specific table
structures for different experimental conditions and replicates. It handles both
strand-specific and non-strand-specific data, and can process various genomic
features including genes, tRNAs, ARS (Autonomously Replicating Sequences),
telomeres, and transposable elements.

Notes:
    - All genomic positions are expected to be 1-based
    - Mitochondrial chromosome should be labeled as 'Mito'
    - Functions handle both '+' and '-' strand orientations
    - Memory management is implemented for processing large datasets
    - Results are saved in both SQLite database and CSV format

Author: Zhuwei Xu
Version: 1.0.0
Date: 25-02-03
"""

import logging
import sqlite3
import os
import gc
import gzip
from sinekit.utility import (calc_methylation_rate, 
    load_sample_sheet, load_config_file, get_chrom_sizes)
import pandas as pd
import numpy as np

logger = logging.getLogger("sinekit")

def extract_mitochondria_data(data_con: sqlite3.Connection,
                            sample_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and analyzes mitochondrial data from SQLite database for given samples.
    
    This function processes mitochondrial chromosome data for each experiment and 
    replicate, calculating percentiles of values at different time points.
    Mitochondria must be named as 'Mito'
    
    Parameters
    ----------
    data_con : sqlite3.Connection
        SQLite database connection containing the mitochondrial data
    sample_pd : pd.DataFrame
        DataFrame containing sample information with columns:
        'experiment', 'rep', and 'time'
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing processed mitochondrial data with columns:
        'Experiment', 'Rep', 'Time', and percentile columns ('5', '15', '25', 
        '50', '75', '85', '95')
    
    Notes
    -----
    The function:
    1. Processes data for each experiment/replicate combination
    2. Extracts Mito chromosome data from corresponding SQLite tables
    3. Calculates percentiles (5,15,25,50,75,85,95) for each time point
    4. Uses garbage collection to manage memory during processing
    """
    percentile_list = [5, 15, 25, 50, 75, 85, 95]
    mito_pd = []  # Initialize list to store results
    
    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        tmp_pd = pd.read_sql_query(
            f'select * from "{exp_}_{rep}_combine" where Chrom = "Mito"', 
            data_con
        )
        
        tmp_sample_pd = sample_pd.loc[
            (sample_pd['experiment'] == exp_) & 
            (sample_pd['rep'] == rep)
        ]
        
        time_list = sorted(tmp_sample_pd['time'].values)
        
        for t_ in time_list:
            values = tmp_pd[str(t_)].values
            values = values[~np.isnan(values)]
            tmp = [exp_, rep, t_]
            
            for p in percentile_list:
                v = np.percentile(values, p)
                tmp.append(v)
            
            mito_pd.append(tmp)
            
        gc.collect()
    
    mito_pd = pd.DataFrame(
        mito_pd,
        columns=['Experiment', 'Rep', 'Time'] + [str(x) for x in percentile_list]
    )
    
    return mito_pd

def extract_gene_percentile_timecourse(
    data_con: sqlite3.Connection,
    sample_pd: pd.DataFrame,
    gene_pd: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts and analyzes gene methylaed fraction time course data, calculating statistics 
    for both gene bodies and NDR (Nucleosome Depleted Regions).

    This function processes genomic data for each experiment and replicate, 
    calculating percentiles of values at different time points for both gene 
    bodies and NDR regions separately.

    Parameters
    ----------
    data_con : sqlite3.Connection
        SQLite database connection containing the genomic data
    sample_pd : pd.DataFrame
        DataFrame containing sample information with columns:
        'experiment', 'rep', and 'time'
    gene_pd : pd.DataFrame
        DataFrame containing gene information with columns:
        'Chrom', 'Start', 'End', 'NDR_Start', 'NDR_End'

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrames containing processed data:
        - ndr_pd: NDR region statistics
        - genebody_pd: Gene body statistics
        Both DataFrames have columns:
        'Experiment', 'Rep', 'Time', and percentile columns 
        ('5', '15', '25', '50', '75', '85', '95')

    Notes
    -----
    The function:
    1. Processes data for each experiment/replicate combination
    2. Extracts data for each chromosome
    3. Calculates statistics for both gene bodies and NDR regions
    4. Uses garbage collection to manage memory during processing
    5. Calculates percentiles (5,15,25,50,75,85,95) for each time point
    """
    genebody_pd = []
    ndr_pd = []
    percentile_list = [5, 15, 25, 50, 75, 85, 95]
    chrom_size_dict = get_chrom_sizes(data_con)

    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        value_ndr = []
        value_genebody = []
        tmp_sample_pd = sample_pd.loc[
            (sample_pd['experiment'] == exp_) & 
            (sample_pd['rep'] == rep)
        ]
        time_list = sorted(tmp_sample_pd['time'].values)

        for chrom, tmp_gene_pd in gene_pd.groupby(by='Chrom', sort=False):
            tmp_pd = pd.read_sql_query(
                f'select * from "{exp_}_{rep}_combine" where Chrom = "{chrom}"', 
                data_con
            )
            pos = tmp_pd['Pos'].values
            csize = chrom_size_dict[chrom]
            values = np.empty((len(time_list), csize + 10))
            values[:] = np.nan

            for i, t_ in enumerate(time_list):
                values[i, pos] = tmp_pd[str(t_)].values

            for _, row in tmp_gene_pd.iterrows():
                gs = row['Start'] - 1
                ge = row['End']
                ns = row['NDR_Start'] - 1
                ne = row['NDR_End']
                val_g = values[:, gs:ge].copy()
                val_n = values[:, ns:ne].copy()
                value_ndr.append(val_n)
                value_genebody.append(val_g)

            del pos, values
            gc.collect()

        value_ndr = np.concatenate(value_ndr, axis=1)
        value_genebody = np.concatenate(value_genebody, axis=1)

        for i, t_ in enumerate(time_list):
            ent_n = [exp_, rep, t_]
            ent_g = [exp_, rep, t_]
            tmp_n = value_ndr[i, :].reshape(-1)
            tmp_g = value_genebody[i, :].reshape(-1)

            for p in percentile_list:
                ent_n.append(np.nanpercentile(tmp_n, p))
                ent_g.append(np.nanpercentile(tmp_g, p))    

            genebody_pd.append(ent_g)
            ndr_pd.append(ent_n)

        del value_ndr, value_genebody
        gc.collect()     

    ndr_pd = pd.DataFrame(
        ndr_pd, 
        columns=['Experiment', 'Rep', 'Time'] + [str(x) for x in percentile_list]
    )       
    genebody_pd = pd.DataFrame(
        genebody_pd, 
        columns=['Experiment', 'Rep', 'Time'] + [str(x) for x in percentile_list]
    ) 

    return ndr_pd, genebody_pd

def extract_mito_gene_median_pd(
    ndr_pd: pd.DataFrame,
    genebody_pd: pd.DataFrame,
    mito_pd: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines median values from mitochondrial DNA, gene body, and NDR data and calculates methylation rates.

    This function extracts the 50th percentile (median) values from three different 
    data sources, combines them into a unified DataFrame with feature labels, and 
    calculates methylation rates for each feature type.

    Parameters
    ----------
    ndr_pd : pd.DataFrame
        DataFrame containing NDR (Nucleosome Depleted Region) data with columns:
        'Experiment', 'Rep', 'Time', '50'
    genebody_pd : pd.DataFrame
        DataFrame containing gene body data with columns:
        'Experiment', 'Rep', 'Time', '50'
    mito_pd : pd.DataFrame
        DataFrame containing mitochondrial DNA data with columns:
        'Experiment', 'Rep', 'Time', '50'

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrames:
        - median_pd: Combined median values with columns:
          'Experiment', 'Rep', 'Time', 'Median', 'Feature'
        - rate_pd: Methylation rates with columns:
          'Experiment', 'Rep', [rate columns], 'Feature'
        where Feature can be 'mtDNA', 'Genebody', or 'NDR'

    Notes
    -----
    The '50' column in input DataFrames represents the median values and is 
    renamed to 'Median' in the output DataFrame. Methylation rates are calculated
    using the calc_methylation_rate function.
    """
    median_pd = []

    # Extract mitochondrial DNA medians
    tmp = mito_pd[['Experiment', 'Rep', 'Time', '50']].copy()
    tmp['Feature'] = 'mtDNA'
    median_pd.append(tmp)

    # Extract gene body medians
    tmp = genebody_pd[['Experiment', 'Rep', 'Time', '50']].copy()
    tmp['Feature'] = 'Genebody'
    median_pd.append(tmp)

    # Extract NDR medians
    tmp = ndr_pd[['Experiment', 'Rep', 'Time', '50']].copy()
    tmp['Feature'] = 'NDR'
    median_pd.append(tmp)

    # Combine all data
    median_pd = pd.concat(median_pd, ignore_index=True)
    median_pd.rename(columns={'50': 'Median'}, inplace=True)

    rate_pd = []
    for feat in ['mtDNA', 'Genebody', 'NDR']:
        tmp_pd = median_pd.loc[median_pd['Feature'] == feat]
        tmp_raw = []
        time_list = None
        for (exp_, rep), tmp_pd2 in tmp_pd.groupby(by=['Experiment', 'Rep'], sort=False):
            tmp = [exp_, rep]
            tmp.extend(list(tmp_pd2['Median'].values))
            tmp_raw.append(tmp)
            if time_list is None:
                time_list = list(tmp_pd2['Time'].values)
        tmp_raw = pd.DataFrame(tmp_raw, columns=['Chrom', 'Pos'] + [str(x) for x in time_list])
        tmp_rate = calc_methylation_rate(tmp_raw, np.array(time_list))
        tmp_rate.rename(columns={'Chrom': 'Experiment', 'Pos': 'Rep'}, inplace=True)
        tmp_rate['Feature'] = feat
        rate_pd.append(tmp_rate)

    rate_pd = pd.concat(rate_pd, ignore_index=True)

    return median_pd, rate_pd

def extract_feature_raw_rate(data_con: sqlite3.Connection,
    sample_pd: pd.DataFrame,
    gene_pd: pd.DataFrame,
    tRNA_pd: pd.DataFrame,
    other_pd: pd.DataFrame) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract relative rates for different genomic features from a SQLite database.
    
    This function processes genomic data to extract relative rates for various features
    including NDRs (Nucleosome Depleted Regions), gene bodies, tRNA genes, ARS
    (Autonomously Replicating Sequences), telomeres, and Ty elements. Raw (unnormalized)
    mtDNA rates are also extracted.
    
    Parameters
    ----------
    data_con : sqlite3.Connection
        SQLite database connection containing rate data tables
    sample_pd : pd.DataFrame
        DataFrame containing experiment and replicate information
    gene_pd : pd.DataFrame
        DataFrame with gene annotations including columns:
        Chrom, NDR_Start, NDR_End, Start, End
    tRNA_pd : pd.DataFrame
        DataFrame with tRNA gene annotations including columns:
        Chrom, Start, End
    other_pd : pd.DataFrame
        DataFrame with other feature annotations (ARS, Telomere, Ty) including columns:
        Feature, Chrom, Start, End
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Returns six DataFrames containing relative rates for:
        1. NDR regions
        2. Gene bodies
        3. tRNA genes
        4. ARS regions
        5. Telomeres
        6. Ty elements
        Each DataFrame contains columns: Experiment, Rep, Relative Rate
        7. mtDNA elements: Experiment, Rep, Raw Rate
    Notes
    -----
    - The function processes each chromosome separately to manage memory usage
    - Uses numpy arrays for efficient data handling
    - Calls gc.collect() periodically to manage memory
    - Expects rate data tables in format: "{experiment}_{rep}_Rate"
    """
    ndr_pd = []
    genebody_pd = []
    trna_gene_pd = []
    chrom_size_dict = get_chrom_sizes(data_con)
    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        tmp_ndr_values = []
        tmp_genebody_values = []
        tmp_pd = pd.read_sql_query(f'select * from "{exp_}_{rep}_Rate"', data_con)
        for chrom, tmp_pd2 in gene_pd.groupby('Chrom', sort=False):
            tmp_pd3 = tmp_pd.loc[tmp_pd['Chrom'] == chrom]
            pos = tmp_pd3['Pos'].values
            csize = chrom_size_dict[chrom]
            values = np.empty(csize + 10)
            values[:] = np.nan
            values[pos] = tmp_pd3['Relative Rate'].values
            for nl, nr in zip(tmp_pd2['NDR_Start'], tmp_pd2['NDR_End']):
                nl -= 1
                tmp = values[nl: nr].copy()
                tmp = tmp[~np.isnan(tmp)]
                if len(tmp) > 0:
                    tmp_ndr_values.append(tmp)
            for gl, gr in zip(tmp_pd2['Start'], tmp_pd2['End']):
                gl -= 1
                tmp = values[gl: gr].copy()
                tmp = tmp[~np.isnan(tmp)]
                if len(tmp) > 0:
                    tmp_genebody_values.append(tmp)                    
            gc.collect()
        tmp_ndr_values = np.concatenate(tmp_ndr_values)
        tmp_ndr_pd = pd.DataFrame({'Experiment': exp_, 'Rep': rep, 'Relative Rate': tmp_ndr_values})
        tmp_genebody_values = np.concatenate(tmp_genebody_values)
        tmp_genebody_pd = pd.DataFrame({'Experiment': exp_, 'Rep': rep, 'Relative Rate': tmp_genebody_values})        
        ndr_pd.append(tmp_ndr_pd)
        genebody_pd.append(tmp_genebody_pd)
    ndr_pd = pd.concat(ndr_pd, ignore_index=True)
    genebody_pd = pd.concat(genebody_pd, ignore_index=True)

    logger.info('Raw gene data are extracted')
    tRNA_pd = tRNA_pd.loc[tRNA_pd['Type'].isin(['Exon', 'Intron'])]
    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        tmp_values = []
        tmp_pd = pd.read_sql_query(f'select * from "{exp_}_{rep}_Rate"', data_con)
        for chrom, tmp_pd2 in tRNA_pd.groupby('Chrom', sort=False):
            tmp_pd3 = tmp_pd.loc[tmp_pd['Chrom'] == chrom]
            pos = tmp_pd3['Pos'].values
            csize = chrom_size_dict[chrom]
            values = np.empty(csize + 10)
            values[:] = np.nan
            values[pos] = tmp_pd3['Relative Rate'].values
            for nl, nr in zip(tmp_pd2['Start'], tmp_pd2['End']):
                nl -= 1
                tmp = values[nl: nr]
                tmp = tmp[~np.isnan(tmp)]
                if len(tmp) > 0:
                    tmp_values.append(tmp)
        tmp_values = np.concatenate(tmp_values)
        tmp_pd = pd.DataFrame({'Experiment': exp_, 'Rep': rep, 'Relative Rate': tmp_values})
        trna_gene_pd.append(tmp_pd)
        gc.collect()
    trna_gene_pd = pd.concat(trna_gene_pd, ignore_index=True)
    logger.info('Raw tRNA data are extracted')

    ars_pd = other_pd.loc[other_pd['Feature'] == 'ARS']
    tel_pd = other_pd.loc[other_pd['Feature'] == 'Telomere']
    ty_pd = other_pd.loc[other_pd['Feature'] == 'Ty']

    ars_raw_pd = []
    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        tmp_values = []
        tmp_pd = pd.read_sql_query(f'select * from "{exp_}_{rep}_Rate"', data_con)
        for chrom, tmp_pd2 in ars_pd.groupby('Chrom', sort=False):
            tmp_pd3 = tmp_pd.loc[tmp_pd['Chrom'] == chrom]
            pos = tmp_pd3['Pos'].values
            csize = chrom_size_dict[chrom]
            values = np.empty(csize + 10)
            values[:] = np.nan
            values[pos] = tmp_pd3['Relative Rate'].values
            for nl, nr in zip(tmp_pd2['Start'], tmp_pd2['End']):
                nl -= 1
                tmp = values[nl: nr]
                tmp = tmp[~np.isnan(tmp)]
                if len(tmp) > 0:
                    tmp_values.append(tmp)
        tmp_values = np.concatenate(tmp_values)
        tmp_pd = pd.DataFrame({'Experiment': exp_, 'Rep': rep, 'Relative Rate': tmp_values})
        ars_raw_pd.append(tmp_pd)
        gc.collect()
    ars_raw_pd = pd.concat(ars_raw_pd, ignore_index=True)
    
    tel_raw_pd = []
    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        tmp_values = []
        tmp_pd = pd.read_sql_query(f'select * from "{exp_}_{rep}_Rate"', data_con)
        for chrom, tmp_pd2 in tel_pd.groupby('Chrom', sort=False):
            tmp_pd3 = tmp_pd.loc[tmp_pd['Chrom'] == chrom]
            pos = tmp_pd3['Pos'].values
            csize = chrom_size_dict[chrom]
            values = np.empty(csize + 10)
            values[:] = np.nan
            values[pos] = tmp_pd3['Relative Rate'].values
            for nl, nr in zip(tmp_pd2['Start'], tmp_pd2['End']):
                nl -= 1
                tmp = values[nl: nr]
                tmp = tmp[~np.isnan(tmp)]
                if len(tmp) > 0:
                    tmp_values.append(tmp)
        tmp_values = np.concatenate(tmp_values)
        tmp_pd = pd.DataFrame({'Experiment': exp_, 'Rep': rep, 'Relative Rate': tmp_values})
        tel_raw_pd.append(tmp_pd)
        gc.collect()
    tel_raw_pd = pd.concat(tel_raw_pd, ignore_index=True)    

    ty_raw_pd = []
    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        tmp_values = []
        tmp_pd = pd.read_sql_query(f'select * from "{exp_}_{rep}_Rate"', data_con)
        for chrom, tmp_pd2 in ty_pd.groupby('Chrom', sort=False):
            tmp_pd3 = tmp_pd.loc[tmp_pd['Chrom'] == chrom]
            pos = tmp_pd3['Pos'].values
            csize = chrom_size_dict[chrom]
            values = np.empty(csize + 10)
            values[:] = np.nan
            values[pos] = tmp_pd3['Relative Rate'].values
            for nl, nr in zip(tmp_pd2['Start'], tmp_pd2['End']):
                nl -= 1
                tmp = values[nl: nr]
                tmp = tmp[~np.isnan(tmp)]
                if len(tmp) > 0:
                    tmp_values.append(tmp)
        tmp_values = np.concatenate(tmp_values)
        tmp_pd = pd.DataFrame({'Experiment': exp_, 'Rep': rep, 'Relative Rate': tmp_values})
        ty_raw_pd.append(tmp_pd)
        gc.collect()
    ty_raw_pd = pd.concat(ty_raw_pd, ignore_index=True)

    logger.info('ARS, Tel and Ty data are extracted')

    mito_raw_pd = []
    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        tmp_pd = pd.read_sql_query(f'select Chrom, Rate from "{exp_}_{rep}_Rate" where Chrom = "Mito"', data_con)
        tmp_values = tmp_pd['Rate'].values
        tmp_values = tmp_values[~np.isnan(tmp_values)]
        tmp_pd2 = pd.DataFrame({'Experiment': exp_, 'Rep': rep, 'Raw Rate': tmp_values})
        mito_raw_pd.append(tmp_pd2)
    mito_raw_pd = pd.concat(mito_raw_pd, ignore_index=True)
    logger.info('mtDNA data are extracted')    
    return ndr_pd, genebody_pd, trna_gene_pd, ars_raw_pd, tel_raw_pd, ty_raw_pd, mito_raw_pd

def extract_tRNA_rate(data_con: sqlite3.Connection,
    sample_pd: pd.DataFrame, 
    tRNA_pd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract and analyze tRNA-related methylated fractions across features and time points.
    
    Parameters
    ----------
    data_con : sqlite3.Connection
        SQLite database connection containing combined rate tables in format "{experiment}_{rep}_combine"
    sample_pd : pd.DataFrame
        Experiment metadata with columns:
        - experiment: str, experiment identifier
        - rep: str/int, replicate number 
        - time: int/float, timepoint
    tRNA_pd : pd.DataFrame
        tRNA annotations with columns:
        - Type: str, one of ['TFIIIB', 'TFIIIC', 'Intron']
        - Chrom: str, chromosome identifier
        - Start: int, feature start position (1-based)
        - End: int, feature end position
    
    Returns
    -------
    percentile_pd : pd.DataFrame
        Percentile calculations with columns:
        - Rep: str/int, replicate identifier
        - Feature: str, feature type
        - Time: int/float, timepoint
        - Percentile columns ('5','15','25','50','75','85','95'): float, percentile values
    
    rate_pd : pd.DataFrame
        Methylation rates with columns:
        - Feature: str, feature type 
        - Experiment: str, experiment identifier
        - Rep: str/int, replicate number
        - Rate columns: float, methylation rates per timepoint
    """
    percentile_list = [5, 15, 25, 50, 75, 85, 95]
    feat_list = ['TFIIIB', 'TFIIIC', 'Intron']
    percentile_pd = []
    chrom_size_dict = get_chrom_sizes(data_con)

    normalization_pd = pd.read_sql_query('select * from "Sample_Median_Rate_combine"', data_con)
    norm_rate_dict= {}
    for _, row in normalization_pd.iterrows():
        exp_ = row['Experiment']
        rep = row['Rep']
        rate = row["Rate"]
        norm_rate_dict[(exp_, rep)] = rate
    
    for (exp_, rep), tmp_sample_pd in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        time_list = tmp_sample_pd["time"].values
        tmp_pd = pd.read_sql_query(f'select * from "{exp_}_{rep}_combine"', data_con)

        for feat in feat_list:
            tmp_dict = {}
            tmp_tRNA_pd = tRNA_pd.loc[tRNA_pd['Type'] == feat]            
            for chrom, tmp_pd2 in tmp_tRNA_pd.groupby('Chrom', sort=False):
                tmp_pd3 = tmp_pd.loc[tmp_pd['Chrom'] == chrom]
                pos = tmp_pd3['Pos'].values
                
                for time_ in time_list:
                    tmp_dict.setdefault(time_, [])
                    csize = chrom_size_dict[chrom]
                    values = np.empty(csize + 10)
                    values[:] = np.nan
                    values[pos] = tmp_pd3[str(time_)].values
                    
                    for nl, nr in zip(tmp_pd2['Start'], tmp_pd2['End']):
                        nl -= 1
                        tmp = values[nl: nr]
                        tmp = tmp[~np.isnan(tmp)]
                        if len(tmp) > 0:
                            tmp_dict[time_].append(tmp)
            
            for time_, da in tmp_dict.items():
                tmp = np.concatenate(da)
                ent = [exp_, rep, feat, time_]
                for p in percentile_list:
                    val = np.percentile(tmp, p)
                    ent.append(val)
                percentile_pd.append(ent)
    
    percentile_pd = pd.DataFrame(
        percentile_pd, 
        columns=['Experiment', 'Rep', 'Feature', 'Time'] + [str(x) for x in percentile_list]
    )


    rate_pd = []
    for feat in feat_list:
        tmp_pd = percentile_pd.loc[percentile_pd['Feature'] == feat]
        tmp_raw = []
        tmp_norm_rates = []
        for (exp_, rep), tmp_pd2 in tmp_pd.groupby(by=['Experiment', 'Rep'], sort=False):
            norm_rate = norm_rate_dict[(exp_, rep)]
            tmp_norm_rates.append(norm_rate)
            tmp = [exp_, rep]
            tmp.extend(list(tmp_pd2['50'].values))
            tmp_raw.append(tmp)
        tmp_raw = pd.DataFrame(tmp_raw, columns=['Chrom', 'Pos'] + [str(x) for x in time_list])
        tmp_rate = calc_methylation_rate(tmp_raw, np.array(time_list))
        tmp_rate.rename(columns={'Chrom': 'Experiment', 'Pos': 'Rep'}, inplace=True)
        tmp_rate['Feature'] = feat
        tmp_norm_rates = np.array(tmp_norm_rates)
        tmp_rate['Relative Rate'] = tmp_rate['Rate'] / tmp_norm_rates
        rate_pd.append(tmp_rate)
    rate_pd = pd.concat(rate_pd, ignore_index=True)
    cols = list(rate_pd.columns)
    rate_pd = rate_pd[cols[-2:] + cols[:-2]]
    
    return percentile_pd, rate_pd

def extract_tRNA_single_exon(data_con: sqlite3.Connection,
    sample_pd: pd.DataFrame, 
    tRNA_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and analyze relative rates around single-exon tRNA genes with flanking regions.
    
    This function processes single-exon tRNA genes to extract relative rates for
    upstream, exon, and downstream regions, calculating average profiles and 
    standard deviations while considering strand orientation.
    
    Parameters
    ----------
    data_con : sqlite3.Connection
        SQLite database connection containing rate data tables
    sample_pd : pd.DataFrame
        DataFrame containing experiment and replicate information
        Must have columns: ['experiment', 'rep']
    tRNA_pd : pd.DataFrame
        DataFrame with tRNA annotations including columns:
        ['IsMulti', 'Type', 'Chrom', 'Start', 'End', 'Strand']
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing processed data with columns:
        - Experiment: experiment identifier
        - Rep: replicate identifier
        - Pos: position relative to tRNA start (-200 to end+200)
        - Average: mean relative rate at each position
        - Std: standard deviation of relative rate at each position
        
    Notes
    -----
    - Processes only single-exon tRNAs (where IsMulti is False)
    - Excludes contigs from mask_contig set
    - Uses flanking regions of 200bp upstream and downstream
    - Handles both + and - strands, flipping data for - strand
    - Expects rate data tables in format: "{experiment}_{rep}_Rate"
    - Aligns and combines data from multiple tRNAs to create average profiles
    """
    left_flank = 200
    right_flank = 200
    chrom_size_dict = get_chrom_sizes(data_con)
    mask_contig = set(['2micron_formA', '2micron_formB', 'Mito'])
    single_exon_pd = tRNA_pd.loc[~tRNA_pd['IsMulti']]
    tRNA_single_dict = {}

    # First phase: Extract raw data for each region
    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):  
        tmp_array_up = []
        tmp_array_exon = []
        tmp_array_down = []    
        combine_rate_pd = pd.read_sql_query(f'select * from "{exp_}_{rep}_Rate"', data_con)
        combine_rate_pd = combine_rate_pd.loc[~combine_rate_pd['Chrom'].isin(mask_contig)].copy()
        
        for chrom, tmp_trna in single_exon_pd.groupby('Chrom', sort=False):
            tmp_pd = combine_rate_pd.loc[combine_rate_pd['Chrom'] == chrom]
            csize = chrom_size_dict[chrom]
            tmp = np.empty(csize + 100)
            tmp.fill(np.nan)
            tmp[tmp_pd['Pos'].values] = tmp_pd['Relative Rate'].values
            
            for _, row in tmp_trna.iterrows():
                tp = row['Type']
                start = row['Start'] - 1
                end = row['End']
                st = row['Strand']
                
                if tp != 'Exon':
                    continue
                    
                if st == '+':
                    tmp_array_up.append(tmp[start - left_flank: start].copy())
                    tmp_array_exon.append(tmp[start: end].copy())
                    tmp_array_down.append(tmp[end: end+right_flank].copy())
                if st == '-':
                    tmp_array_down.append(np.flip(tmp[start - left_flank: start]))
                    tmp_array_exon.append(np.flip(tmp[start: end]))
                    tmp_array_up.append(np.flip(tmp[end: end+right_flank]))
                    
        tRNA_single_dict[exp_, rep] = [tmp_array_up, tmp_array_exon, tmp_array_down]
    
    # Second phase: Process data and calculate statistics
    tRNA_single_pd = []
    for (exp_, rep), (tmp_array_up, tmp_array_exon, tmp_array_down) in tRNA_single_dict.items():
        # Calculate maximum sizes for each region
        size_up = max([len(x) for x in tmp_array_up])
        size_exon = max([len(x) for x in tmp_array_exon])
        size_down = max([len(x) for x in tmp_array_down])
        size_list = [size_up, size_exon, size_down]
        
        # Calculate starting positions for each region
        start_pos = [0]
        for size in size_list[:-1]:
            start_pos.append(start_pos[-1] + size)
        
        # Initialize combined array
        tmp_array = np.empty((len(tmp_array_up), sum(size_list)))
        tmp_array[:] = np.nan
        
        # Fill array with data from each region
        for i, t in enumerate(tmp_array_up):
            start = start_pos[0]
            val_pos = np.where(~np.isnan(t))[0]
            if len(val_pos) == 0:
                continue
            tmp_array[i, val_pos + start] = t[val_pos]
            
        for i, t in enumerate(tmp_array_exon):
            start = start_pos[1]
            val_pos = np.where(~np.isnan(t))[0]
            if len(val_pos) == 0:
                continue
            tmp_array[i, val_pos + start] = t[val_pos]   
            
        for i, t in enumerate(tmp_array_down):
            start = start_pos[2]
            val_pos = np.where(~np.isnan(t))[0]
            if len(val_pos) == 0:
                continue
            tmp_array[i, val_pos + start] = t[val_pos]
        
        # Calculate statistics
        tmp_ave = np.nanmean(tmp_array, axis=0)
        tmp_std = np.nanstd(tmp_array, axis=0)
        pos = np.arange(-left_flank, len(tmp_ave) - left_flank)
        
        # Create DataFrame for this experiment/replicate
        tmp_pd = pd.DataFrame({
            'Experiment': exp_,
            'Rep': rep,
            'Pos': pos,
            'Average': tmp_ave,
            'Std': tmp_std
        })
        tRNA_single_pd.append(tmp_pd)
    
    # Combine all results
    tRNA_single_pd = pd.concat(tRNA_single_pd, ignore_index=True)
    return tRNA_single_pd


def extract_gene_rate(
    data_con: sqlite3.Connection,
    sample_pd: pd.DataFrame, 
    gene_pd: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract and calculate relative gene rates around transcription start sites,
    both averaged across all genes and for individual genes.
    
    This function processes genomic data to calculate relative rates around gene 
    regions in two ways: averaged across all genes and individually for each gene.
    It considers strand orientation and experimental replicates, using a flanking
    region around the +1 position of each gene and handling both forward and 
    reverse strands appropriately.
    
    Parameters
    ----------
    data_con : sqlite3.Connection
        SQLite database connection containing rate data tables
    sample_pd : pd.DataFrame
        DataFrame containing experiment and replicate information
        Must have columns: ['experiment', 'rep']
    gene_pd : pd.DataFrame
        DataFrame containing gene information
        Must have columns: ['Chrom', 'Plus1', 'Start', 'End', 'Strand', 'Gene']
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        1. Averaged rates DataFrame with columns:
           ['Experiment', 'Rep', 'Pos', 'Value']
           where:
           - 'Pos' represents position relative to +1 site
           - 'Value' represents the mean relative rate at that position
        2. Individual gene rates DataFrame with columns:
           ['Experiment', 'Rep', 'Gene', 'Pos', 'Value']
           where:
           - 'Gene' is the gene identifier
           - 'Pos' and 'Value' are as above but for individual genes
    
    Notes
    -----
    - Uses a flanking region of 1010bp on each side of the +1 position
    - Handles both '+' and '-' strand orientations
    - Processes data chromosome by chromosome to manage memory usage
    - Filters out positions with NaN values for individual gene data
    - Returns both population-averaged values and individual gene profiles
    - Uses garbage collection to manage memory during processing
    """
    flank: int = 1010
    gene_rate_raw_pd: list = []
    gene_individual_rate_raw_pd: list = []
    chrom_size_dict: dict = get_chrom_sizes(data_con)
    
    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        tmp_array: list = []
        for chrom, tmp_gene_pd in gene_pd.groupby(by='Chrom', sort=False):
            tmp_pd = pd.read_sql_query(
                f'select * from "{exp_}_{rep}_Rate" where Chrom = "{chrom}"',
                data_con
            )
            pos = tmp_pd["Pos"].values
            csize = chrom_size_dict[chrom]
            values = np.empty(csize + 10)
            values[:] = np.nan
            values[pos] = tmp_pd['Relative Rate'].values
            
            for (p1, start, end, st, gene) in zip(
                tmp_gene_pd['Plus1'],
                tmp_gene_pd['Start'],
                tmp_gene_pd['End'],
                tmp_gene_pd['Strand'],
                tmp_gene_pd['Gene']
            ):
                p1 -= 1
                start -= 1
                tmp = np.empty(2*flank+1)
                tmp[:] = np.nan
                
                if st == '+':
                    l = p1 - flank
                    r = min([p1 + flank + 1, end])
                    val = values[l: r]
                    tmp[:len(val)] = val
                else:
                    l = max([start, p1 - flank])
                    r = p1 + flank + 1
                    val = values[l: r]
                    tmp[-len(val):] = val
                    tmp = np.flip(tmp)
                valid = ~np.isnan(tmp)
                if np.any(valid):
                    p = np.where(valid)[0] - flank
                    v = tmp[valid]
                    tmp_ent = pd.DataFrame({
                        'Experiment': exp_,
                        'Rep': rep,
                        'Gene': gene,
                        'Pos': p,
                        'Value': v
                    })
                    gene_individual_rate_raw_pd.append(tmp_ent)
                    tmp_array.append(tmp)
            del values, pos
            gc.collect()
            
        tmp_array = np.vstack(tmp_array)
        tmp_array = np.nanmean(tmp_array, axis=0)
        xpos = np.arange(-flank, -flank+len(tmp_array))
        tmp = pd.DataFrame({
            'Experiment': exp_,
            'Rep': rep,
            'Pos': xpos,
            'Value': tmp_array
        })
        gene_rate_raw_pd.append(tmp)
        gc.collect()
        logger.info('%s %s done', exp_, rep)
        
    gene_rate_raw_pd = pd.concat(gene_rate_raw_pd, ignore_index=True)
    gene_individual_rate_raw_pd = pd.concat(gene_individual_rate_raw_pd, ignore_index=True)
    return gene_rate_raw_pd, gene_individual_rate_raw_pd


def calc_smooth_pd(raw_pd: pd.DataFrame, window: int=21) -> pd.DataFrame:
    """
    Calculate smoothed values using sliding window convolution.

    Parameters
    ----------
    raw_pd : pd.DataFrame
        Input data with columns:
        - Experiment: str, experiment identifier
        - Rep: str/int, replicate number
        - Pos: int, position
        - Value: float, measurement value
    window : int, default=21
        Size of smoothing window
    
    Returns
    -------
    pd.DataFrame
        Smoothed data with columns:
        - Experiment: str, experiment identifier
        - Rep: str/int, replicate number 
        - Pos: int, position
        - SmoothedValue: float, smoothed measurement value
    """
    flank = 1010
    smooth_pd = []
    for (exp_, rep), tmp_pd in raw_pd.groupby(by=['Experiment', 'Rep'], sort=False):
        tmp = np.empty(2 *flank + 1)
        tmp[:] = np.nan
        pos = tmp_pd['Pos'].values
        value = tmp_pd['Value'].values
        tmp[pos+flank] = value
        inval = np.isnan(tmp)
        tmp[inval] = np.interp(np.where(inval)[0], np.where(~inval)[0], tmp[~inval])
        sm = np.convolve(tmp, np.ones(window)/window, 'valid')
        xpos = np.arange(-flank+window//2, -flank+window//2+len(sm))
        sm_pd = pd.DataFrame({'Experiment': exp_, 'Rep': rep, 
                                'Pos': xpos, 'SmoothedValue': sm})
        smooth_pd.append(sm_pd)
    smooth_pd = pd.concat(smooth_pd, ignore_index=True)
    return smooth_pd


def extract_gene_fraction(
    data_con: sqlite3.Connection,
    sample_pd: pd.DataFrame,
    gene_pd: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract and calculate gene fractions around transcription start sites at different timepoints.
    
    This function processes genomic data to calculate average fractions around gene 
    regions for different experimental timepoints, considering strand orientation 
    and experimental replicates.
    
    Parameters
    ----------
    data_con : sqlite3.Connection
        SQLite database connection containing combined data tables
    sample_pd : pd.DataFrame
        DataFrame containing experiment, replicate, and time information
        Must have columns: ['experiment', 'rep', 'time']
    gene_pd : pd.DataFrame
        DataFrame containing gene information
        Must have columns: ['Chrom', 'Plus1', 'Start', 'End', 'Strand']
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing processed fraction data with columns:
        ['Experiment', 'Rep', 'Time', 'Pos', 'Value']
        where:
        - 'Pos' represents position relative to +1 site
        - 'Value' represents the mean fraction at that position
        - 'Time' represents the timepoint of measurement
    
    Notes
    -----
    - Uses a flanking region of 1010bp on each side of the +1 position
    - Handles both '+' and '-' strand orientations
    - Processes data chromosome by chromosome to manage memory usage
    - Returns averaged values across all genes for each experiment/replicate/timepoint
    - Uses garbage collection to manage memory during processing
    """
    flank: int = 1010
    gene_frac_raw_pd: list = []
    gene_frac_ind_pd: list = []
    chrom_size_dict: dict = get_chrom_sizes(data_con)
    
    for (exp_, rep), _ in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        tmp_sample_pd = sample_pd.loc[
            (sample_pd['experiment'] == exp_) & 
            (sample_pd['rep'] == rep)
        ]
        time_list = tmp_sample_pd['time'].values
        
        for time_ in time_list:
            tmp_array: list = []
            for chrom, tmp_gene_pd in gene_pd.groupby(by='Chrom', sort=False):
                tmp_pd = pd.read_sql_query(
                    f'select Chrom, Pos, "{time_}" from "{exp_}_{rep}_combine" '
                    f'where Chrom = "{chrom}"',
                    data_con
                )
                pos = tmp_pd["Pos"].values
                csize = chrom_size_dict[chrom]
                values = np.empty(csize + 10)
                values[:] = np.nan
                values[pos] = tmp_pd[str(time_)].values
                
                for (p1, start, end, st, gene) in zip(
                    tmp_gene_pd['Plus1'],
                    tmp_gene_pd['Start'],
                    tmp_gene_pd['End'],
                    tmp_gene_pd['Strand'],
                    tmp_gene_pd['Gene']
                ):
                    p1 -= 1
                    start -= 1
                    tmp = np.empty(2*flank+1)
                    tmp[:] = np.nan
                    
                    if st == '+':
                        l = p1 - flank
                        r = min([p1 + flank + 1, end])
                        val = values[l: r]
                        tmp[:len(val)] = val
                    else:
                        l = max([start, p1 - flank])
                        r = p1 + flank + 1
                        val = values[l: r]
                        tmp[-len(val):] = val
                        tmp = np.flip(tmp)
                        
                    valid = ~np.isnan(tmp)
                    if np.any(valid):
                        p = np.where(valid)[0] - flank
                        v = tmp[valid]
                        tmp_ent = pd.DataFrame({
                            'Experiment': exp_,
                            'Rep': rep,
                            'Time': time_,
                            'Gene': gene,
                            'Pos': p,
                            'Value': v
                        })
                        gene_frac_ind_pd.append(tmp_ent)
                        tmp_array.append(tmp)
                    
                del values, pos
                gc.collect()
                
            tmp_array = np.vstack(tmp_array)
            tmp_array = np.nanmean(tmp_array, axis=0)
            xpos = np.arange(-flank, -flank+len(tmp_array))
            tmp = pd.DataFrame({
                'Experiment': exp_,
                'Rep': rep,
                'Time': time_,
                'Pos': xpos,
                'Value': tmp_array
            })
            gene_frac_raw_pd.append(tmp)
            gc.collect()
            
        logger.info('%s %s done', exp_, rep)
        
    gene_frac_raw_pd = pd.concat(gene_frac_raw_pd, ignore_index=True)
    gene_frac_ind_pd = pd.concat(gene_frac_ind_pd, ignore_index=True)
    return gene_frac_raw_pd, gene_frac_ind_pd


def extract_all(config: str):
    """
    Extract and process methylation data from database and save results.

    Parameters
    ----------
    config : str
        Path to config file containing:
        - data.sample_sheet: path to sample metadata
        - resource.gene_table: gene annotations
        - resource.trna_table: tRNA annotations 
        - resource.other_table: other feature annotations
        - output.prefix: output file prefix
        - output.output_folder: output directory

    Processes:
    - mtDNA methylation timecourse
    - NDR/gene body methylation timecourse and rates
    - Feature-specific raw rates (NDR, gene body, tRNA, ARS, TEL, Ty)
    - tRNA element methylation data
    - Gene +/-1kb average methylation rates and fractions

    Saves results to SQLite database and CSV files in output_folder/spreadsheet/
    """
    config_dict = load_config_file(config)
    sample_fn = config_dict['data']['sample_sheet']
    sample_pd = load_sample_sheet(sample_fn)
    gene_pd = pd.read_csv(config_dict['resource']['gene_table'], index_col=False)
    trna_pd = pd.read_csv(config_dict['resource']['trna_table'], index_col=False)
    other_pd = pd.read_csv(config_dict['resource']['other_table'], index_col=False)

    prefix = config_dict["output"]["prefix"]
    output_folder = config_dict["output"]["output_folder"]    
    csv_folder = os.path.join(output_folder, 'Spreadsheet')
    os.makedirs(csv_folder, exist_ok=True)

    data_db = os.path.join(output_folder, f'{prefix}.SQLite.db')
    if not os.path.isfile(data_db):
        logger.error('Cannot found database file %s, please run load command first')
        raise RuntimeError
    
    data_con = sqlite3.connect(data_db)
    # mtDNA timecourse
    mito_pd = extract_mitochondria_data(data_con, sample_pd)
    mito_pd.to_sql('mtDNA_timecourse_percentile', data_con, index=False, if_exists='replace')
    csv_fn = os.path.join(csv_folder, f'{prefix}.mtDNA.timecourse.percentile.csv')
    mito_pd.to_csv(csv_fn, index=False)
    logger.info('mtDNA timecourse is processed')
    # NDR and genebody timecourse
    ndr_pd, genebody_pd = extract_gene_percentile_timecourse(
        data_con, sample_pd, gene_pd)
    ndr_pd.to_sql('gene_NDR_timecourse_percentile', data_con, index=False, if_exists='replace')
    csv_fn = os.path.join(csv_folder, f'{prefix}.gene_NDR.timecourse.percentile.csv')
    ndr_pd.to_csv(csv_fn, index=False)

    genebody_pd.to_sql('genebody_timecourse_percentile', data_con, index=False, if_exists='replace')    
    csv_fn = os.path.join(csv_folder, f'{prefix}.genebody.timecourse.percentile.csv')
    genebody_pd.to_csv(csv_fn, index=False)
    logger.info('genebody and NDR timecousrse are processed')
    median_pd, rate_pd = extract_mito_gene_median_pd(
        ndr_pd, genebody_pd, mito_pd
    )

    median_pd.to_sql('mtDNA_genebody_NDR_timecourse_median', data_con, index=False, if_exists='replace')    
    csv_fn = os.path.join(csv_folder, f'{prefix}.mtDNA_genebody_NDR.timecourse.median.csv')
    median_pd.to_csv(csv_fn, index=False)    

    rate_pd.to_sql('mtDNA_genebody_NDR_median_rate', data_con, index=False, if_exists='replace')    
    csv_fn = os.path.join(csv_folder, f'{prefix}.mtDNA_genebody_NDR.rate.median.csv')
    rate_pd.to_csv(csv_fn, index=False)    
    logger.info('median rate of mtDNA, genebody and NDR are processed')

    raw_pd_list = extract_feature_raw_rate(
        data_con, sample_pd, gene_pd, trna_pd, other_pd)
    feat_list = ['NDR', 'Genebody', 'tRNA', 'ARS', 'TEL', 'Ty', 'mtDNA']
    for tmp_pd, feat in zip(raw_pd_list, feat_list):
        tmp_pd.to_sql(f'Raw_{feat}_Rate', data_con, index=False, if_exists='replace')
        csv_fn = os.path.join(csv_folder, f'{prefix}.{feat}.raw_rate.csv.gz')
        with gzip.open(csv_fn, 'wt', compresslevel=9) as filep:
            tmp_pd.to_csv(filep, index=False, encoding='utf-8')          
    logger.info('Raw data for NDR, Genebody, tRNA, ARS, TEL, Ty and mtDNA are extracted')
    percentile_pd, rate_pd = extract_tRNA_rate(
        data_con, sample_pd, trna_pd
    )
    percentile_pd.to_sql('tRNA_element_timecourse_percentile', 
        data_con, index=False, if_exists='replace')
    csv_fn = os.path.join(csv_folder, f'{prefix}.tRNA_element.timecourse.percentile.csv')
    percentile_pd.to_csv(csv_fn, index=False)    

    rate_pd.to_sql('tRNA_element_median_rate', data_con, index=False, if_exists='replace')    
    csv_fn = os.path.join(csv_folder, f'{prefix}.tRNA_element.rate.median.csv')
    rate_pd.to_csv(csv_fn, index=False)    

    trna_single_exon_pd = extract_tRNA_single_exon(data_con,
        sample_pd, trna_pd)
    trna_single_exon_pd.to_sql('tRNA_single_exon_mean_std', data_con, index=False, if_exists='replace')    
    csv_fn = os.path.join(csv_folder, f'{prefix}.tRNA_single_exon.mean_std.csv')
    trna_single_exon_pd.to_csv(csv_fn, index=False)

    logger.info('tRNA element data is processed')

    gene_rate_raw_pd, gene_rate_ind_pd = extract_gene_rate(
        data_con, sample_pd, gene_pd)

    gene_rate_smooth_pd = calc_smooth_pd(gene_rate_raw_pd)

    gene_rate_raw_pd.to_sql('gene_1010bp_average_rate', data_con, index=False, if_exists='replace')    
    csv_fn = os.path.join(csv_folder, f'{prefix}.gene.1kb_average_rate.raw.csv')
    gene_rate_raw_pd.to_csv(csv_fn, index=False) 

    gene_rate_ind_pd.to_sql('gene_1010bp_individual_rate', data_con, index=False, if_exists='replace')    
    csv_fn = os.path.join(csv_folder, f'{prefix}.gene.1kb_individual_rate.raw.csv.gz')
    with gzip.open(csv_fn, 'wt', compresslevel=9) as filep:
        gene_rate_ind_pd.to_csv(filep, index=False, encoding='utf-8')
    
    gene_rate_smooth_pd.to_sql('gene_1000bp_smooth_rate', data_con, index=False, if_exists='replace')    
    csv_fn = os.path.join(csv_folder, f'{prefix}.gene.1kb_average_rate.smooth.csv')
    gene_rate_smooth_pd.to_csv(csv_fn, index=False) 
    logger.info('Average Methylation Rates for gene +/- 1 kb are calculated')
    gene_fraction_pd, gene_ind_pd = extract_gene_fraction(
        data_con, sample_pd, gene_pd)

    gene_fraction_pd.to_sql('gene_1010bp_average_fraction', data_con, index=False, if_exists='replace')    
    csv_fn = os.path.join(csv_folder, f'{prefix}.gene.1kb_average_fraction.raw.csv')
    gene_fraction_pd.to_csv(csv_fn, index=False)

    gene_ind_pd.to_sql('gene_1010bp_individual_fraction', data_con, index=False, if_exists='replace')    
    csv_fn = os.path.join(csv_folder, f'{prefix}.gene.1kb_individual_fraction.raw.csv.gz')
    with gzip.open(csv_fn, 'wt', compresslevel=9) as filep:
        gene_ind_pd.to_csv(filep, index=False, encoding='utf-8')    
    logger.info('Average Methylated Fractions for gene +/- 1 kb are extracted')    
    data_con.close()
