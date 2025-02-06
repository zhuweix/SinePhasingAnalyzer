"""Process BigWig files containing Methylation Fraction data.

This module transforms BigWig files containing Methylation Fraction data into a SQLite 
database for downstream analysis. The resulting database contains a combined spreadsheet 
with methylation data across all time points.

Key Features:
- Loads sample metadata from a configuration file
- Processes multiple BigWig files containing methylation data
- Combines data across different time points and replicates
- Stores results in a SQLite database with tables for:
    - Sample metadata
    - Chromosome sizes
    - Combined methylation data per experiment/replicate

Dependencies:
    - pandas: Data manipulation
    - pyBigWig: BigWig file processing
    - sqlite3: Database operations
    - numpy: Numerical operations
"""

import logging
import sqlite3
import os
from sinekit.utility import (load_sample_sheet,
    load_config_file, get_chrom_sizes)
import pandas as pd
import pyBigWig
import numpy as np

logger = logging.getLogger("sinekit")


def process_bigwig(config: str) -> None:
    """Process BigWig files and store methylation data in SQLite database.

    Args:
        config (str): Path to configuration file containing:
            - Sample sheet location
            - BigWig files directory
            - Output directory settings
            - Prefix for output files

    Raises:
        OSError: If BigWig file not found
        IOError: If unable to read BigWig file header

    The function performs the following operations:
    1. Loads sample metadata from configuration file
    2. Creates SQLite database with tables for:
        - SampleSheet: Sample metadata
        - ChromSize: Chromosome sizes from first BigWig file
        - {experiment}_{replicate}_combine: Methylation data per experiment/replicate
    3. Processes BigWig files by:
        - Reading methylation values per chromosome
        - Combining data across time points
        - Organizing data by experiment and replicate
    4. Stores processed data in SQLite database
    """

    config_dict = load_config_file(config)
    sample_fn = config_dict['data']['sample_sheet']
    sample_pd = load_sample_sheet(sample_fn)

    bw_dirn = config_dict['data']['bigwig_folder']

    output_folder = config_dict["output"]["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    prefix = config_dict["output"]["prefix"]
    data_db = os.path.join(output_folder, f'{prefix}.SQLite.db')
    data_con = sqlite3.connect(data_db)

    sample_pd.to_sql('SampleSheet', data_con, if_exists='replace', index=False)

    for _ ,row in sample_pd.iterrows():
        fn = row['filename']
        fn = os.path.join(bw_dirn, fn)    
        if not os.path.isfile(fn):
            msg = f"{fn} not found"
            logger.error(msg)
            raise OSError
    for (exp_, rep), tmp_pd in sample_pd.groupby(by=['experiment', 'rep'], sort=False):
        bw_fn = tmp_pd['filename'].values[0]
        bw_fn = os.path.join(bw_dirn, bw_fn)
        chrom_size_dict = get_chrom_sizes(data_con)
        tmp_pd = tmp_pd.sort_values(by='time')
        if chrom_size_dict is None:
            with pyBigWig.open(bw_fn) as bw:
                chrom_size_dict = bw.chroms()
                if chrom_size_dict is None:
                    msg = f'Error reading the header of {bw_fn}'
                    logger.error(msg)
                    raise IOError
                chrom_pd = pd.DataFrame({
                    'Chrom': list(chrom_size_dict.keys()), 
                    'Size': list(chrom_size_dict.values())})
                chrom_pd.to_sql("ChromSize", data_con, index=False, if_exists='replace')      
        isfirst = True
        for chrom, size in chrom_size_dict.items():
            tmp_chrom_pd = None
            for _, row in tmp_pd.iterrows():
                bw_fn = row['filename']
                bw_fn = os.path.join(bw_dirn, bw_fn)
                time_ = str(row['time'])
                with pyBigWig.open(bw_fn) as bw:
                    values = bw.values(chrom, 0, size, numpy=True)
                    pos = np.where(~np.isnan(values))[0]
                    if len(pos) == 0:
                        continue
                    values = values[pos]
                    tmp = pd.DataFrame({'Chrom': chrom, 'Pos': pos, time_: values})
                    if tmp_chrom_pd is None:
                        tmp_chrom_pd = tmp
                    else:
                        tmp_chrom_pd = pd.merge(tmp_chrom_pd, tmp, on=['Chrom', 'Pos'], how='outer',
                                                validate="one_to_one")
            tmp_chrom_pd.sort_values(by='Pos', inplace=True)
            if isfirst:
                tmp_chrom_pd.to_sql(f'{exp_}_{rep}_combine', data_con, if_exists='replace', index=False) 
                isfirst = False
            else:
                tmp_chrom_pd.to_sql(f'{exp_}_{rep}_combine', data_con, if_exists='append', index=False)
        logger.info("%s %s is loaded to SQLite database", exp_, rep)
    data_con.close()
    