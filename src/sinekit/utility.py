"""
Utility functions for methylation data analysis and signal processing.

This module provides a collection of utility functions for processing and analyzing
methylation time series data, calculating kinetic rates, and performing signal
analysis. It includes functionality for:

- Configuration management and logging setup
- Sample sheet validation and loading
- Methylation rate calculations
- Sine wave fitting and analysis
- Chromosome size management

The module requires the following dependencies:
    - numpy
    - pandas
    - statsmodels
    - scipy
    - tomllib (Python 3.11+)
Author: Zhuwei Xu
Version: 1.0.0
Date: 25-02-03
"""

import logging
import logging.config
import sys
import os
import tomllib
from pathlib import Path

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger("sinekit")

def setup_logging():
    """
    Set up logging configuration with console and file handlers.
    
    Attempts to load logging configuration from 'logger.conf'. If that fails,
    falls back to a basic configuration with two handlers:
    - Console handler: Outputs simple format "[LEVEL]: message" to stdout
    - File handler: Writes detailed format "[timestamp] [name] LEVEL: message" 
      to 'sinekit.log'
    
    The logger is named "sinekit" and set to INFO level. Both handlers are 
    configured to match the settings specified in logger.conf.
    
    Notes
    -----
    - The configuration file path is resolved relative to this module's location
    - Existing loggers are preserved (disable_existing_loggers=False)
    - In case of configuration failure, a warning is logged and basic handlers
      are established
    - The fallback configuration mimics the file-based configuration for consistency
    
    Raises
    ------
    No exceptions are raised. Configuration errors are caught and logged, with
    the system falling back to a basic configuration.
    """
    try:
        config_path = os.path.join(Path(__file__).parent, 'logger.conf')
        logging.config.fileConfig(config_path, disable_existing_loggers=False)
    except Exception as e:
        # Fallback to basic configuration
        logger = logging.getLogger("sinekit")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Console handler with simple format
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("[%(levelname)s]:%(message)s")
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # File handler with detailed format
        file_handler = logging.FileHandler('sinekit.log', 'a')
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.warning(f"Failed to load logging config from {config_path}: {e}")


def load_config_file(config_fn: str, logger=None) -> dict:
    """
    Load and parse a TOML configuration file.
    
    Parameters
    ----------
    config_fn : str
        Path to the TOML configuration file
    logger : logging.Logger, optional
        Logger instance for debugging and error reporting
        
    Returns
    -------
    dict
        Parsed configuration dictionary
        
    Raises
    ------
    RuntimeError
        If file is not found, permission denied, or invalid TOML format
    """
    try:
        with open(config_fn, "rb") as filep:
            if logger:
                logger.debug(f"Loading config file: {config_fn}")
            config = tomllib.load(filep)
            if logger:
                logger.debug(f"Successfully loaded config from {config_fn}")
            return config
    except FileNotFoundError as e:
        msg = f'Config file not found: {config_fn}'
        if logger:
            logger.error(msg)
        raise RuntimeError(msg) from e
    except PermissionError as e:
        msg = f'Permission denied reading config file: {config_fn}'
        if logger:
            logger.error(msg)
        raise RuntimeError(msg) from e
    except tomllib.TOMLDecodeError as e:
        msg = f'Invalid TOML format in {config_fn}'
        if logger:
            logger.error(msg)
        raise RuntimeError(msg) from e



def load_sample_sheet(sample_fn: str) -> pd.DataFrame:
    """
    Load and validate a sample sheet containing experiment metadata.
    
    Parameters
    ----------
    sample_fn : str
        Path to the sample sheet CSV file
        
    Returns
    -------
    pd.DataFrame
        Validated sample sheet with required columns:
        - experiment: Experiment identifier
        - rep: Replicate number
        - time: Time point
        - filename: Associated data file
        
    Raises
    ------
    IOError
        If sample sheet file doesn't exist
    RuntimeError
        If CSV format is invalid or required data is missing
    ValueError
        If required columns are missing
    """
    if not os.path.isfile(sample_fn):
        raise IOError(f'Sample spreadsheet {sample_fn} is not found!')
    
    try:
        sample_pd = pd.read_csv(sample_fn, index_col=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {str(e)}")

    required_columns = ['experiment', 'rep', 'time', 'filename']
    assert all(col in sample_pd.columns for col in required_columns), \
        "Samplesheet must contain columns: 'experiment', 'rep', 'time', 'filename'"
        
    # Additional validations
    if sample_pd.empty:
        raise RuntimeError("Sample sheet is empty")

    # Check missing values
    missing_vals = sample_pd[required_columns].isnull().any()
    if missing_vals.any():
        missing_cols = missing_vals[missing_vals].index.tolist()
        raise RuntimeError(f"Missing values found in columns: {missing_cols}")

    return sample_pd


def get_chrom_sizes(sql_con, table_name: str = "ChromSize") -> pd.DataFrame | None:
    """
    Retrieve chromosome sizes from a SQL database.
    
    Parameters
    ----------
    sql_con : SQLAlchemy.Connection
        Active SQL database connection
    table_name : str, optional
        Name of the chromosome size table, defaults to "ChromSize"
        
    Returns
    -------
    dict or None
        Dictionary mapping chromosome names to sizes if table exists,
        None if table doesn't exist or query fails
    """
    try:
        tmp_pd = pd.read_sql_query(f'select * from {table_name}', sql_con)
        return dict(zip(tmp_pd['Chrom'], tmp_pd['Size']))
    except:  # Table not found
        return None


def calc_kinetic_rate(data_row, time_points) -> tuple[np.float64, np.float64, np.float64]:
    """
    Calculate kinetic rate constant using linear regression.
    
    Parameters
    ----------
    data_row : numpy.ndarray
        1D array containing methylation data points
    time_points : numpy.ndarray
        1D array containing corresponding time points
        
    Returns
    -------
    tuple[np.float64, np.float64, np.float64]
        Tuple containing:
        - Negative slope (rate constant)
        - Y-intercept
        - R-squared value of the fit
        
    Notes
    -----
    Returns (nan, nan, nan) if insufficient valid data points or NaN values present.
    """
    valid_pos = ~np.isnan(data_row)
    time_points = time_points[valid_pos]
    if time_points.shape[0] < 2:
        return np.nan, np.nan, np.nan
    data_row = data_row[valid_pos]
    X = time_points
    y = data_row.reshape(-1, 1)
    if np.any(np.isnan(y)):
        return np.nan, np.nan, np.nan

    X = sm.add_constant(X)
    result = sm.OLS(y, X).fit()
    r2 = result.rsquared

    c, k = result.params
    return -k, c, r2

def calc_methylation_rate(data_pd, data_points, cutoff=.99) -> pd.DataFrame:
    """Calculate methylation rates from time series data using linear regression.

    This function processes methylation time series data to calculate reaction rates
    by fitting a linear model to the log-transformed methylation values. It handles
    saturation effects by applying a cutoff and special processing for multiple
    saturation points.

    Parameters
    ----------
    data_pd : pd.DataFrame
        Input DataFrame containing methylation data with columns:
        - 'Chrom': Chromosome identifier
        - 'Pos': Position on chromosome
        - Time point columns named by values in data_points
        Values in time point columns should be percentages (0-100)
    
    data_points : list
        List of time points corresponding to columns in data_pd.
        Values should be in the same units as your time measurements.
        Zero time point (<1e-6) is filtered out.
    
    cutoff : float, optional
        Maximum allowed methylation rate (as fraction, not percentage).
        Values above this are capped and treated as saturation.
        Default is 0.99 (99%).

    Returns
    -------
    pd.DataFrame
        DataFrame containing calculated rates with columns:
        - 'Chrom': Chromosome identifier
        - 'Pos': Position on chromosome
        - 'Rate': Calculated methylation rate (k)
        - 'Intercept': Y-intercept of fitted line (c)
        - 'RSquared': R-squared value of fit
        - Additional columns for log-transformed values at each time point
    """   
    xpos = np.array([da for da in data_points if da > 1e-6])
    rate_pd = []
    for _, row in data_pd.iterrows():
        values = []
        max_val = 0
        chrom = row['Chrom']
        pos = row['Pos']
        for da in data_points:
            if da < 1e-6:
                continue
            val = row[f'{da}'] / 100
            if val > cutoff:
                val = cutoff
                max_val += 1
            values.append(val)
        values = np.array(values)
        if max_val > 1:
            max_pos = np.where(values>=cutoff)[0]
            values[max_pos[1]:] = np.nan
        values = np.log(1 - np.array(values))
        k, c, r2 = calc_kinetic_rate(values, xpos)
        row = [chrom, pos, k, c, r2] + list(values)
        rate_pd.append(row)
    columns = ['Chrom', 'Pos', 'Rate', 'Intercept', 'RSquared'] + [f'{x}' for x in xpos]
    rate_pd = pd.DataFrame(rate_pd, columns=columns)
    return rate_pd

def fit_function(x, A, l, w_0, x_0, b, s):
    """
    Damped sine wave function with linear trend.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input time points
    A : float
        Amplitude
    l : float
        Decay constant
    w_0 : float
        Angular frequency
    x_0 : float
        Phase offset
    b : float
        Vertical offset
    s : float
        Linear slope
        
    Returns
    -------
    numpy.ndarray
        Function values at input points
    """
    return A * np.exp(-l * x) * np.sin(w_0  * x + x_0) + b + s*x

def upper_function(x, A, l, b, s):
    return A * np.exp(-l *x) + b + s*x

def lower_function(x, A, l, b, s):
    return -A * np.exp(-l *x) + b + s*x

def calc_sine_fit(y: np.ndarray, xpos: np.ndarray) -> dict:
    """
    Fit a damped sine wave with linear trend to experimental data.
    
    This function fits data to the model:
    y(x) = A * exp(-λx) * sin(ω₀x + θ₀) + b₀ + sx
    where:
    - A: amplitude
    - λ: decay rate
    - ω₀: angular frequency
    - θ₀: phase offset
    - b₀: baseline offset
    - s: linear slope
    
    Parameters
    ----------
    y : numpy.ndarray
        Input measurements (dependent variable).
        Should be a 1D array of float values.
    
    xpos : numpy.ndarray
        Position or time points (independent variable).
        Should be a 1D array of the same length as y.
        
    Returns
    -------
    dict
        A dictionary containing fit results and statistics:
        - Adj.R2 : float
            Adjusted R-squared value of the fit
        - Spacing : float
            Period of oscillation (2π/ω₀)
        - Error_spacing : float
            Standard error of the spacing
        - Adj.Mean : float
            Mean of fitted values
        - Amplitude : float
            Fitted amplitude (A)
        - Error_Amp : float
            Standard error of amplitude
        - Slope : float
            Linear slope in per kb (s * 1000)
        - Error_Slope : float
            Standard error of slope
        - Decay : float
            Decay factor per period (exp(-λ * spacing))
        - b0 : float
            Baseline offset
        - theta0 : float
            Phase offset in radians
        - fit_params : array
            All fitted parameters [A, λ, ω₀, θ₀, b₀, s]
        - fit_errors : array
            Standard errors for all parameters
        
    Notes
    -----
    Initial parameter guesses:
    - Amplitude (A): 110% of maximum y value
    - Period (2π/ω₀): 160 units
    - Phase offset (θ₀): -π/2
    - Baseline (b₀): Mean of y values
    - Decay rate (λ): Default 1/160
    - Slope (s): 0
    
    The function uses scipy.optimize.curve_fit for parameter optimization
    and calculates standard errors from the covariance matrix.
    
    Statistical measures:
    - R² is calculated as 1 - SSR/SST
    - Adjusted R² accounts for the number of parameters
    - Standard errors are derived from the diagonal of the covariance matrix
    """
    max_a = np.max(y) * 1.1
    guess_w_0 = 2*np.pi / 160
    guess_x_0 = -np.pi/2
    guess_b = np.mean(y)
    initial_guess = [max_a, 1/160, guess_w_0, guess_x_0, guess_b, 0]   
    try:
        popt, pcov = curve_fit(fit_function, xpos, y, p0=initial_guess)
    except (RuntimeError, ValueError) as e:
        print(f"Fitting error: {type(e).__name__}: {str(e)}")
        result_dict = {
            'Adj.R2': np.nan,
            'Spacing': np.nan,
            'Error_spacing': np.nan,
            'Adj.Mean': np.nan,
            'Amplitude': np.nan,
            'Error_Amp': np.nan,
            'Slope': np.nan,
            'Error_Slope': np.nan,
            'Decay': np.nan,
            'b0': np.nan,
            'theta0': np.nan,
            'fit_params': np.array([np.nan] * 6),
            'fit_errors': np.array([np.nan] * 6)
        }        
        return result_dict
    y_fit = fit_function(xpos, *popt)    
    # Extract parameters
    _, l_fit, w_0_fit, theta0_fit, _, _ = popt
    
    # Calculate statistics
    spacing = 2*np.pi / w_0_fit
    sst = np.sum((y-np.mean(y))**2)
    ssr = np.sum((y-y_fit)**2)
    r2 = 1 - ssr/sst
    adj_r2 = 1 - (1-r2)*(len(y)-1)/(len(y)-len(initial_guess)-1)
    decay = np.exp(-l_fit * spacing)
    
    # Calculate errors
    perr = np.sqrt(np.diag(pcov))
    s_fit = popt[-1] * 1000
    A_fit = popt[0]
    b_fit = popt[-2]
    adj_mean = np.mean(y_fit)
    err_A = perr[0]
    err_s = perr[-1]*1000
    err_w0 = perr[2]
    err_spacing = 2*np.pi / (w_0_fit**2)*err_w0
    
    # Compile results
    result_dict = {
        'Adj.R2': adj_r2,
        'Spacing': spacing,
        'Error_spacing': err_spacing,
        'Adj.Mean': adj_mean,
        'Amplitude': A_fit,
        'Error_Amp': err_A,
        'Slope': s_fit,
        'Error_Slope': err_s,
        'Decay': decay,
        'b0': b_fit,
        'theta0': theta0_fit,
        'fit_params': popt,
        'fit_errors': perr
    }
    return result_dict


def calculate_adj_gene_level(y: np.ndarray, xpos: np.ndarray, fit_params: np.ndarray) -> tuple[float, float]:
    """
    Calculate the adjusted gene level and goodness of fit using a fitted function.
    
    This function computes the difference between observed values and a fitted curve,
    returning both the mean adjusted rate and the R-squared value that indicates
    how well the fit explains the data. The adjustment accounts for systematic
    trends in the data as described by the fitted function.
    
    Parameters
    ----------
    y : numpy.ndarray
        Observed values for the gene methylation rates / methylated fractions
    xpos : numpy.ndarray
        Position values corresponding to the observations
    fit_params : numpy.ndarray
        Parameters for the fitted function, used to generate the expected values
        
    Returns
    -------
    adj_rate : float
        Mean adjusted rate, calculated as the average difference between observed
        values and the fitted curve
    r2 : float
        R-squared value (coefficient of determination) indicating goodness of fit,
        ranges from 0 to 1 where 1 indicates perfect fit
        
    Notes
    -----
    The R-squared value is calculated using the formula:
    R² = 1 - SSR/SST
    where:
    - SSR is the sum of squared residuals after adjustment
    - SST is the total sum of squares
    
    The adjusted rate represents the average deviation of the observed values
    from the fitted trend, which can be interpreted as the baseline level
    after removing systematic variations.
    """
    popt = fit_params
    def fit_f(x):
        return fit_function(x, *popt)
    y_fit = fit_f(xpos)
    if not np.any(~np.isnan(y-y_fit)):
        return np.nan, np.nan
    adj_rate = np.nanmean(y - y_fit)
    sst = np.sum((y - np.mean(y))**2)
    ssr = np.sum((y-y_fit-adj_rate)**2)
    if np.abs(sst) < np.finfo(float).eps:
        r2 = np.nan
    else:
        r2 = 1 - ssr/sst
    return adj_rate, r2


