'''CLI entry of the Sine Phasing Analyzer program'''
import logging
from pathlib import Path
from typing import Annotated, Optional
import typer
from sinekit.utility import setup_logging
from sinekit.process_bw import process_bigwig
from sinekit.calculate_rate import calculate_rate
from sinekit.extract import extract_all
from sinekit.phasing import sinekit_analysis


setup_logging()
logger = logging.getLogger("sinekit")

# Set up app
app = typer.Typer(
    help='''Phasing analysis using decaying sine model.
Execute full Command for full analysis or execute 
load, calc, extract, phase, plot in sequence.\n
Example Usage: sinekit full [CONFIG.TOML]''',
    name="Sine Phasing Analyzer"
)

@app.callback()
def callback():
    '''Phasing analysis using decaying sine model.'''
    pass


@app.command(help='Load genomic data from BigWig files into SQLite database')
def load(
    config_file: Annotated[Path, typer.Argument(
        help="Path to configuration file in TOML format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True)]
) -> None:
    """
    Import genomic coverage data from BigWig files into a SQLite database.

    This command processes BigWig files containing genomic methylated fraction data and stores
    them in a SQLite database for efficient querying and analysis. The process 
    includes data validation, compression, and indexing for optimal performance.

    Args:
        config_file: Path to TOML configuration file specifying input and output parameters

    Notes:
        - BigWig files must be properly formatted and indexed
        - Sufficient disk space is required for the SQLite database
        - The process may take significant time for large datasets
        - Consider backing up existing database before running
    """
    if not config_file.is_file():
        raise typer.BadParameter(f'{config_file} does not exist!')

    process_bigwig(config=config_file)
    logger.info('Bigwig files are successfully loaded.')    

@app.command(help='Calculate DNA methylation rates from sequencing data')
def calc(
    config_file: Annotated[Path, typer.Argument(
        help="Path to configuration file in TOML format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True)]
) -> None:
    """
    Calculate DNA methylation rates at each genomic position.

    This command processes sequencing data to compute methylation rates
    across the genome. It requires a TOML configuration file that specifies
    input data locations, processing parameters, and output preferences.

    Args:
        config_file: Path to TOML configuration file specifying analysis parameters

    Configuration file must include:
    - Input data paths
    - Quality thresholds for methylation calling
    - Output file specifications
    - Processing options for rate calculation

    Note: Input data must be loaded before running this calculation step.
    """
    if not config_file.is_file():
        raise typer.BadParameter(f'{config_file} does not exist!')

    calculate_rate(config=config_file)
    logger.info('Methylation rates are successfully calculated.')    

@app.command(help='Extract processed data after load and calculation steps')
def extract(
    config_file: Annotated[Path, typer.Argument(
        help="Path to configuration file in TOML format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True)]
) -> None:
    """
    Extract and export processed data after running load and calc commands.
    
    Args:
        config_file: Path to TOML configuration file specifying extraction parameters
        
    Note:
        This command should only be run after successfully completing the 'load' 
        and 'calc' steps of the pipeline.
    """   
    try:
        extract_all(config=config_file)
        logger.info('Processed data are extracted.')        
    except Exception as e:
        raise typer.BadParameter(f'Data extraction failed: {str(e)}')
    
@app.command(help='Sine Wave Phasing Analysis')
def phase(
    config_file: Annotated[Path, typer.Argument(
        help="Path to configuration file in TOML format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True)],
    user_quantile_fn: Annotated[Optional[Path], typer.Option(
        "--quantilefn", "-q",
        help="Optional: Path to user-defined quantile assignments in CSV format.\n"
             "Must contain columns: GroupName, Gene, Quantile",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True)] = None
) -> None:
    """
    Perform sine wave phasing analysis on gene expression data.
    
    Args:
        config_file: Path to TOML configuration file containing analysis parameters
        user_quantile_fn: Optional path to CSV file with user-defined quantile assignments
    
    If user_quantile_fn is provided, it must be a CSV file with columns:
    - GroupName: Name of the gene group
    - Gene: Gene identifier
    - Quantile: Quantile assignment
    """

    try:
        if user_quantile_fn is None:
            sinekit_analysis(config=config_file)
        else:
            sinekit_analysis(config=config_file, user_quantile_fn=user_quantile_fn)
        logger.info('Phasing analysis is successful.')            
    except Exception as e:
        raise typer.BadParameter(f'Analysis failed: {str(e)}')

@app.command(help='Run complete methylation analysis pipeline')
def full(
    config_file: Annotated[Path, typer.Argument(
        help="Path to configuration file in TOML format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True)]
) -> None:
    """
    Execute the complete methylation analysis pipeline in sequence.

    This command runs the full analysis pipeline by executing the following steps:
    1. Load BigWig files into SQLite database
    2. Calculate methylation rates
    3. Extract processed data
    4. Perform sine wave phasing analysis

    Each step builds upon the results of the previous steps, so they must be
    executed in order. The pipeline handles all necessary data transformations
    and validations between steps.

    Args:
        config_file: Path to TOML configuration file containing all pipeline parameters

    Configuration Requirements:
        The TOML file must include settings for all pipeline stages:
        - BigWig file processing
        - Methylation rate calculation
        - Data extraction parameters
        - Phasing analysis configuration

    Resource Requirements:
        - Sufficient disk space for intermediate files
        - Adequate memory for processing large datasets
        - BigWig files must be properly formatted
        - Write permissions in output directories

    Output:
        - SQLite database with processed data
        - CSV files with analysis results
        - Methylation rate calculations
        - Phasing analysis results
        - Log files documenting the process
    """
    process_bigwig(config=config_file)
    logger.info('Bigwig files are successfully loaded.')
    calculate_rate(config=config_file)
    logger.info('Methylation rates are successfully calculated.') 
    extract_all(config=config_file)
    logger.info('Processed data are extracted.')
    sinekit_analysis(config=config_file)
    logger.info('Phasing analysis is successful.')
    
if __name__ == "__main__":
    app()