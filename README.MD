# Sine Phasing Analyzer Documentation

A comprehensive Python CLI tool for performing phasing analysis using decaying sine model.

**[Overview](#overview) | [Installation](#installation) | [Commands](#commands) | [Citation](#citation)**

## Overview

Sine Phasing Analyzer is designed to process genomic data and perform methylation analysis through a series of commands that can be run individually or as a complete pipeline.

## Installation

This package requires Python 3.11 or later. Conda or Mamba is recommended for package management.

```bash
git clone https://github.com/zhuweix/SinePhasingAnalyzer.git
cd SinePhasingAnalyzer
pip install .
```

## Quick Start

Prepare the TOML configuration file and Sample Sheet CSV file. 
The example TOML file and Sample Sheet CSV file can be found in the `config` folder.


Run complete analysis pipeline:
```bash
sinekit full CONFIG.TOML
```
### Important Parameters in the TOML Configuration File
```toml
[data]
# Folder containing bigwig files
bigwig_folder = "<PATH TO BIGWIG FOLDER>"
# Sample sheet containing the metadata of the bigwig files
sample_sheet = "<PATH TO SAMPLE SHEET CSV>"

[output]
# prefix of the output files
# SQL database file will be named as <prefix>.db
# CSV files will be named as <prefix>.<table name>.csv
prefix = "<PREFIX OF OUTPUT FILES>"
# Folder to store the output files
output_folder = "<PATH TO OUTPUT FOLDER>"
```


## Commands

### load
Import genomic coverage data from BigWig files into SQLite database:
```bash
sinekit load CONFIG.TOML
```

### calc
Calculate DNA methylation rates at each genomic position:
```bash
sinekit calc CONFIG.TOML
```

### extract
Extract processed data after loading and calculation steps:
```bash
sinekit extract CONFIG.TOML
```

### phase
Perform sine wave phasing analysis:
```bash
sinekit phase CONFIG.TOML [OPTIONAL --quantilefn USER_QUANTILE.CSV]
```

Optional argument `--quantilefn` specifies the user-defined gene quantile file.
The user quantile file must be in CSV format and contain columns:
- GroupName: Name of the User-defined gene group
- Gene: Gene identifier
- Quantile: Quantile assignment


### plot
Generate analysis figures:
```bash
sinekit plot CONFIG.TOML [OPTIONAL --use-userquantile]
```
Boolean flag `--use-userquantile` specifies whether to use user-defined gene quantiles for plotting.
Default is --no-use-userquantile which uses the quantiles calculated during the phasing analysis.



## Citation

Zhuwei Xu et al. ,The ISW1 and CHD1 chromatin remodelers suppress global nucleosome dynamics in living yeast cells.Sci. Adv.11,eadw7108(2025).DOI:10.1126/sciadv.adw7108

## License

This project is licensed under the MIT License - see the [LICENSE]
