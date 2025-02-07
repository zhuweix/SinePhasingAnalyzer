# Sine Phasing Analyzer Documentation

A comprehensive Python CLI tool for performing phasing analysis using decaying sine model.

## Overview

Sine Phasing Analyzer is designed to process genomic data and perform methylation analysis through a series of commands that can be run individually or as a complete pipeline.

## Features

- BigWig file processing and SQLite database storage
- DNA methylation rate calculation
- Data extraction and processing
- Sine wave phasing analysis
- Visualization capabilities

## Installation

```bash
git clone https://github.com/zhuweix/SinePhasingAnalyzer.git
cd SinePhasingAnalyzer
pip install .
```

## Command Reference

### Full Pipeline

Run the complete analysis pipeline:

```bash
sinekit full CONFIG.TOML
```

The full command executes all steps in sequence:
1. Load BigWig files
2. Calculate methylation rates
3. Extract processed data
4. Perform phasing analysis
5. Generate plots

### Individual Commands

#### load
```bash
sinekit load CONFIG.TOML
```
- Imports BigWig files into SQLite database
- Performs data validation and compression
- Creates indexes for optimal performance

#### calc
```bash
sinekit calc CONFIG.TOML
```
- Calculates DNA methylation rates
- Processes sequencing data
- Applies quality thresholds

#### extract
```bash
sinekit extract CONFIG.TOML
```
- Extracts processed data
- Exports results after load and calc steps
- Prepares data for phasing analysis

#### phase
```bash
sinekit phase CONFIG.TOML [--quantilefn USER_QUANTILE.CSV]
```
- Performs sine wave phasing analysis
- Optional user-defined quantile assignments
- Processes gene methylation rate data

#### plot
```bash
sinekit plot CONFIG.TOML [--use-userquantile]
```
- Generates analysis figures
- Supports user-defined quantile plots
- Creates visualization of results

## Configuration

The tool uses TOML configuration files for all commands. Example configuration:



## Getting Help

For any command, use the --help option for detailed information:

```bash
sinekit --help
sinekit load --help
sinekit calc --help
# etc.
```