import gc
import os
import logging
import pickle

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sinekit.utility import (
    fit_function, upper_function, lower_function)


# set default figure paremeters
plt.rcParams.update({
    'savefig.transparent': False,
    'figure.facecolor': 'w',
    'figure.dpi': 300,    
    'font.size': 12,          # Default text size
    'axes.titlesize': 14,     # Title size
    'axes.labelsize': 14,     # Label size (x and y labels)
    'xtick.labelsize': 12,    # X tick labels size
    'ytick.labelsize': 12,    # Y tick labels size
    'legend.fontsize': 12,    # Legend font size
    'figure.titlesize': 16    # Figure title size
})

logger = logging.getLogger("sinekit")


def plot_mt_timecourse(
    figure_prefix: str,
    mito_fn: str,
    xlabel: str,
    ylabel: str,
    xlim: list,
    sample_pd: pd.DataFrame) -> None:
    mito_pd = pd.read_csv(mito_fn, index_col=False)
    percentile_list = [5, 15, 25, 50, 75, 85, 95]
    
    for _, row in sample_pd.iterrows():
        exp_ = row['experiment']
        rep = row['rep']
        tmp_pd = mito_pd.loc[(mito_pd['Experiment'] == exp_) & (mito_pd['Rep'] == rep)]
        figure_fn = f'{figure_prefix}.Mitochondria.timecourse.{exp_}.{rep}.png'
        plt.figure(figsize=(4.5, 3.5))
        v50 = tmp_pd['50'].values
        time_list = tmp_pd['Time'].values
        g = sns.lineplot(x=time_list, y=v50, color='r', label='Median', zorder=9)
        for i in range(len(percentile_list)//2):
            alpha = .1 + .07*i
            plow = percentile_list[i]
            phigh = percentile_list[-i-1]
            vlow = tmp_pd[str(plow)].values
            vhigh = tmp_pd[str(phigh)].values       
            g.fill_between(x=time_list, y1=vlow, y2=vhigh, label=f'{plow}%-{phigh}%', color='r', alpha=alpha, zorder=9-i)
        g.legend()
        _ = g.set(xlim=xlim, ylim=[0, 100], yticks=np.arange(0, 101, 20), xticks=time_list,
                ylabel=ylabel,
                xlabel=xlabel, )
        plt.tight_layout()
        plt.savefig(figure_fn)
        plt.close()
    logger.info('mtDNA timecourse with quantiles are plotted')
    del mito_pd
    gc.collect()


def plot_feature_median_timecourse(
        figure_prefix: str,
        median_fn: str,
        xlabel: str,
        ylabel: str,
        xlim: list,
    ) -> None:
    median_pd = pd.read_csv(median_fn, index_col=False)
    for (exp_, rep), tmp_pd in median_pd.groupby(by=['Experiment', 'Rep'], sort=False):
        figure_name = f'{figure_prefix}.Feature.Median.timecourse.{exp_}.{rep}.png'
        plt.figure(figsize=(3.5, 3))
        time_list = None
        for feat in ['mtDNA', 'Genebody', 'NDR']:
            tmp_pd2 = tmp_pd.loc[tmp_pd['Feature'] == feat]
            xpos = tmp_pd2['Time'].values
            if time_list is None:
                time_list = xpos
            val = tmp_pd2['Median'].values
            g = sns.lineplot(x=xpos, y=val, label=feat)
        g.legend()
        _ = g.set(xlim=xlim, ylim=[0, 100], 
                 yticks=np.arange(0, 101, 20), xticks=time_list,
                xlabel=xlabel, ylabel=ylabel)
        plt.tight_layout()
        plt.savefig(figure_name)
        plt.close()
    del median_pd
    gc.collect()
    logger.info('Median plots of features are plotted.')


def plot_feature_rate_regression(
        figure_prefix: str,
        sample_pd: pd.DataFrame,
        rate_fn: str,
        xlabel: str,
        xlim: list,
    ) -> None:
    rate_pd = pd.read_csv(rate_fn, index_col=False)
    for (exp_, rep), tmp_pd in rate_pd.groupby(by=['Experiment', 'Rep'], sort=False):
        figure_name = f'{figure_prefix}.Feature.Rate.regression.{exp_}.{rep}.png'
        tmp_sample = sample_pd.loc[(sample_pd['experiment'] == exp_) & (sample_pd['rep'] == rep)]
        time_list = tmp_sample['time'].values
        time_list.sort()
        plt.figure(figsize=(3.5, 3))
        xticks = time_list[time_list > 0]
        for feat in ['mtDNA', 'Genebody', 'NDR']:
            tmp_pd2 = tmp_pd.loc[tmp_pd['Feature'] == feat]
            xpos = time_list[time_list > 0]
            val = [tmp_pd2[str(t)].values[0] for t in xpos]
            g = sns.scatterplot(x=xpos, y=val)
            x0 = xpos[0]
            x1 = xpos[-1]
            k = -tmp_pd2['Rate'].values[0]
            b = tmp_pd2['Intercept'].values[0]
            y0 = k*x0+b
            y1 = k*x1+b
            g = sns.lineplot(x=(x0, x1), y=(y0, y1), label=feat)

        g.legend()
        _ = g.set(xlim=xlim,  xticks=xticks, ylim=[min(-3.5, y1-.2), 0.1],
                xlabel=xlabel, ylabel='ln(1 - Fraction Methylated)')
        plt.tight_layout()
        plt.savefig(figure_name)
        plt.close()
    logger.info('mtDNR methylation rate plot is done.')

def plot_mtDNA_rate_boxplot(
    figure_prefix: str,
    mt_fn: str,
    ) -> None:
    figure_name = f'{figure_prefix}.MitochondriaMethylation.png'
    mito_pd = pd.read_csv(mt_fn, index_col=False)
    plt.figure(figsize=(4, 3.5))
    g = sns.boxplot(data=mito_pd, x='Experiment', y='Rate', showfliers=False, hue='Rep')
    _ = g.set(xlabel='', ylabel='Methylation Rate',)
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close()
    del mito_pd
    gc.collect()
    logger.info('Boxplot of mtDNA raw rates is plotted.')


def plot_phasing_rate_with_mnase(
    figure_prefix: str,
    phase_fn: str,
    mnase_fn: str,
    xlabel: str,
    ylabel: str,
    ) -> None:
    mnase_pd = pd.read_csv(mnase_fn, index_col=False)
    phase_pd = pd.read_csv(phase_fn, index_col=False)
    mnase_signal = mnase_pd['MNase'].values *.5
    mnase_pos = mnase_pd['Pos'].values
    for exp_, tmp_pd in phase_pd.groupby('Experiment', sort=False):
        figure_name = f'{figure_prefix}.Gene_Rate_Phasing.{exp_}.png'
        for rep, tmp_pd2 in tmp_pd.groupby("Rep", sort=False):
            xpos = tmp_pd2['Pos'].values
            rate = tmp_pd2['SmoothedValue'].values
            name = f'{exp_} {rep}'
            g = sns.lineplot(x=xpos, y=rate, label=name, lw=.5)
        g.fill_between(mnase_pos, y1=mnase_signal, 
                       y2=0, label='$\mathrm{MNase}$', color='.8')
        
        _ = g.set(xlim=[-1000, 1000], ylim=[0, 2], 
                xticks=np.arange(-1000, 1001, 250),
                xlabel=xlabel, ylabel=ylabel)
        g.legend(loc='upper right',  prop={'style': 'italic', 'size':9}, bbox_to_anchor=(1, 1))
        plt.subplots_adjust(left=0.17, right=.95, top=.98, bottom=.13)
        plt.savefig(figure_name)
        plt.close()
    del phase_pd
    del mnase_pd
    gc.collect()
    logger.info('Phasing plots of smoothed gene methylation rates are plotted.')


def plot_feature_relative_rate_boxplot(
    figure_prefix: str,
    csv_prefix: str,
    ) -> None:
    feature_list = ['NDR', 'Genebody', 'tRNA', 'ARS', 'TEL', 'Ty']
    for feat in feature_list:
        data_fn = f'{csv_prefix}.{feat}.raw_rate.csv.gz'
        data_pd = pd.read_csv(data_fn, index_col=False)
        figure_fn = f'{figure_prefix}.RelativeRate.{feat}.boxplot.png'
        plt.figure(figsize=(4, 3.5))
        g = sns.boxplot(data=data_pd, x='Experiment', y='Value', hue='Rep', 
                        showfliers=False)
        _ = g.set(ylim=(-.4, 3.5), xlabel='',
                  ylabel='Relative Methylation Rate',
                  title=feat) 
        plt.tight_layout()
        plt.savefig(figure_fn)
        plt.close()
        del data_pd
        gc.collect()
    logger.info('Boxplots of relative methylation rates are plotted')



def plot_sinewave_rate(
        figure_prefix: str,
        phase_fn: str,
        fit_pkl_fn: str,
        xlabel: str,
        ylabel: str,
        ylim: list,
    ) -> None:

    phase_pd = pd.read_csv(phase_fn, index_col=False)

    with open(fit_pkl_fn, 'rb') as filep:
        fit_result_dict = pickle.load(filep)

    for (exp_, rep), tmp_rate_pd in phase_pd.groupby(by=['Experiment', "Rep"], sort=False):
        figure_name = f'{figure_prefix}.Rate.SineWave.{exp_}.{rep}.png'
        popt = fit_result_dict[(exp_, rep)]['fit_params']
        xpos = tmp_rate_pd['Pos'].values
        x_fit = np.arange(-50, 1001)
        y_fit = fit_function(x_fit, *popt)        
        y = tmp_rate_pd['Value']
        g = sns.scatterplot(x=xpos, y=y, s=3, label='Data')
        g = sns.lineplot(x=x_fit, y=y_fit, label=f'Fitted', ax=g, color='red', lw=1)
        s_fit = popt[-1]
        b_fit = popt[-2]
        A_fit = popt[0]
        l_fit = popt[1] 
        bleft = x_fit[0] * s_fit + b_fit
        bright = x_fit[-1] * s_fit + b_fit  
        y_high = upper_function(x_fit, A_fit, l_fit, b_fit, s_fit)
        y_low = lower_function(x_fit, A_fit, l_fit, b_fit, s_fit)
        g = sns.lineplot(x=(x_fit[0], x_fit[-1]), y=(bleft, bright), lw=2, ls='--', color='.2', ax=g)
        g = sns.lineplot(x=x_fit, y=y_high, lw=2, ls='--', color='.5', ax=g)
        g = sns.lineplot(x=x_fit, y=y_low, lw=2, ls='--', color='.5', ax=g)
        _ = g.set(xlabel=xlabel,
                ylabel=ylabel,
                xlim=[-50, 1000], xticks=np.arange(0, 1001, 200),
                ylim=ylim, title=f'{exp_} {rep}')
        _ = g.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(figure_name)
        plt.close()   
    logger.info('Sine Wave figures for relative rates are plotted')     

def plot_sinewave_rate_quantile(
        figure_prefix: str,
        phase_fn: str,
        fit_pkl_fn: str,
        xlabel: str,
        ylabel: str,
        ylim: list,
        is_user: bool=False,
    ) -> None:
    phase_pd = pd.read_csv(phase_fn, index_col=False)

    with open(fit_pkl_fn, 'rb') as filep:
        fit_result_dict = pickle.load(filep)
    if not is_user:
        for (exp_, rep, q), tmp_rate_pd in phase_pd.groupby(
                by=['Experiment', "Rep", "Quantile"], sort=False):
            figure_name = f'{figure_prefix}.SineWaveQuantile.{exp_}.{rep}.{q}.png'
            popt = fit_result_dict[(exp_, rep, q)]['fit_params']
            xpos = tmp_rate_pd['Pos'].values
            x_fit = np.arange(-50, 1001)
            y_fit = fit_function(x_fit, *popt)        
            y = tmp_rate_pd['Value']
            g = sns.scatterplot(x=xpos, y=y, s=3, label='Data')
            g = sns.lineplot(x=x_fit, y=y_fit, label=f'Fitted', ax=g, color='red', lw=1)
            s_fit = popt[-1]
            b_fit = popt[-2]
            A_fit = popt[0]
            l_fit = popt[1] 
            bleft = x_fit[0] * s_fit + b_fit
            bright = x_fit[-1] * s_fit + b_fit  
            y_high = upper_function(x_fit, A_fit, l_fit, b_fit, s_fit)
            y_low = lower_function(x_fit, A_fit, l_fit, b_fit, s_fit)
            g = sns.lineplot(x=(x_fit[0], x_fit[-1]), y=(bleft, bright), lw=2, ls='--', color='.2', ax=g)
            g = sns.lineplot(x=x_fit, y=y_high, lw=2, ls='--', color='.5', ax=g)
            g = sns.lineplot(x=x_fit, y=y_low, lw=2, ls='--', color='.5', ax=g)
            _ = g.set(xlabel=xlabel,
                    ylabel=ylabel,
                    xlim=[-50, 1000], xticks=np.arange(0, 1001, 200),
                    ylim=ylim, title=f'{exp_} {rep}')
            _ = g.legend(markerscale=2)
            plt.tight_layout()
            plt.savefig(figure_name)
            plt.close()  
    else:
        for (name, exp_, rep, q), tmp_rate_pd in phase_pd.groupby(
                by=['GroupName', 'Experiment', "Rep", "Quantile"], sort=False):
            figure_name = f'{figure_prefix}.Rate.SineWaveUserQuantile.{exp_}.{rep}.{name}.{q}.png'
            popt = fit_result_dict[(name, exp_, rep, q)]['fit_params']
            xpos = tmp_rate_pd['Pos'].values
            x_fit = np.arange(-50, 1001)
            y_fit = fit_function(x_fit, *popt)        
            y = tmp_rate_pd['Value'].values
            g = sns.scatterplot(x=xpos, y=y, s=3, label='Data')
            g = sns.lineplot(x=x_fit, y=y_fit, label=f'Fitted', ax=g, color='red', lw=1)
            s_fit = popt[-1]
            b_fit = popt[-2]
            A_fit = popt[0]
            l_fit = popt[1] 
            bleft = x_fit[0] * s_fit + b_fit
            bright = x_fit[-1] * s_fit + b_fit  
            y_high = upper_function(x_fit, A_fit, l_fit, b_fit, s_fit)
            y_low = lower_function(x_fit, A_fit, l_fit, b_fit, s_fit)
            g = sns.lineplot(x=(x_fit[0], x_fit[-1]), y=(bleft, bright), lw=2, ls='--', color='.2', ax=g)
            g = sns.lineplot(x=x_fit, y=y_high, lw=2, ls='--', color='.5', ax=g)
            g = sns.lineplot(x=x_fit, y=y_low, lw=2, ls='--', color='.5', ax=g)
            _ = g.set(xlabel=xlabel,
                    ylabel=ylabel,
                    xlim=[-50, 1000], xticks=np.arange(0, 1001, 200),
                    ylim=ylim, title=f'{name} {exp_} {rep}')
            _ = g.legend(markerscale=2)
            plt.tight_layout()
            plt.savefig(figure_name)
            plt.close()        
    logger.info('Sine Wave figures for relative rates in quantiles are plotted')   

def plot_sinewave_fraction(
        figure_prefix: str,
        phase_fn: str,
        fit_pkl_fn: str,
        xlabel: str,
        ylabel: str,
        ylim: list,
    ) -> None:

    phase_pd = pd.read_csv(phase_fn, index_col=False)

    with open(fit_pkl_fn, 'rb') as filep:
        fit_result_dict = pickle.load(filep)

    for (exp_, time_, rep), tmp_rate_pd in phase_pd.groupby(by=['Experiment', 'Time', "Rep"], sort=False):
        figure_name = f'{figure_prefix}.MethylatedFraction.SineWave.{exp_}.{time_}.{rep}.png'
        popt = fit_result_dict[(exp_, time_, rep)]['fit_params']
        xpos = tmp_rate_pd['Pos'].values
        x_fit = np.arange(-50, 1001)
        y_fit = fit_function(x_fit, *popt)        
        y = tmp_rate_pd['Value']
        g = sns.scatterplot(x=xpos, y=y, s=3, label='Data')
        g = sns.lineplot(x=x_fit, y=y_fit, label=f'Fitted', ax=g, color='red', lw=1)
        s_fit = popt[-1]
        b_fit = popt[-2]
        A_fit = popt[0]
        l_fit = popt[1] 
        bleft = x_fit[0] * s_fit + b_fit
        bright = x_fit[-1] * s_fit + b_fit  
        y_high = upper_function(x_fit, A_fit, l_fit, b_fit, s_fit)
        y_low = lower_function(x_fit, A_fit, l_fit, b_fit, s_fit)
        g = sns.lineplot(x=(x_fit[0], x_fit[-1]), y=(bleft, bright), lw=2, ls='--', color='.2', ax=g)
        g = sns.lineplot(x=x_fit, y=y_high, lw=2, ls='--', color='.5', ax=g)
        g = sns.lineplot(x=x_fit, y=y_low, lw=2, ls='--', color='.5', ax=g)
        _ = g.set(xlabel=xlabel,
                ylabel=ylabel,
                xlim=[-50, 1000], xticks=np.arange(0, 1001, 200),
                ylim=ylim, title=f'{exp_} {time_} {rep}')
        _ = g.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(figure_name)
        plt.close()     
    logger.info('Sine Wave figures for methylated fractions are plotted')

    