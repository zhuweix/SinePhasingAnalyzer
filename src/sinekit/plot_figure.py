import gc
import os
import logging
import pickle

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sinekit.utility import (
    load_config_file, load_sample_sheet,
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
        g.legend(loc='upper left', bbox_to_anchor=(1,1))
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
        time_list = None
        for feat in ['mtDNA', 'Genebody', 'NDR']:
            tmp_pd2 = tmp_pd.loc[tmp_pd['Feature'] == feat]
            xpos = tmp_pd2['Time'].values
            if time_list is None:
                time_list = xpos
            val = tmp_pd2['Median'].values
            g = sns.lineplot(x=xpos, y=val, label=feat)
        g.legend(loc='upper left', bbox_to_anchor=(1,1))
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

        g.legend(loc='upper left', bbox_to_anchor=(1,1))
        _ = g.set(xlim=[xpos[0] *.9, xpos[-1]*1.1],  xticks=xticks, ylim=[min(-3.5, y1-.2), 0.1],
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
    g = sns.boxplot(data=mito_pd, x='Experiment', y='Raw Rate', showfliers=False, hue='Rep')
    _ = g.set(xlabel='', ylabel='Methylation Rate',)
    sns.move_legend(g, loc='upper left', bbox_to_anchor=(1,1))
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
    ylim: list,
    xlim: list,
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
        
        _ = g.set(xlim=xlim, ylim=ylim, 
                xticks=np.arange(-1000, 1001, 250),
                xlabel=xlabel, ylabel=ylabel)
        g.legend(loc='upper right',  prop={'size':9}, bbox_to_anchor=(1, 1))
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
        g = sns.boxplot(data=data_pd, x='Experiment', y='Relative Rate', hue='Rep', 
                        showfliers=False)
        _ = g.set(xlabel='',
                  ylabel='Relative Methylation Rate',
                  title=feat) 
        sns.move_legend(g, loc='upper left', bbox_to_anchor=(1,1))                  
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
                    ylim=ylim, title=f'{exp_} {rep} {q}')
            _ = g.legend(markerscale=2)
            plt.tight_layout()
            plt.savefig(figure_name)
            plt.close()  
    else:
        for (name, exp_, rep, q), tmp_rate_pd in phase_pd.groupby(
                by=['GroupName', 'Experiment', "Rep", "Quantile"], sort=False):
            figure_name = f'{figure_prefix}.Rate.SineWaveUserQuantile.{name}.{exp_}.{rep}.{name}.{q}.png'
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
                    ylim=ylim, title=f'{exp_} {rep} {q}')
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

    for (exp_, rep, time_), tmp_rate_pd in phase_pd.groupby(by=['Experiment', 'Time', "Rep"], sort=False):
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

def plot_sinewave_fraction_qiuantile(
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
        for (exp_, rep, time_, q), tmp_rate_pd in phase_pd.groupby(by=['Experiment', 'Time', "Rep", "Quantile"], sort=False):
            figure_name = f'{figure_prefix}.MethylatedFraction.SineWaveQuantile.{exp_}.{time_}.{rep}.{q}.png'
            popt = fit_result_dict[(exp_, time_, rep, q)]['fit_params']
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
                    ylim=ylim, title=f'{exp_} {time_} {rep} {q}')
            _ = g.legend(markerscale=2)
            plt.tight_layout()
            plt.savefig(figure_name)
            plt.close()
    else:
        for (name, exp_, rep, time_, q), tmp_rate_pd in phase_pd.groupby(by=['GroupName', 'Experiment', 'Time', "Rep", "Quantile"], sort=False):
            figure_name = f'{figure_prefix}.MethylatedFraction.UserSineWaveQuantile.{name}.{exp_}.{time_}.{rep}.{q}.png'
            popt = fit_result_dict[(name, exp_, time_, rep, q)]['fit_params']
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
                    ylim=ylim, title=f'{exp_} {time_} {rep} {q}')
            _ = g.legend(markerscale=2)
            plt.tight_layout()
            plt.savefig(figure_name)
            plt.close()
    logger.info('Sine Wave figures for methylated fractions in quantiles are plotted')


def plot_trna_timecourse(
    figure_prefix: str,
    trna_fn: str,
    xlabel: str,
    ylabel: str,
    xlim: list,
    sample_pd: pd.DataFrame) -> None:
    trna_pd = pd.read_csv(trna_fn, index_col=False)
    percentile_list = [5, 15, 25, 50, 75, 85, 95]
    feat_list = ['TFIIIB', 'TFIIIC', 'Intron']
    for _, row in sample_pd.iterrows():
        exp_ = row['experiment']
        rep = row['rep']
        tmp_pd = trna_pd.loc[(trna_pd['Experiment'] == exp_) & (trna_pd['Rep'] == rep)]
        for feat in feat_list:
            tmp_pd2 = tmp_pd.loc[tmp_pd['Feature'] == feat]
            figure_fn = f'{figure_prefix}.tRNA.{feat}.timecourse.{exp_}.{rep}.png'
            plt.figure(figsize=(4.5, 3.5))
            v50 = tmp_pd2['50'].values
            time_list = tmp_pd2['Time'].values
            g = sns.lineplot(x=time_list, y=v50, color='r', label='Median', zorder=9)
            for i in range(len(percentile_list)//2):
                alpha = .1 + .07*i
                plow = percentile_list[i]
                phigh = percentile_list[-i-1]
                vlow = tmp_pd2[str(plow)].values
                vhigh = tmp_pd2[str(phigh)].values       
                g.fill_between(x=time_list, y1=vlow, y2=vhigh, label=f'{plow}%-{phigh}%', color='r', alpha=alpha, zorder=9-i)
            g.legend()
            _ = g.set(xlim=xlim, ylim=[0, 100], yticks=np.arange(0, 101, 20), xticks=time_list,
                    ylabel=ylabel,
                    xlabel=xlabel, )
            plt.tight_layout()
            plt.savefig(figure_fn)
            plt.close()
    logger.info('tRNA timecourse with quantiles are plotted')
    del trna_pd
    gc.collect()


def plot_trna_rate_regression(
        figure_prefix: str,
        sample_pd: pd.DataFrame,
        rate_fn: str,
        xlabel: str,
        xlim: list,
    ) -> None:
    rate_pd = pd.read_csv(rate_fn, index_col=False)
    feat_list = ['TFIIIB', 'TFIIIC', 'Intron']
    for (exp_, rep), tmp_pd in rate_pd.groupby(by=['Experiment', 'Rep'], sort=False):
        figure_name = f'{figure_prefix}.tRNA.Rate.regression.{exp_}.{rep}.png'
        tmp_sample = sample_pd.loc[(sample_pd['experiment'] == exp_) & (sample_pd['rep'] == rep)]
        time_list = tmp_sample['time'].values
        time_list.sort()
        plt.figure(figsize=(3.5, 3))
        xticks = time_list[time_list > 0]
        for feat in feat_list:
            tmp_pd2 = tmp_pd.loc[tmp_pd['Feature'] == feat]
            xpos = time_list[time_list > 0]
            val = [tmp_pd2[str(t)].values[0] for t in xpos]
            g = sns.scatterplot(x=xpos, y=val)
            x0 = xpos[0]
            x1 = xpos[-1]
            k = -tmp_pd2['Rate'].values[0]
            b = tmp_pd2['Intercept'].values[0]
            r = tmp_pd2['Relative Rate'].values[0]
            y0 = k*x0+b
            y1 = k*x1+b
            g = sns.lineplot(x=(x0, x1), y=(y0, y1), label=f'{feat} ({r:.1f})')

        g.legend(title='Feature (Rel Rate)')
        _ = g.set(xlim=[xpos[0] *.9, xpos[-1]*1.1],  xticks=xticks, ylim=[min(-2, y1-.2), 0.1],
                xlabel=xlabel, ylabel='ln(1 - Fraction Methylated)')
        plt.tight_layout()
        plt.savefig(figure_name)
        plt.close()
    logger.info('tRNA methylation rate plot is done.')

def plot_trna_rate_visual(
        figure_prefix: str,
        rate_fn: str
    ) -> None:
    left_flank = 200
    right_flank = 200
    b_left = -56
    b_right = -13
    c_left = -2
    c_right = 10
    trna_single_pd = pd.read_csv(rate_fn, index_col=False)
    for (exp_, rep), tmp_pd in trna_single_pd.groupby(by=['Experiment', 'Rep'], sort=False):
        xpos = tmp_pd['Pos'].values
        tmp_ave = tmp_pd['Average'].values
        tmp_std = tmp_pd['Std'].values
        tmp_up = tmp_ave + tmp_std
        tmp_down = tmp_ave - tmp_std         
        figure_name = f'{figure_prefix}.tRNA.Map.{exp_}.{rep}.png'
        
        plt.figure(figsize=(4, 3))
        g = sns.lineplot(x=xpos, y=tmp_ave, linewidth=.7, markers='x', label='Mean')
        g.fill_between(xpos, tmp_down, tmp_up, alpha=0.4, color='red', label='$\pm$Std', ec='None')
        g.axvline(b_left, color='.7', lw=1, ls='--')
        g.axvline(b_right, color='.7', lw=1, ls='--')    
        g.annotate('TFIIIB', ((b_left+b_right)/2, 4), fontsize=12, ha='center')
        cl = c_left
        cr = len(tmp_ave) - left_flank - right_flank + c_right
        g.axvline(cl, color='.7', lw=1, ls='--')
        g.axvline( cr, color='.7', lw=1, ls='--')    
        g.annotate('TFIIIC', ((cl+cr)/2, 4), fontsize=12, ha='center')
        g.set(xlim=(-left_flank, len(tmp_ave)-left_flank), xlabel='Relative to TSS (bp)', ylabel='Relative Methylation Rate',
            ylim=[0, 5], title=f'                   {exp_} {rep}')
        g.legend(loc='lower left', fontsize='9', bbox_to_anchor=(0, 1), ncols=2)

        plt.tight_layout()
        plt.savefig(figure_name)
        plt.close()
    logger.info('tRNA maps are plotted.')


def plot_gene_rate_group_by_length_rnaseq(
    figure_prefix: str,
    gene_adj_fn: str,
    rnaseq_fn: str,
    ) -> None:

    rnaseq_pd = pd.read_csv(rnaseq_fn, index_col=False)
    gene_length_dict = dict(zip(rnaseq_pd['Gene'], rnaseq_pd['GeneLength']))
    gene_tpm_dict = dict(zip(rnaseq_pd['Gene'], rnaseq_pd['TPM_Ave']))    
    gene_adj_pd = pd.read_csv(gene_adj_fn, index_col=False)
    gene_adj_pd['Length']  = gene_adj_pd['Gene'].apply(lambda x: gene_length_dict.get(x, np.nan))
    gene_adj_pd['TPM']  = gene_adj_pd['Gene'].apply(lambda x: gene_tpm_dict.get(x, np.nan))
    for exp_, tmp_pd in gene_adj_pd.groupby('Experiment', sort=False):
        # gene group vs Length
        figure_name = f'{figure_prefix}.Gene_RateGroup.GeneLength.{exp_}.png'
        q_list = list(tmp_pd['Quantile'].unique())
        q_list.sort()
        g = sns.boxplot(tmp_pd, x='Quantile', y='Length', hue='Rep', showfliers=False, 
                        order=q_list)
        _ = g.set(xlabel='', ylabel='Gene Length (bp)',)
        _ = g.legend()
        plt.tight_layout()
        plt.savefig(figure_name)
        plt.close() 
    # gene group vs TPM
        figure_name = f'{figure_prefix}.Gene_RateGroup.rnaseq.{exp_}.png'
        q_list = list(tmp_pd['Quantile'].unique())
        q_list.sort()
        g = sns.boxplot(tmp_pd, x='Quantile', y='TPM', hue='Rep', showfliers=False, 
                        order=q_list)
        _ = g.set(xlabel='', ylabel='Gene expression level (TPM)',)
        _ = g.legend()
        plt.tight_layout()
        plt.savefig(figure_name)
        plt.close() 
                            



def plot_figures(config: str, is_user: bool = False):
    config_dict = load_config_file(config)
    sample_fn = config_dict['data']['sample_sheet']
    sample_pd = load_sample_sheet(sample_fn)

    mnase_fn = config_dict['resource']['mnase_table']
    rnaseq_fn = config_dict['resource']['rnaseq_table']
    timecourse_xlabel = config_dict['plot_parameter']['timecourse_xlabel']
    timecourse_frac_ylabel = config_dict['plot_parameter']['timecourse_frac_ylabel']
    timecourse_xlim = config_dict['plot_parameter']['timecourse_xlim']
    phasing_xlabel = config_dict['plot_parameter']['phasing_xlabel']
    phasing_xlim = config_dict['plot_parameter']['phasing_xlim']
    phasing_ylabel = config_dict['plot_parameter']['phasing_ylabel']    
    phasing_ylim = config_dict['plot_parameter']['phasing_ylim']
    phasing_ylim_sinewave = config_dict['plot_parameter']['phasing_ylim_sinewave']

    prefix = config_dict["output"]["prefix"]
    output_folder = config_dict["output"]["output_folder"]    
    csv_folder = os.path.join(output_folder, 'Spreadsheet')
    fit_folder = os.path.join(output_folder, 'FitResult')    
    figure_folder = os.path.join(output_folder, 'Figure')
    os.makedirs(figure_folder, exist_ok=True)

    #mtDNA figures
    subfig_folder = os.path.join(figure_folder, 'mtDNA_Related')
    os.makedirs(subfig_folder, exist_ok=True)
    figure_prefix = os.path.join(subfig_folder, prefix)
    # timecourse
    mito_fn = os.path.join(csv_folder, f'{prefix}.mtDNA.timecourse.percentile.csv')
    plot_mt_timecourse(
        figure_prefix, mito_fn,
        timecourse_xlabel,
        timecourse_frac_ylabel,
        timecourse_xlim,
        sample_pd)
    # median curve
    median_fn = os.path.join(csv_folder, f'{prefix}.mtDNA_genebody_NDR.timecourse.median.csv')
    plot_feature_median_timecourse(
        figure_prefix,
        median_fn,
        timecourse_xlabel,
        timecourse_frac_ylabel,
        timecourse_xlim,
    )

    # median rate
    rate_fn = os.path.join(csv_folder, f'{prefix}.mtDNA_genebody_NDR.rate.median.csv')
    plot_feature_rate_regression(
        figure_prefix,
        sample_pd,
        rate_fn,
        timecourse_xlabel,
        timecourse_xlim,
    )

    # rate boxplot
    rate_fn = os.path.join(csv_folder, f'{prefix}.mtDNA.raw_rate.csv.gz')
    plot_mtDNA_rate_boxplot(
    figure_prefix, rate_fn)

    # Phasing plots
    subfig_folder = os.path.join(figure_folder, 'Phasing_MNase')
    os.makedirs(subfig_folder, exist_ok=True)

    figure_prefix = os.path.join(subfig_folder, prefix)
    phase_fn = os.path.join(csv_folder, f'{prefix}.gene.1kb_average_rate.smooth.csv')
    plot_phasing_rate_with_mnase(
        figure_prefix, phase_fn,
        mnase_fn, phasing_xlabel, phasing_ylabel, phasing_ylim, phasing_xlim)
    
    # feature boxplot
    subfig_folder = os.path.join(figure_folder, 'Feature_boxplot')
    os.makedirs(subfig_folder, exist_ok=True)
    
    figure_prefix = os.path.join(subfig_folder, prefix)
    csv_prefix = os.path.join(csv_folder, f'{prefix}')
    plot_feature_relative_rate_boxplot(
        figure_prefix, csv_prefix)
    
    # sine wave average
    subfig_folder = os.path.join(figure_folder, 'SineWave_Rate')
    os.makedirs(subfig_folder, exist_ok=True)
    figure_prefix = os.path.join(subfig_folder, prefix)

    phase_fn = os.path.join(csv_folder, f'{prefix}.gene.1kb_average_rate.raw.csv')
    fit_pkl_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageRate.SineWave.pkl') 
    plot_sinewave_rate(
        figure_prefix,
        phase_fn, fit_pkl_fn,
        phasing_xlabel, phasing_ylabel, phasing_ylim_sinewave)

    # sine wave quantiles
    if not is_user:
        subfig_folder = os.path.join(figure_folder, 'SineWave_Rate_Quantile')
        os.makedirs(subfig_folder, exist_ok=True)
        figure_prefix = os.path.join(subfig_folder, prefix)

        phase_fn = os.path.join(csv_folder, f'{prefix}.Gene.AverageRate_1kb.Quantile.csv')
        fit_pkl_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageRate.Quantile.SineWave.pkl')

        if not os.path.isfile(fit_pkl_fn):
            logger.error('Cannot find sine wave analysis for quantiles. Try re-run phase command.') 
            raise RuntimeError
        plot_sinewave_rate_quantile(
            figure_prefix,
            phase_fn,
            fit_pkl_fn,
            phasing_xlabel, phasing_ylabel, phasing_ylim_sinewave)
    else:
        subfig_folder = os.path.join(figure_folder, 'SineWave_Rate_UserQuantile')
        os.makedirs(subfig_folder, exist_ok=True)
        figure_prefix = os.path.join(subfig_folder, prefix)

        phase_fn = os.path.join(csv_folder, f'{prefix}.Gene.AverageRate_1kb.UserQuantile.csv')
        fit_pkl_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageRate.UserQuantile.SineWave.pkl')

        if not os.path.isfile(fit_pkl_fn):
            logger.error('Cannot find sine wave analysis for user quantiles. Try re-run phase command.') 
            raise RuntimeError
        plot_sinewave_rate_quantile(
            figure_prefix,
            phase_fn,
            fit_pkl_fn,
            phasing_xlabel, phasing_ylabel, phasing_ylim_sinewave, is_user=True)

    # sine wave fraction
    subfig_folder = os.path.join(figure_folder, 'SineWave_MethylatedFraction')
    os.makedirs(subfig_folder, exist_ok=True)
    figure_prefix = os.path.join(subfig_folder, prefix)

    phase_fn = os.path.join(csv_folder, f'{prefix}.gene.1kb_average_fraction.raw.csv')
    fit_pkl_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageFraction.SineWave.pkl') 
    plot_sinewave_fraction(
        figure_prefix,
        phase_fn, fit_pkl_fn,
        phasing_xlabel, timecourse_frac_ylabel, [0, 100])

   # sine wave quantiles
    if not is_user:
        subfig_folder = os.path.join(figure_folder, 'SineWave_Fraction_Quantile')
        os.makedirs(subfig_folder, exist_ok=True)
        figure_prefix = os.path.join(subfig_folder, prefix)

        phase_fn = os.path.join(csv_folder, f'{prefix}.Gene.AverageFraction_1kb.Quantile.csv')
        fit_pkl_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageFraction.Quantile.SineWave.pkl')

        if not os.path.isfile(fit_pkl_fn):
            logger.error('Cannot find sine wave analysis for quantiles. Try re-run phase command.')
            raise RuntimeError
        plot_sinewave_fraction_qiuantile(
            figure_prefix,
            phase_fn,
            fit_pkl_fn,
            phasing_xlabel, timecourse_frac_ylabel, [0, 100])
    else:
        subfig_folder = os.path.join(figure_folder, 'SineWave_Fraction_UserQuantile')
        os.makedirs(subfig_folder, exist_ok=True)
        figure_prefix = os.path.join(subfig_folder, prefix)

        phase_fn = os.path.join(csv_folder, f'{prefix}.Gene.AverageFraction_1kb.UserQuantile.csv')
        fit_pkl_fn = os.path.join(fit_folder, f'{prefix}.Gene.AverageFraction.UserQuantile.SineWave.pkl')

        if not os.path.isfile(fit_pkl_fn):
            logger.error('Cannot find sine wave analysis for user quantiles. Try re-run phase command.') 
            raise RuntimeError
        plot_sinewave_fraction_qiuantile(
            figure_prefix,
            phase_fn,
            fit_pkl_fn,
            phasing_xlabel, timecourse_frac_ylabel, [0, 100], is_user=True)
    # tRNA figures
    subfig_folder = os.path.join(figure_folder, 'tRNA')
    os.makedirs(subfig_folder, exist_ok=True)
    # time course
    figure_prefix = os.path.join(subfig_folder, prefix)    
    trna_fn = os.path.join(csv_folder, f'{prefix}.tRNA_element.timecourse.percentile.csv')
    plot_trna_timecourse(
        figure_prefix,
        trna_fn,
        timecourse_xlabel,
        timecourse_frac_ylabel,
        timecourse_xlim,
        sample_pd)
    # rate
    rate_fn = os.path.join(csv_folder, f'{prefix}.tRNA_element.rate.median.csv')
    plot_trna_rate_regression(
        figure_prefix,
        sample_pd,
        rate_fn,
        timecourse_xlabel,
        timecourse_xlim)

    # tRNA map
    rate_fn = os.path.join(csv_folder, f'{prefix}.tRNA_single_exon.mean_std.csv')
    plot_trna_rate_visual(figure_prefix, rate_fn)
    # tRNA figures
    subfig_folder = os.path.join(figure_folder, 'other')
    os.makedirs(subfig_folder, exist_ok=True)
    # other figures
    figure_prefix = os.path.join(subfig_folder, prefix) 
    gene_adj_fn = os.path.join(csv_folder, f'{prefix}.Gene.Adjusted_Rate.csv') 
    plot_gene_rate_group_by_length_rnaseq(
        figure_prefix,
        gene_adj_fn,
        rnaseq_fn)