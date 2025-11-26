"""
Plotting functions for 2x2 regime analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class RegimePlotter:
    """Plotting utilities for regime analysis."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize plotter.
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_scatter(
        self,
        data: pd.DataFrame,
        regime_def,
        save: bool = True
    ):
        """
        Create scatter plot: Growth vs Inflation, colored by regime.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with regime assignments
        regime_def : RegimeDefinitions
            Regime definitions object
        save : bool
            Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot each regime with different colors
        for regime_id in range(4):
            regime_mask = data['regime'] == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            color = regime_def.get_regime_color(regime_id)
            name = regime_def.get_regime_short_name(regime_id)
            
            ax.scatter(
                regime_data['growth_factor'],
                regime_data['inflation_factor'],
                c=color,
                label=name,
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidths=0.5
            )
        
        # Add threshold lines
        growth_thresh = regime_def.growth_threshold
        infl_thresh = regime_def.inflation_threshold
        
        ax.axhline(
            infl_thresh,
            color='black',
            linestyle='--',
            linewidth=1.5,
            alpha=0.5,
            label='Inflation Threshold'
        )
        ax.axvline(
            growth_thresh,
            color='black',
            linestyle='--',
            linewidth=1.5,
            alpha=0.5,
            label='Growth Threshold'
        )
        
        # Add quadrant labels
        ax.text(0.02, 0.02, 'Goldilocks\n(High G / Low I)',
               transform=ax.transAxes, fontsize=10, ha='left', va='bottom',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax.text(0.98, 0.02, 'Overheating\n(High G / High I)',
               transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        ax.text(0.02, 0.98, 'Stagflation\n(Low G / High I)',
               transform=ax.transAxes, fontsize=10, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.3))
        ax.text(0.98, 0.98, 'Slowdown\n(Low G / Low I)',
               transform=ax.transAxes, fontsize=10, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        ax.set_xlabel('Growth Factor', fontsize=12, fontweight='bold')
        ax.set_ylabel('Inflation Factor', fontsize=12, fontweight='bold')
        ax.set_title('2x2 Regime Classification: Growth vs Inflation', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'regime_scatter_plot.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    Saved scatter plot to {output_file.name}")
        
        plt.close()
    
    def plot_boxplots(
        self,
        data: pd.DataFrame,
        regime_def,
        save: bool = True
    ):
        """
        Create boxplots: ERP by regime.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with regime assignments and ERP
        regime_def : RegimeDefinitions
            Regime definitions object
        save : bool
            Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data for boxplot
        plot_data = []
        for regime_id in range(4):
            regime_mask = data['regime'] == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            for erp_val in regime_data['erp'].values:
                plot_data.append({
                    'regime': regime_def.get_regime_short_name(regime_id),
                    'regime_id': regime_id,
                    'erp': erp_val
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Boxplot 1: ERP by regime
        regime_order = [regime_def.get_regime_short_name(i) for i in range(4) 
                       if i in plot_df['regime_id'].values]
        colors = [regime_def.get_regime_color(i) for i in range(4) 
                 if i in plot_df['regime_id'].values]
        
        bp1 = axes[0].boxplot(
            [plot_df[plot_df['regime'] == name]['erp'].values 
             for name in regime_order],
            labels=regime_order,
            patch_artist=True,
            showmeans=True,
            meanline=True
        )
        
        # Color the boxes
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        axes[0].set_ylabel('ERP (Equity Risk Premium)', fontsize=11, fontweight='bold')
        axes[0].set_title('ERP Distribution by Regime', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Boxplot 2: ERP Volatility by regime
        vol_data = []
        for regime_id in range(4):
            regime_mask = data['regime'] == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            for vol_val in regime_data['erp_volatility'].dropna().values:
                vol_data.append({
                    'regime': regime_def.get_regime_short_name(regime_id),
                    'regime_id': regime_id,
                    'erp_volatility': vol_val
                })
        
        vol_df = pd.DataFrame(vol_data)
        
        if len(vol_df) > 0:
            bp2 = axes[1].boxplot(
                [vol_df[vol_df['regime'] == name]['erp_volatility'].values 
                 for name in regime_order],
                labels=regime_order,
                patch_artist=True,
                showmeans=True,
                meanline=True
            )
            
            # Color the boxes
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[1].set_ylabel('ERP Volatility (12-month rolling)', 
                              fontsize=11, fontweight='bold')
            axes[1].set_title('ERP Volatility Distribution by Regime', 
                            fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].tick_params(axis='x', rotation=45)
        else:
            axes[1].text(0.5, 0.5, 'Insufficient volatility data',
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('ERP Volatility Distribution by Regime', 
                            fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'erp_boxplots_by_regime.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    Saved boxplots to {output_file.name}")
        
        plt.close()
    
    def plot_time_series(
        self,
        data: pd.DataFrame,
        regime_def,
        save: bool = True
    ):
        """
        Create time series plot showing regime assignments over time.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with regime assignments
        regime_def : RegimeDefinitions
            Regime definitions object
        save : bool
            Whether to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        dates = pd.to_datetime(data['date'])
        
        # Plot 1: Regime assignments
        for regime_id in range(4):
            regime_mask = data['regime'] == regime_id
            if regime_mask.sum() == 0:
                continue
            
            color = regime_def.get_regime_color(regime_id)
            name = regime_def.get_regime_short_name(regime_id)
            
            axes[0].scatter(
                dates[regime_mask],
                data.loc[regime_mask, 'regime'],
                c=color,
                label=name,
                alpha=0.7,
                s=30
            )
        
        axes[0].set_ylabel('Regime', fontsize=11, fontweight='bold')
        axes[0].set_title('Regime Assignments Over Time', 
                         fontsize=12, fontweight='bold')
        axes[0].set_yticks(range(4))
        axes[0].set_yticklabels([regime_def.get_regime_short_name(i) 
                                for i in range(4)])
        axes[0].legend(loc='upper left', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Growth and Inflation factors
        axes[1].plot(dates, data['growth_factor'], 
                    label='Growth Factor', linewidth=1.5, alpha=0.8, color='#2ca02c')
        axes[1].axhline(regime_def.growth_threshold, 
                       color='green', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].set_ylabel('Growth Factor', fontsize=11, fontweight='bold', color='#2ca02c')
        axes[1].tick_params(axis='y', labelcolor='#2ca02c')
        axes[1].grid(True, alpha=0.3)
        
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(dates, data['inflation_factor'], 
                     label='Inflation Factor', linewidth=1.5, alpha=0.8, color='#d62728')
        ax1_twin.axhline(regime_def.inflation_threshold, 
                        color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1_twin.set_ylabel('Inflation Factor', fontsize=11, fontweight='bold', color='#d62728')
        ax1_twin.tick_params(axis='y', labelcolor='#d62728')
        
        axes[1].set_title('Growth and Inflation Factors Over Time', 
                         fontsize=12, fontweight='bold')
        
        # Plot 3: ERP
        axes[2].plot(dates, data['erp'], 
                    label='ERP', linewidth=1.5, alpha=0.8, color='#1f77b4')
        axes[2].axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        axes[2].set_ylabel('ERP (Equity Risk Premium)', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Date', fontsize=11, fontweight='bold')
        axes[2].set_title('Equity Risk Premium Over Time', 
                         fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'regime_time_series.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    Saved time series plot to {output_file.name}")
        
        plt.close()

