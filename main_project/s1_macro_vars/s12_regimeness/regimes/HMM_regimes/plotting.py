"""
Plotting functions for HMM regime analysis.

Creates visualizations for:
- Regime probabilities over time (stacked area/colored bands)
- Transition matrix heatmap
- Comparison chart: HMM regimes vs simple 4 quadrants (cross-tab)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class HMMPlotter:
    """Plotting utilities for HMM regime analysis."""
    
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
    
    def plot_regime_probabilities(
        self,
        dates: pd.Series,
        regime_probs: np.ndarray,
        regime_names: Dict[int, str],
        save: bool = True
    ):
        """
        Plot regime probabilities over time (stacked area chart).
        
        Parameters:
        -----------
        dates : pd.Series
            Date series
        regime_probs : np.ndarray
            Regime probabilities (n_samples, n_regimes)
        regime_names : Dict[int, str]
            Mapping of regime_id to name
        save : bool
            Whether to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        dates = pd.to_datetime(dates)
        n_regimes = regime_probs.shape[1]
        
        # Color scheme
        colors = plt.cm.Set3(np.linspace(0, 1, n_regimes))
        
        # Plot 1: Stacked area chart
        axes[0].stackplot(
            dates,
            *[regime_probs[:, i] for i in range(n_regimes)],
            labels=[regime_names.get(i, f'Regime {i}') for i in range(n_regimes)],
            colors=colors,
            alpha=0.7
        )
        axes[0].set_ylabel('Probability', fontsize=12, fontweight='bold')
        axes[0].set_title('Regime Probabilities Over Time (Stacked Area)', 
                         fontsize=14, fontweight='bold', pad=20)
        axes[0].legend(loc='upper left', fontsize=9, ncol=min(n_regimes, 4))
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Plot 2: Individual probability lines
        for i in range(n_regimes):
            axes[1].plot(
                dates,
                regime_probs[:, i],
                label=regime_names.get(i, f'Regime {i}'),
                color=colors[i],
                linewidth=2,
                alpha=0.8
            )
        axes[1].set_ylabel('Probability', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Date', fontsize=12, fontweight='bold')
        axes[1].set_title('Regime Probabilities Over Time (Individual Lines)', 
                         fontsize=14, fontweight='bold', pad=20)
        axes[1].legend(loc='upper left', fontsize=9, ncol=min(n_regimes, 4))
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'regime_probabilities_time_series.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    Saved regime probabilities plot to {output_file.name}")
        
        plt.close()
    
    def plot_transition_matrix(
        self,
        transition_matrix: np.ndarray,
        regime_names: Dict[int, str],
        save: bool = True
    ):
        """
        Plot transition matrix as heatmap.
        
        Parameters:
        -----------
        transition_matrix : np.ndarray
            Transition matrix (n_regimes, n_regimes)
        regime_names : Dict[int, str]
            Mapping of regime_id to name
        save : bool
            Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        n_regimes = transition_matrix.shape[0]
        labels = [regime_names.get(i, f'R{i}') for i in range(n_regimes)]
        
        # Create heatmap
        im = ax.imshow(transition_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(n_regimes))
        ax.set_yticks(range(n_regimes))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        
        # Add text annotations
        for i in range(n_regimes):
            for j in range(n_regimes):
                text = ax.text(
                    j, i, f'{transition_matrix[i, j]:.3f}',
                    ha="center", va="center",
                    color="white" if transition_matrix[i, j] > 0.5 else "black",
                    fontsize=11, fontweight='bold'
                )
        
        ax.set_xlabel('To Regime', fontsize=12, fontweight='bold')
        ax.set_ylabel('From Regime', fontsize=12, fontweight='bold')
        ax.set_title('Regime Transition Matrix\n(Probability of switching from row regime to column regime)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add interpretation
        ax.text(0.02, 0.02, 'Interpretation: Diagonal = persistence (staying in same regime).\nOff-diagonal = transition probability. Higher values = more persistent regimes.', 
                transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Transition Probability', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'transition_matrix_heatmap.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    Saved transition matrix heatmap to {output_file.name}")
        
        plt.close()
    
    def plot_regime_comparison(
        self,
        hmm_regimes: np.ndarray,
        quadrant_regimes: np.ndarray,
        regime_names_hmm: Dict[int, str],
        regime_names_quad: Dict[int, str],
        save: bool = True
    ):
        """
        Create cross-tabulation comparison: HMM regimes vs simple 4 quadrants.
        
        Parameters:
        -----------
        hmm_regimes : np.ndarray
            HMM regime assignments
        quadrant_regimes : np.ndarray
            Simple 2x2 quadrant regime assignments
        regime_names_hmm : Dict[int, str]
            HMM regime names
        regime_names_quad : Dict[int, str]
            Quadrant regime names
        save : bool
            Whether to save the plot
        """
        # Create cross-tabulation
        crosstab = pd.crosstab(
            pd.Series(hmm_regimes, name='HMM Regime'),
            pd.Series(quadrant_regimes, name='Quadrant Regime'),
            normalize='index'  # Normalize by row (HMM regime)
        )
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Heatmap of cross-tabulation
        sns.heatmap(
            crosstab,
            annot=True,
            fmt='.2%',
            cmap='YlOrRd',
            cbar_kws={'label': 'Proportion'},
            ax=axes[0],
            linewidths=0.5,
            linecolor='gray'
        )
        axes[0].set_title('HMM Regimes vs 2x2 Quadrants\n(Normalized by HMM Regime)', 
                         fontsize=12, fontweight='bold', pad=15)
        axes[0].set_xlabel('Quadrant Regime', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('HMM Regime', fontsize=11, fontweight='bold')
        
        # Update labels
        hmm_labels = [regime_names_hmm.get(i, f'R{i}') for i in crosstab.index]
        quad_labels = [regime_names_quad.get(i, f'Q{i}') for i in crosstab.columns]
        axes[0].set_yticklabels(hmm_labels, rotation=0, fontsize=9)
        axes[0].set_xticklabels(quad_labels, rotation=45, ha='right', fontsize=9)
        
        # Plot 2: Bar chart showing agreement
        agreement = []
        for hmm_regime in crosstab.index:
            # Find the quadrant regime with highest proportion
            max_quad = crosstab.loc[hmm_regime].idxmax()
            max_prop = crosstab.loc[hmm_regime, max_quad]
            agreement.append({
                'HMM_Regime': regime_names_hmm.get(hmm_regime, f'R{hmm_regime}'),
                'Most_Common_Quadrant': regime_names_quad.get(max_quad, f'Q{max_quad}'),
                'Agreement': max_prop
            })
        
        agreement_df = pd.DataFrame(agreement)
        
        bars = axes[1].bar(
            range(len(agreement_df)),
            agreement_df['Agreement'],
            color=plt.cm.viridis(np.linspace(0, 1, len(agreement_df))),
            alpha=0.7,
            edgecolor='black',
            linewidth=1.5
        )
        
        axes[1].set_xticks(range(len(agreement_df)))
        axes[1].set_xticklabels(agreement_df['HMM_Regime'], rotation=45, ha='right', fontsize=9)
        axes[1].set_ylabel('Proportion in Most Common Quadrant', fontsize=11, fontweight='bold')
        axes[1].set_title('HMM-Quadrant Agreement', fontsize=12, fontweight='bold', pad=15)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim([0, 1])
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, agreement_df['Agreement'])):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{val:.1%}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'hmm_vs_quadrants_comparison.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    Saved HMM vs quadrants comparison to {output_file.name}")
        
        plt.close()
        
        # Also save cross-tabulation as CSV
        crosstab_file = self.output_dir / 'hmm_quadrant_crosstab.csv'
        crosstab.to_csv(crosstab_file)
        print(f"    Saved cross-tabulation to {crosstab_file.name}")
        
        return crosstab
    
    def plot_regime_time_series(
        self,
        dates: pd.Series,
        regime_states: np.ndarray,
        regime_names: Dict[int, str],
        save: bool = True
    ):
        """
        Plot regime assignments over time.
        
        Parameters:
        -----------
        dates : pd.Series
            Date series
        regime_states : np.ndarray
            Regime assignments
        regime_names : Dict[int, str]
            Mapping of regime_id to name
        save : bool
            Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(16, 6))
        
        dates = pd.to_datetime(dates)
        n_regimes = len(np.unique(regime_states))
        colors = plt.cm.Set3(np.linspace(0, 1, n_regimes))
        
        # Plot each regime with different colors
        for regime_id in range(n_regimes):
            mask = regime_states == regime_id
            if mask.sum() == 0:
                continue
            
            ax.scatter(
                dates[mask],
                regime_states[mask],
                c=[colors[regime_id]],
                label=regime_names.get(regime_id, f'Regime {regime_id}'),
                alpha=0.7,
                s=30,
                edgecolors='white',
                linewidths=0.5
            )
        
        ax.set_ylabel('Regime', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_title('HMM Regime Assignments Over Time', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks(range(n_regimes))
        ax.set_yticklabels([regime_names.get(i, f'R{i}') for i in range(n_regimes)], fontsize=9)
        ax.legend(loc='upper left', fontsize=9, ncol=min(n_regimes, 4))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'regime_assignments_time_series.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    Saved regime assignments time series to {output_file.name}")
        
        plt.close()
    
    def plot_regime_interpretation(
        self,
        data: pd.DataFrame,
        regime_states: np.ndarray,
        regime_characteristics: Dict[int, Dict],
        save: bool = True
    ):
        """
        Plot regime interpretation showing how regimes are labeled based on macro averages.
        
        Creates plots showing:
        1. Growth vs Inflation scatter with regime assignments
        2. Macro variable distributions by regime
        3. Comparison of regime averages vs overall medians
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        dates = pd.to_datetime(data['date'])
        n_regimes = len(regime_characteristics)
        colors = plt.cm.Set3(np.linspace(0, 1, n_regimes))
        
        # Plot 1: Growth vs Inflation scatter with regimes
        ax1 = fig.add_subplot(gs[0, 0])
        for regime_id, chars in regime_characteristics.items():
            regime_mask = regime_states == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            ax1.scatter(
                regime_data['growth_factor'],
                regime_data['inflation_factor'],
                c=[colors[regime_id]],
                label=f"R{regime_id}: {chars['name']}",
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidths=0.5
            )
        
        # Add median lines
        growth_median = data['growth_factor'].median()
        infl_median = data['inflation_factor'].median()
        ax1.axhline(infl_median, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Median')
        ax1.axvline(growth_median, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Add regime centroids
        for regime_id, chars in regime_characteristics.items():
            ax1.scatter(
                chars['avg_growth'],
                chars['avg_inflation'],
                c='red',
                marker='*',
                s=300,
                edgecolors='black',
                linewidths=2,
                zorder=10
            )
            ax1.annotate(
                f"R{regime_id}",
                (chars['avg_growth'], chars['avg_inflation']),
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center',
                color='white'
            )
        
        ax1.set_xlabel('Growth Factor', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Inflation Factor', fontsize=11, fontweight='bold')
        ax1.set_title('Regime Interpretation: Growth vs Inflation\n(Points = observations, Stars = regime centroids, Dashed lines = medians)', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8, ncol=1)
        ax1.grid(True, alpha=0.3)
        
        # Add interpretation text
        interpretation_text = (
            "Interpretation Method:\n"
            "1. Calculate average macro values for each regime\n"
            "2. Compare to overall median (dashed lines)\n"
            "3. Label: High if avg ≥ median, Low otherwise"
        )
        ax1.text(0.02, 0.98, interpretation_text, transform=ax1.transAxes,
                fontsize=8, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Plot 2: Macro variable averages by regime (bar chart)
        ax2 = fig.add_subplot(gs[0, 1])
        regime_ids = sorted(regime_characteristics.keys())
        x_pos = np.arange(len(regime_ids))
        width = 0.2
        
        growth_avgs = [regime_characteristics[r]['avg_growth'] for r in regime_ids]
        infl_avgs = [regime_characteristics[r]['avg_inflation'] for r in regime_ids]
        policy_avgs = [regime_characteristics[r]['avg_policy'] for r in regime_ids]
        vol_avgs = [regime_characteristics[r]['avg_volatility'] for r in regime_ids]
        
        # Normalize policy and vol for visualization (use z-scores)
        policy_mean = np.mean(policy_avgs)
        policy_std = np.std(policy_avgs)
        vol_mean = np.mean(vol_avgs)
        vol_std = np.std(vol_avgs)
        policy_norm = [(p - policy_mean) / policy_std if policy_std > 0 else 0 for p in policy_avgs]
        vol_norm = [(v - vol_mean) / vol_std if vol_std > 0 else 0 for v in vol_avgs]
        
        ax2.bar(x_pos - 1.5*width, growth_avgs, width, label='Growth', alpha=0.8, color='#2ca02c')
        ax2.bar(x_pos - 0.5*width, infl_avgs, width, label='Inflation', alpha=0.8, color='#d62728')
        ax2.bar(x_pos + 0.5*width, policy_norm, width, label='Policy (norm)', alpha=0.8, color='#9467bd')
        ax2.bar(x_pos + 1.5*width, vol_norm, width, label='Volatility (norm)', alpha=0.8, color='#8c564b')
        
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.set_xlabel('Regime', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Average Value (normalized)', fontsize=11, fontweight='bold')
        ax2.set_title('Average Macro Variables by Regime', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"R{r}" for r in regime_ids])
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Growth vs Inflation with quadrant labels
        ax3 = fig.add_subplot(gs[0, 2])
        for regime_id, chars in regime_characteristics.items():
            ax3.scatter(
                chars['avg_growth'],
                chars['avg_inflation'],
                c=[colors[regime_id]],
                s=300,
                alpha=0.7,
                edgecolors='black',
                linewidths=2,
                label=f"R{regime_id}: {chars['name']}"
            )
            ax3.annotate(
                f"R{regime_id}",
                (chars['avg_growth'], chars['avg_inflation']),
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                color='white'
            )
        
        # Add quadrant lines
        ax3.axhline(infl_median, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.axvline(growth_median, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Add quadrant labels
        ax3.text(0.02, 0.02, 'Low G\nLow I', transform=ax3.transAxes,
                fontsize=9, ha='left', va='bottom', alpha=0.5,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax3.text(0.98, 0.02, 'High G\nLow I', transform=ax3.transAxes,
                fontsize=9, ha='right', va='bottom', alpha=0.5,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax3.text(0.02, 0.98, 'Low G\nHigh I', transform=ax3.transAxes,
                fontsize=9, ha='left', va='top', alpha=0.5,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        ax3.text(0.98, 0.98, 'High G\nHigh I', transform=ax3.transAxes,
                fontsize=9, ha='right', va='top', alpha=0.5,
                bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.3))
        
        ax3.set_xlabel('Average Growth Factor', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Average Inflation Factor', fontsize=11, fontweight='bold')
        ax3.set_title('Regime Centroids in Growth-Inflation Space', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4-7: Distribution of each macro variable by regime
        macro_vars = [
            ('growth_factor', 'Growth Factor', 1, 0),
            ('inflation_factor', 'Inflation Factor', 1, 1),
            ('monetary_policy_factor', 'Monetary Policy Factor', 1, 2),
            ('market_volatility_factor', 'Market Volatility Factor', 2, 0)
        ]
        
        for i, (var_col, var_name, row, col) in enumerate(macro_vars):
            ax = fig.add_subplot(gs[row, col])
            
            # Create violin plots or box plots
            plot_data = []
            labels = []
            for regime_id in sorted(regime_characteristics.keys()):
                regime_mask = regime_states == regime_id
                regime_data = data.loc[regime_mask, var_col].dropna()
                if len(regime_data) > 0:
                    plot_data.append(regime_data.values)
                    labels.append(f"R{regime_id}")
            
            if plot_data:
                bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showmeans=True)
                for patch, regime_id in zip(bp['boxes'], sorted(regime_characteristics.keys())):
                    patch.set_facecolor(colors[regime_id])
                    patch.set_alpha(0.7)
                
                # Add median line for overall data
                overall_median = data[var_col].median()
                ax.axhline(overall_median, color='red', linestyle='--', linewidth=2, 
                          alpha=0.7, label=f'Overall Median: {overall_median:.3f}')
            
            ax.set_ylabel(var_name, fontsize=10, fontweight='bold')
            ax.set_title(f'{var_name} Distribution by Regime', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 8: Regime interpretation table (text summary)
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        
        # Create text summary
        summary_text = "Regime Interpretation Method:\n\n"
        summary_text += "1. For each regime, calculate average macro values\n"
        summary_text += "2. Compare averages to overall median:\n"
        summary_text += "   - Growth: High if avg >= median, Low otherwise\n"
        summary_text += "   - Inflation: High if avg >= median, Low otherwise\n"
        summary_text += "3. Label regime as: '{Growth Level} Growth / {Inflation Level} Inflation'\n\n"
        summary_text += "Regime Details:\n"
        summary_text += "=" * 80 + "\n"
        
        for regime_id in sorted(regime_characteristics.keys()):
            chars = regime_characteristics[regime_id]
            summary_text += f"\nRegime {regime_id}: {chars['name']}\n"
            summary_text += f"  Avg Growth: {chars['avg_growth']:.3f} (vs median: {data['growth_factor'].median():.3f}) → {chars['growth_level']}\n"
            summary_text += f"  Avg Inflation: {chars['avg_inflation']:.3f} (vs median: {data['inflation_factor'].median():.3f}) → {chars['inflation_level']}\n"
            summary_text += f"  Avg Policy: {chars['avg_policy']:.3f} (vs median: {data['monetary_policy_factor'].median():.3f}) → {chars['policy_level']}\n"
            summary_text += f"  Avg Volatility: {chars['avg_volatility']:.3f} (vs median: {data['market_volatility_factor'].median():.3f}) → {chars['volatility_level']}\n"
            summary_text += f"  Observations: {chars['n_observations']} ({chars['pct_of_total']:.1f}%)\n"
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        if save:
            output_file = self.output_dir / 'regime_interpretation_plots.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    Saved regime interpretation plots to {output_file.name}")
        
        plt.close()

