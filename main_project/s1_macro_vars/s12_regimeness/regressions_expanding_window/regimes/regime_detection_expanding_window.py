"""
Expanding Window Regime Detection (No Look-Ahead Bias)

This module implements regime detection using expanding windows to avoid look-ahead bias.
For each time t, regimes are detected using only data from start to t (expanding window).

Two approaches:
1. HMM Optimal (Growth + Policy) - Expanding window HMM
2. 2x2 Regime (Growth + Inflation) - Expanding window thresholds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import io
from contextlib import contextmanager

# Suppress convergence warnings from hmmlearn
warnings.filterwarnings('ignore', category=UserWarning, module='hmmlearn')
warnings.filterwarnings('ignore', message='.*convergence.*')
warnings.filterwarnings('ignore', message='.*not converging.*')


@contextmanager
def suppress_convergence_warnings():
    """Context manager to suppress hmmlearn convergence warnings."""
    # Capture stderr and filter out convergence messages
    old_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        stderr_output = sys.stderr.getvalue()
        sys.stderr = old_stderr
        # Only print if there are non-convergence errors
        if stderr_output and 'convergence' not in stderr_output.lower() and 'not converging' not in stderr_output.lower():
            print(stderr_output, file=old_stderr, end='')

# Import shared helpers and regime modules
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
SECTION_DIR = SCRIPT_DIR.parents[3]  # s1_macro_vars
if str(SECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SECTION_DIR))

from path_utils import get_project_root

sys.path.insert(0, str(SCRIPT_DIR.parent.parent / 'regimes' / 'HMM_regimes'))
from hmm_model import HMMRegimeModel

sys.path.insert(0, str(SCRIPT_DIR.parent.parent / 'regimes' / '2x2_regimes'))
from regime_definitions import RegimeDefinitions


class ExpandingWindowRegimeDetector:
    """
    Detects regimes using expanding windows to avoid look-ahead bias.
    
    For each time t:
    - Uses data from start to t (expanding window)
    - Fits model on this window
    - Gets regime assignment for time t
    - Only uses information available at time t
    """
    
    def __init__(
        self,
        data_dir: Path,
        regime_model: str = 'hmm_optimal',  # 'hmm_optimal' or '2x2'
        min_window_size: int = 24,  # Minimum months before starting regime detection
        output_dir: Optional[Path] = None
    ):
        """
        Initialize expanding window regime detector.
        
        Parameters:
        -----------
        data_dir : Path
            Path to main_project directory
        regime_model : str
            Which regime model to use: 'hmm_optimal' or '2x2'
        min_window_size : int
            Minimum number of months before starting regime detection
        output_dir : Path, optional
            Output directory for regime assignments
        """
        self.data_dir = Path(data_dir)
        self.regime_model = regime_model
        self.min_window_size = min_window_size
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 's1_macro_vars' / 's12_regimeness' / 'regressions_expanding_window' / 'results' / 'regime_assignments' / regime_model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.macro_data = None
        self.combined_data = None
        self.regime_assignments = None
        
    def load_data(self):
        """Load macro data and prepare for regime detection."""
        print("Loading data for expanding window regime detection...")
        
        # Load macro factors
        macro_file = self.data_dir / 'data' / 'macro_final' / 'final_macro.csv'
        if not macro_file.exists():
            raise FileNotFoundError(f"Macro file not found: {macro_file}")
        
        df = pd.read_csv(macro_file, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.set_index('date')
        
        self.macro_data = df
        print(f"  Loaded {len(df)} observations")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def detect_hmm_regimes_expanding(self) -> pd.DataFrame:
        """
        Detect HMM regimes using expanding windows.
        
        For each time t:
        1. Use data from start to t
        2. Fit HMM on this expanding window
        3. Get regime assignment for time t
        """
        print("\n" + "="*80)
        print("EXPANDING WINDOW HMM REGIME DETECTION (Growth + Policy)")
        print("="*80)
        
        if self.macro_data is None:
            self.load_data()
        
        # Prepare features (Growth + Policy)
        variables = ['growth_factor', 'monetary_policy_factor']
        data = self.macro_data[variables].dropna()
        
        if len(data) < self.min_window_size:
            raise ValueError(f"Not enough data. Need at least {self.min_window_size} observations.")
        
        print(f"\nUsing variables: {variables}")
        print(f"Total observations: {len(data)}")
        print(f"Minimum window size: {self.min_window_size}")
        print(f"Will detect regimes for {len(data) - self.min_window_size} periods")
        
        # Storage for regime assignments
        regime_assignments = []
        regime_probs_list = []
        dates_list = []
        
        # Determine optimal K using full sample (for consistency, but we'll refit each window)
        # Actually, let's use K=4 as determined from full sample analysis
        n_regimes = 4
        
        print(f"\nUsing K = {n_regimes} regimes")
        print("\nDetecting regimes with expanding windows...")
        
        # For each time point, fit HMM on expanding window
        for t_idx in range(self.min_window_size, len(data)):
            # Get expanding window: start to current time
            window_data = data.iloc[:t_idx+1]  # Include current time point
            
            # Standardize features
            scaler = StandardScaler()
            features = scaler.fit_transform(window_data.values)
            
            # Fit HMM on expanding window
            hmm_model = HMMRegimeModel(n_regimes=n_regimes)
            hmm_model.scaler = scaler
            
            try:
                # Suppress convergence warnings during fitting
                with warnings.catch_warnings(), suppress_convergence_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='hmmlearn')
                    warnings.filterwarnings('ignore', message='.*convergence.*')
                    warnings.filterwarnings('ignore', message='.*not converging.*')
                    hmm_model.fit(features, n_init=5)  # Fewer iterations for speed
                
                # Get regime assignment for current time point (last in window)
                # Use the last state from the full sequence prediction
                all_states = hmm_model.model.predict(features)
                regime_state = all_states[-1]
                
                # Get probabilities for last observation
                log_probs = hmm_model.model.score_samples(features[-1:].reshape(1, -1))[1]
                log_probs_stable = log_probs - np.max(log_probs, axis=1, keepdims=True)
                regime_probs = np.exp(log_probs_stable)
                regime_probs = regime_probs / np.sum(regime_probs, axis=1, keepdims=True)
                regime_probs = regime_probs[0]  # Get first (and only) row
                
                # Store results
                dates_list.append(window_data.index[-1])
                regime_assignments.append(regime_state)
                regime_probs_list.append(regime_probs)
                
                if (t_idx - self.min_window_size + 1) % 50 == 0:
                    print(f"  Processed {t_idx - self.min_window_size + 1} / {len(data) - self.min_window_size} periods")
                    
            except Exception as e:
                print(f"  Warning: Failed at t={window_data.index[-1]}: {e}")
                # Use previous regime if available, otherwise use 0
                if len(regime_assignments) > 0:
                    regime_assignments.append(regime_assignments[-1])
                    regime_probs_list.append(regime_probs_list[-1])
                else:
                    regime_assignments.append(0)
                    regime_probs_list.append(np.array([0.25] * n_regimes))
                dates_list.append(window_data.index[-1])
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'date': dates_list,
            'regime': regime_assignments
        })
        
        # Add probability columns
        for i in range(n_regimes):
            results_df[f'prob_R{i}'] = [probs[i] for probs in regime_probs_list]
        
        # Add regime names (will be interpreted later)
        results_df['regime_name'] = results_df['regime'].apply(lambda x: f"Regime {x}")
        
        results_df = results_df.set_index('date')
        self.regime_assignments = results_df
        
        print(f"\nCompleted expanding window regime detection")
        print(f"  Detected regimes for {len(results_df)} periods")
        print(f"  Date range: {results_df.index.min()} to {results_df.index.max()}")
        
        return results_df
    
    def detect_2x2_regimes_expanding(self, use_mahalanobis: bool = True) -> pd.DataFrame:
        """
        Detect 2x2 regimes using expanding windows.
        
        Two modes:
        1. Hard threshold (use_mahalanobis=False): Simple above/below median classification
        2. Mahalanobis distance (use_mahalanobis=True): Uses cluster statistics and probabilities
        
        For each time t:
        1. Use data from start to t
        2. Calculate thresholds (median) on this expanding window
        3. Classify current observation into 4 regimes based on thresholds
        4. (If Mahalanobis): Compute cluster statistics and probabilities
        """
        method_name = "Mahalanobis Distance" if use_mahalanobis else "Hard Thresholds"
        print("\n" + "="*80)
        print(f"EXPANDING WINDOW 2x2 REGIME DETECTION ({method_name})")
        print("="*80)
        
        if self.macro_data is None:
            self.load_data()
        
        # Prepare data
        variables = ['growth_factor', 'inflation_factor']
        data = self.macro_data[variables].dropna()
        
        if len(data) < self.min_window_size:
            raise ValueError(f"Not enough data. Need at least {self.min_window_size} observations.")
        
        print(f"\nUsing variables: {variables}")
        print(f"Total observations: {len(data)}")
        print(f"Minimum window size: {self.min_window_size}")
        print(f"Will detect regimes for {len(data) - self.min_window_size} periods")
        print(f"\nMethod: {'Mahalanobis distance with probability assignment' if use_mahalanobis else 'Hard threshold classification (above/below median)'}")
        
        # Storage for regime assignments and probabilities
        regime_assignments = []
        dates_list = []
        prob_columns = {0: [], 1: [], 2: [], 3: []}
        
        print("\nDetecting regimes with expanding windows...")
        
        # For each time point, calculate thresholds
        for t_idx in range(self.min_window_size, len(data)):
            # Get expanding window: start to current time
            window_data = data.iloc[:t_idx+1].copy()  # Include current time point
            
            # Step 1: Calculate thresholds on expanding window
            growth_threshold = window_data['growth_factor'].median()
            inflation_threshold = window_data['inflation_factor'].median()
            
            # Step 2: Get current observation
            current_point = window_data.iloc[-1]
            current_growth = current_point['growth_factor']
            current_inflation = current_point['inflation_factor']
            
            if use_mahalanobis:
                # MAHALANOBIS METHOD: Use cluster statistics and probabilities
                # Classify historical data into regimes (excluding current point for cluster estimation)
                historical_data = window_data.iloc[:-1].copy()
                historical_regimes = []
                
                for idx, row in historical_data.iterrows():
                    growth = row['growth_factor']
                    inflation = row['inflation_factor']
                    
                    if growth >= growth_threshold:
                        if inflation >= inflation_threshold:
                            regime = 1  # Overheating
                        else:
                            regime = 0  # Goldilocks
                    else:
                        if inflation >= inflation_threshold:
                            regime = 2  # Stagflation
                        else:
                            regime = 3  # Slowdown
                    
                    historical_regimes.append(regime)
                
                historical_data['regime'] = historical_regimes
                
                # Step 3: Compute cluster centers and covariance matrices for each regime
                regime_stats = {}
                for regime_id in [0, 1, 2, 3]:
                    regime_data = historical_data[historical_data['regime'] == regime_id][variables]
                    
                    if len(regime_data) >= 2:  # Need at least 2 points for covariance
                        mean = regime_data.mean().values
                        cov = regime_data.cov().values
                        
                        # Add small regularization to avoid singular matrices
                        cov += np.eye(2) * 1e-6
                        
                        regime_stats[regime_id] = {
                            'mean': mean,
                            'cov': cov,
                            'n_obs': len(regime_data)
                        }
                    else:
                        # If regime has too few observations, use overall statistics
                        mean = historical_data[variables].mean().values
                        cov = historical_data[variables].cov().values + np.eye(2) * 1e-6
                        regime_stats[regime_id] = {
                            'mean': mean,
                            'cov': cov,
                            'n_obs': 0
                        }
                
                # Step 4: Calculate Mahalanobis distance for current observation
                current_point_vec = np.array([current_growth, current_inflation])
                
                mahalanobis_distances = {}
                for regime_id in [0, 1, 2, 3]:
                    stats = regime_stats[regime_id]
                    mean = stats['mean']
                    cov = stats['cov']
                    
                    # Calculate Mahalanobis distance
                    diff = current_point_vec - mean
                    try:
                        cov_inv = np.linalg.inv(cov)
                        mahal_dist = np.sqrt(diff.T @ cov_inv @ diff)
                    except np.linalg.LinAlgError:
                        # Fallback if matrix is singular
                        mahal_dist = np.linalg.norm(diff)
                    
                    mahalanobis_distances[regime_id] = mahal_dist
                
                # Step 5: Convert distances to probabilities using softmax
                # Use negative distances so smaller distance = higher probability
                distances_array = np.array([mahalanobis_distances[i] for i in [0, 1, 2, 3]])
                
                # Softmax on negative distances (temperature parameter for smoothing)
                temperature = 1.0  # Can be adjusted
                exp_scores = np.exp(-distances_array / temperature)
                probabilities = exp_scores / exp_scores.sum()
                
                # Step 6: Assign regime (highest probability)
                regime = np.argmax(probabilities)
                
                # Store probabilities
                for regime_id in [0, 1, 2, 3]:
                    prob_columns[regime_id].append(probabilities[regime_id])
            else:
                # HARD THRESHOLD METHOD: Simple above/below median classification
                # Classify current observation directly
                if current_growth >= growth_threshold:
                    if current_inflation >= inflation_threshold:
                        regime = 1  # Overheating
                    else:
                        regime = 0  # Goldilocks
                else:
                    if current_inflation >= inflation_threshold:
                        regime = 2  # Stagflation
                    else:
                        regime = 3  # Slowdown
                
                # For hard threshold, probabilities are binary (1.0 for assigned regime, 0.0 for others)
                for regime_id in [0, 1, 2, 3]:
                    prob_columns[regime_id].append(1.0 if regime_id == regime else 0.0)
            
            # Store results
            dates_list.append(window_data.index[-1])
            regime_assignments.append(regime)
            
            if (t_idx - self.min_window_size + 1) % 50 == 0:
                print(f"  Processed {t_idx - self.min_window_size + 1} / {len(data) - self.min_window_size} periods")
        
        # Create DataFrame
        regime_names = {
            0: "Goldilocks",
            1: "Overheating",
            2: "Stagflation",
            3: "Slowdown"
        }
        
        results_df = pd.DataFrame({
            'date': dates_list,
            'regime': regime_assignments,
            'regime_name': [regime_names[r] for r in regime_assignments],
            'prob_R0': prob_columns[0],
            'prob_R1': prob_columns[1],
            'prob_R2': prob_columns[2],
            'prob_R3': prob_columns[3]
        })
        
        results_df = results_df.set_index('date')
        self.regime_assignments = results_df
        
        print(f"\nCompleted expanding window regime detection")
        print(f"  Detected regimes for {len(results_df)} periods")
        print(f"  Date range: {results_df.index.min()} to {results_df.index.max()}")
        if use_mahalanobis:
            print(f"\nProbability statistics:")
            for regime_id in [0, 1, 2, 3]:
                prob_col = f'prob_R{regime_id}'
                print(f"  {regime_names[regime_id]}: mean={results_df[prob_col].mean():.3f}, std={results_df[prob_col].std():.3f}")
        else:
            print(f"\nRegime distribution:")
            regime_counts = results_df['regime'].value_counts().sort_index()
            for regime_id, count in regime_counts.items():
                print(f"  {regime_names[regime_id]}: {count} periods ({count/len(results_df)*100:.1f}%)")
        
        return results_df
    
    def save_regime_assignments(self):
        """Save regime assignments to CSV."""
        if self.regime_assignments is None:
            raise ValueError("No regime assignments to save. Run detection first.")
        
        output_file = self.output_dir / 'regime_assignments.csv'
        self.regime_assignments.to_csv(output_file)
        print(f"\nSaved regime assignments to: {output_file}")
        
        return output_file
    
    def run_detection(self, use_mahalanobis: bool = True):
        """Run regime detection based on model type."""
        if self.regime_model == 'hmm_optimal':
            return self.detect_hmm_regimes_expanding()
        elif self.regime_model == '2x2':
            return self.detect_2x2_regimes_expanding(use_mahalanobis=use_mahalanobis)
        else:
            raise ValueError(f"Unknown regime model: {self.regime_model}")


def main():
    """Main execution function."""
    print("="*80)
    print("EXPANDING WINDOW REGIME DETECTION (No Look-Ahead Bias)")
    print("="*80)
    
    # Set up paths
    base_dir = get_project_root(__file__)
    
    # Detect regimes for both models
    models = ['hmm_optimal', '2x2']
    
    for model in models:
        print("\n" + "="*80)
        print(f"Detecting regimes: {model.upper()}")
        print("="*80)
        
        detector = ExpandingWindowRegimeDetector(
            data_dir=base_dir,
            regime_model=model,
            min_window_size=24  # 2 years minimum
        )
        
        # Run detection
        detector.run_detection()
        
        # Save results
        detector.save_regime_assignments()
        
        print(f"\nCompleted regime detection for {model}!")
        print(f"Results saved to: {detector.output_dir}")
    
    print("\n" + "="*80)
    print("ALL REGIME DETECTION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
