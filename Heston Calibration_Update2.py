from cmath import tau
from operator import call
from pyexpat import model
from unittest import result

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import KFold
from scipy.optimize import minimize, differential_evolution
import QuantLib as ql
import warnings

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, suppress=True)

# Display name mapping for objectives
OBJECTIVE_NAMES = {
    'sse': 'SSE',
    'vega_weighted': 'Vega-Weighted',
    'relative': 'Relative',
    'implied_vol': 'IV'
}

# ============================================================================
# 1. HESTON MODEL (Simplified)
# ============================================================================

class HestonModelQL:
    def __init__(self, S0, r, q, v0, kappa, theta, sigma, rho):
        """
        S0: spot price
        r: risk-free rate
        q: dividend yield
        v0: initial variance
        kappa: mean reversion speed
        theta: long-term variance
        sigma: volatility of volatility
        rho: correlation
        """
        self.S0 = S0
        self.r = r
        self.q = q
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
    
    def call_prices(self, strikes, maturities):
        """Compute Heston call prices using QuantLib."""
        prices = []
    
        for strike, tau in zip(strikes, maturities):
            K = float(strike)
            T = float(tau)
        
            today = ql.Date.todaysDate()
            riskFreeTS = ql.YieldTermStructureHandle(
                ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(self.r)), 
                            ql.Actual365Fixed())
            )
            dividendTS = ql.YieldTermStructureHandle(
                ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(self.q)), 
                            ql.Actual365Fixed())
            )
        
            heston_process = ql.HestonProcess(
                riskFreeTS, dividendTS,
                ql.QuoteHandle(ql.SimpleQuote(float(self.S0))),
                float(self.v0), float(self.kappa), float(self.theta), 
                float(self.sigma), float(self.rho)
            )
        
            engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process))
        
            expiry = today + ql.Period(int(T * 365), ql.Days)
            option = ql.VanillaOption(
                ql.PlainVanillaPayoff(ql.Option.Call, K),
                ql.EuropeanExercise(expiry)
            )
            option.setPricingEngine(engine)
        
            prices.append(option.NPV())
    
        return np.array(prices)
    
    def implied_vol(self, price, strike, tau):
        """NEW ROBUST VERSION"""
        from scipy.optimize import brentq, minimize_scalar
    
        def bs_call_price(sigma):
            d1 = (np.log(self.S0 / strike) + 
                (self.r - self.q + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
            d2 = d1 - sigma * np.sqrt(tau)
        
            call = (self.S0 * np.exp(-self.q * tau) * norm.cdf(d1) - 
                strike * np.exp(-self.r * tau) * norm.cdf(d2))
            return call - price
    
        intrinsic = max(self.S0 * np.exp(-self.q * tau) - 
                    strike * np.exp(-self.r * tau), 0)
    
        if price < intrinsic * 0.99:
            return 0.001
    
        try:
            f_low = bs_call_price(0.001)
            f_high = bs_call_price(5.0)
        
            if f_low * f_high > 0:
                result = minimize_scalar(lambda s: (bs_call_price(s))**2, 
                                    bounds=(0.001, 5.0), method='bounded')
                return max(0.001, result.x)
        
            iv = brentq(bs_call_price, 0.001, 5.0, maxiter=100, xtol=1e-6)
            return iv
        except Exception as e:
            return np.nan  # Return NaN, not 0.5


# ============================================================================
# 2. OBJECTIVE FUNCTIONS
# ============================================================================

class ObjectiveFunctions:
    """Collection of objective functions."""
    
    @staticmethod
    def sse(model_prices, market_prices):
        """Sum of squared errors."""
        return np.sum((model_prices - market_prices)**2)
    
    @staticmethod
    def vega_weighted(model_prices, market_prices, K, S0, r, q, tau):
        """Vega-weighted errors."""
        d1 = (np.log(S0 / K) + (r - q + 0.2**2 / 2) * tau) / (0.2 * np.sqrt(tau))
        weights = S0 * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(tau)
        weights = weights / np.sum(weights)
        diff = model_prices - market_prices
        return np.sum(weights * diff**2)
    
    @staticmethod
    def relative(model_prices, market_prices):
        """Relative squared errors."""
        rel_diff = (model_prices - market_prices) / np.maximum(market_prices, 1e-6)
        return np.sum(rel_diff**2)
    
    @staticmethod
    def implied_vol(model_prices, market_prices, K, S0, r, q, tau, model):
        """Implied vol errors (normalized by market IV)."""
        iv_model = np.array([model.implied_vol(p, k, t) 
                            for p, k, t in zip(model_prices, K, tau)])
        iv_market = np.array([model.implied_vol(p, k, t) 
                            for p, k, t in zip(market_prices, K, tau)])
    
        # Filter out NaN and invalid values
        valid = ~(np.isnan(iv_model) | np.isnan(iv_market))
    
        if np.sum(valid) < 2:
            return 1e10
    
        # Normalize by market IV to avoid scale issues
        iv_market_valid = iv_market[valid]
        iv_model_valid = iv_model[valid]
    
        # Avoid division by zero
        normalized_diff = (iv_model_valid - iv_market_valid) / np.maximum(np.abs(iv_market_valid), 0.01)
    
        return np.mean(normalized_diff**2)


# ============================================================================
# 3. CALIBRATOR
# ============================================================================

class HestonCalibrator:
    def __init__(self, S0, r, q):
        self.S0 = S0
        self.r = r
        self.q = q
    
    def calibrate(self, K, prices, tau, obj_type='sse'):
        """Calibrate using scipy.optimize with custom objective functions."""
    
        # Initial guess
        x0 = np.array([0.04, 0.04, 1.5, 0.3, -0.5])
    
        # Bounds: [v0, theta, kappa, sigma, rho]
        bounds = [
            (0.001, 1.0),      # v0
            (0.001, 1.0),      # theta
            (0.01, 10.0),      # kappa
            (0.01, 2.0),       # sigma
            (-0.999, 0.999)    # rho
        ]
    
        def objective(x):
            """Wrapper for all objectives."""
            v0, theta_param, kappa, sigma, rho = x
        
            model = HestonModelQL(self.S0, self.r, self.q, v0, kappa, theta_param, sigma, rho)
        
            try:
                model_prices = model.call_prices(K, tau)
            except:
                return 1e10
        
            if obj_type == 'sse':
                return ObjectiveFunctions.sse(model_prices, prices)
            elif obj_type == 'vega_weighted':
                return ObjectiveFunctions.vega_weighted(model_prices, prices, K, self.S0, 
                                                       self.r, self.q, tau)
            elif obj_type == 'relative':
                return ObjectiveFunctions.relative(model_prices, prices)
            elif obj_type == 'implied_vol':
                return ObjectiveFunctions.implied_vol(model_prices, prices, K, self.S0, 
                                                  self.r, self.q, tau, model)
            else:
                return ObjectiveFunctions.sse(model_prices, prices)
    
        # ===== USE DIFFERENTIAL EVOLUTION FOR IV =====
        if obj_type == 'implied_vol':
            result = differential_evolution(objective, bounds, seed=42, 
                                           maxiter=500, atol=1e-4, tol=1e-4,
                                           workers=1, updating='immediate')
        else:
            # Keep L-BFGS-B for price-based objectives (faster)
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-6})
    
        v0_opt, theta_opt, kappa_opt, sigma_opt, rho_opt = result.x
        theta = np.array([v0_opt, theta_opt, rho_opt, kappa_opt, sigma_opt])
    
        return theta, result.fun
    
# ============================================================================
# 4. DATA LOADING
# ============================================================================

def load_csv_data(csv_path, target_date=None):
    """Load option data from CSV (handles multiple expiration dates)."""
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(r'[\[\]]', '', regex=True)
    
    df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
    
    if target_date:
        df = df[df['QUOTE_DATE'] == pd.to_datetime(target_date)]
    else:
        df = df[df['QUOTE_DATE'] == df['QUOTE_DATE'].max()]
    
    spot = df['UNDERLYING_LAST'].iloc[0]
    quote_date = df['QUOTE_DATE'].iloc[0]
    
    # Clean data
    df = df[['STRIKE', 'C_BID', 'C_ASK', 'EXPIRE_DATE']].copy()
    df.columns = ['K', 'bid', 'ask', 'exp']
    
    df['exp'] = pd.to_datetime(df['exp'], unit='ns')
    df[['K', 'bid', 'ask']] = df[['K', 'bid', 'ask']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['moneyness'] = df['K'] / spot
    
    # Filter
    df = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['bid'] <= df['ask'])]
    df = df[(df['moneyness'] >= 0.75) & (df['moneyness'] <= 1.25)]
    df = df.drop_duplicates(subset=['K']).sort_values('K')
    
    # ===== KEEP ALL EXPIRATIONS > 7 DAYS =====
    df['days_to_exp'] = (df['exp'] - pd.Timestamp(quote_date)).dt.days
    df = df[df['days_to_exp'] >= 7]  # Filter here
    
    print(f"✓ Loaded {len(df)} options (all expirations ≥ 7 days)")
    print(f"  Spot: ${spot:.2f}, Quote Date: {quote_date.date()}")
    
    print(f"\nExpiration dates included:")
    unique_exps = sorted(df['exp'].unique())
    for exp in unique_exps:
        days = (exp - pd.Timestamp(quote_date)).days
        count = len(df[df['exp'] == exp])
        print(f"  {exp.date()}: {days:3d} days, {count:2d} options")
    
    # Calculate tau for each option
    df['tau'] = df['days_to_exp'] / 365.0
    
    print(f"\n  Total options: {len(df)}")
    
    return df[['K', 'mid', 'tau']].values, spot  # Return tau with data

# ============================================================================
# 5. ANALYSIS
# ============================================================================

def run_analysis(csv_path, r=0.05, q=0.0):
    """Complete analysis: in-sample, out-of-sample, robustness."""
    
    # Load data
    data, spot = load_csv_data(csv_path)
    K = data[:, 0]
    prices = data[:, 1]
    tau = data[:, 2]

    # ===== DIAGNOSTICS =====
    print("\n" + "="*80)
    print("DIAGNOSTICS")
    print("="*80)
    print(f"Market prices - Min: ${prices.min():.2f}, Max: ${prices.max():.2f}, Mean: ${prices.mean():.2f}")
    print(f"Strikes - Min: ${K.min():.2f}, Max: ${K.max():.2f}, Moneyness: {K.min()/spot:.3f}-{K.max()/spot:.3f}")
    print(f"Intrinsic value (ATM): ${max(spot - K[len(K)//2], 0):.2f}")
    print(f"Time to expiration: Min={tau.min():.6f} years ({tau.min()*365:.1f} days), "
          f"Max={tau.max():.6f} years ({tau.max()*365:.1f} days)")
    
    # Quick test calibration
    calibrator = HestonCalibrator(spot, r, q)
    theta_test, _ = calibrator.calibrate(K, prices, tau, 'sse')
    model_test = HestonModelQL(spot, r, q, 
                      theta_test[0],  # v0
                      theta_test[3],  # kappa
                      theta_test[1],  # theta
                      theta_test[4],  # sigma
                      theta_test[2])  # rho
    prices_test = model_test.call_prices(K, tau)
    
    print(f"\nQUICK MODEL TEST (SSE):")
    print(f"  Calibrated θ = {theta_test}")
    print(f"  Model prices - Min: ${prices_test.min():.2f}, Max: ${prices_test.max():.2f}, Mean: ${prices_test.mean():.2f}")
    print(f"\n  First 10 prices (K, Market, Model, Error):")
    for i in range(min(10, len(K))):
        err = prices_test[i] - prices[i]
        err_pct = (err / prices[i] * 100) if prices[i] > 0 else 0
        print(f"    ${K[i]:7.2f}: ${prices[i]:7.2f} → ${prices_test[i]:7.2f} (${err:+7.2f}, {err_pct:+7.1f}%)")
    
    # ===== END DIAGNOSTICS =====
    objectives = ['sse', 'vega_weighted', 'relative', 'implied_vol']
    
    results = {}
    
    for obj in objectives:
        print(f"\n[{obj}]")
        
        # === IN-SAMPLE: Full dataset ===
        theta_opt, obj_val = calibrator.calibrate(K, prices, tau, obj)
        model = HestonModelQL(spot, r, q, 
                            theta_opt[0],  # v0
                            theta_opt[3],  # kappa
                            theta_opt[1],  # theta
                            theta_opt[4],  # sigma
                            theta_opt[2])  # rho
        prices_pred = model.call_prices(K, tau)
        
        # Absolute errors
        in_sample_rmse = np.sqrt(np.mean((prices_pred - prices)**2))
        in_sample_mae = np.mean(np.abs(prices_pred - prices))
        
        # Relative errors (%)
        in_sample_rmse_rel = np.sqrt(np.mean(((prices_pred - prices) / prices)**2)) * 100
        in_sample_mae_rel = np.mean(np.abs(prices_pred - prices) / prices) * 100
        
        # === OUT-OF-SAMPLE: 5-fold CV ===
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oos_rmses = []
        oos_rmses_rel = []
        
        for train_idx, test_idx in kf.split(K):
            K_train, K_test = K[train_idx], K[test_idx]
            tau_train, tau_test = tau[train_idx], tau[test_idx]
            prices_train, prices_test_cv = prices[train_idx], prices[test_idx]
            
            theta_cv, _ = calibrator.calibrate(K_train, prices_train, tau_train, obj)
            model_cv = HestonModelQL(spot, r, q, 
                                    theta_cv[0],  # v0
                                    theta_cv[3],  # kappa
                                    theta_cv[1],  # theta
                                    theta_cv[4],  # sigma
                                    theta_cv[2])  # rho
            prices_test_pred = model_cv.call_prices(K_test, tau_test)
            
            oos_rmses.append(np.sqrt(np.mean((prices_test_pred - prices_test_cv)**2)))
            oos_rmses_rel.append(np.sqrt(np.mean(((prices_test_pred - prices_test_cv) / prices_test_cv)**2)) * 100)
        
        oos_rmse_mean = np.mean(oos_rmses)
        oos_rmse_std = np.std(oos_rmses)
        oos_rmse_rel_mean = np.mean(oos_rmses_rel)
        oos_rmse_rel_std = np.std(oos_rmses_rel)
        
        # === ROBUSTNESS: Parameter stability across folds ===
        param_list = []
        for train_idx, test_idx in kf.split(K):
            K_train = K[train_idx]
            tau_train = tau[train_idx]
            prices_train = prices[train_idx]
            theta_cv, _ = calibrator.calibrate(K_train, prices_train, tau_train, obj)
            param_list.append(theta_cv)
        
        param_array = np.array(param_list)
        param_cv = np.std(param_array, axis=0) / np.mean(np.abs(param_array), axis=0)
        
        results[obj] = {
            'theta': theta_opt,
            'in_sample_rmse': in_sample_rmse,
            'in_sample_rmse_rel': in_sample_rmse_rel,
            'in_sample_mae': in_sample_mae,
            'in_sample_mae_rel': in_sample_mae_rel,
            'oos_rmse_mean': oos_rmse_mean,
            'oos_rmse_std': oos_rmse_std,
            'oos_rmse_rel_mean': oos_rmse_rel_mean,
            'oos_rmse_rel_std': oos_rmse_rel_std,
            'param_stability': param_cv,
            'prices_pred': prices_pred,
            'oos_rmses': oos_rmses
        }
        
        print(f"  ✓ In-sample RMSE: ${in_sample_rmse:.4f}")
        print(f"  ✓ OOS RMSE: ${oos_rmse_mean:.4f} ± ${oos_rmse_std:.4f}")
        print(f"  ✓ Gen. Gap: ${oos_rmse_mean - in_sample_rmse:.4f}")
        print(f"  ✓ Param Stability (CV): {param_cv}")
    
    return results, K, prices, spot, tau


def plot_comparison(results, K, prices, spot):
    """Compare objectives with absolute and relative errors."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    objs = list(results.keys())
    obj_labels = [OBJECTIVE_NAMES[obj] for obj in objs]
    
    # ===== ROW 1: ABSOLUTE ERRORS =====
    
    # Plot 1: In-sample vs Out-of-sample (Absolute $)
    ax = axes[0, 0]
    in_sample = [results[obj]['in_sample_rmse'] for obj in objs]
    oos_mean = [results[obj]['oos_rmse_mean'] for obj in objs]
    x = np.arange(len(objs))
    width = 0.35
    ax.bar(x - width/2, in_sample, width, label='In-sample', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, oos_mean, width, label='Out-of-sample', alpha=0.8, color='coral')
    ax.set_ylabel('RMSE ($)', fontweight='bold')
    ax.set_title('Absolute Error: RMSE ($)', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(obj_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Generalization gap (Absolute $)
    ax = axes[0, 1]
    gen_gaps = [results[obj]['oos_rmse_mean'] - results[obj]['in_sample_rmse'] for obj in objs]
    colors = ['green' if gap < 1 else 'orange' if gap < 3 else 'red' for gap in gen_gaps]
    bars = ax.bar(x, gen_gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('OOS - In-Sample ($)', fontweight='bold')
    ax.set_title('Generalization Gap (Absolute)', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(obj_labels, rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')
    for bar, gap in zip(bars, gen_gaps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{gap:.3f}',
               ha='center', va='bottom' if gap > 0 else 'top', fontsize=9)
    
    # Plot 3: Price fit
    ax = axes[0, 2]
    sorted_idx = np.argsort(K)
    ax.plot(K[sorted_idx], prices[sorted_idx], 'o-', label='Market', color='black', 
           linewidth=2.5, markersize=6, zorder=5)
    colors_objs = plt.cm.Set2(np.linspace(0, 1, len(objs)))
    for obj, color in zip(objs, colors_objs):
        ax.plot(K[sorted_idx], results[obj]['prices_pred'][sorted_idx], '--', 
               label=OBJECTIVE_NAMES[obj], alpha=0.7, linewidth=2, color=color)
    ax.set_xlabel('Strike ($)', fontweight='bold')
    ax.set_ylabel('Option Price ($)', fontweight='bold')
    ax.set_title('Model vs Market Prices', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # ===== ROW 2: RELATIVE ERRORS =====
    
    # Plot 4: In-sample vs Out-of-sample (Relative %)
    ax = axes[1, 0]
    in_sample_rel = [results[obj]['in_sample_rmse_rel'] for obj in objs]
    oos_rel_mean = [results[obj]['oos_rmse_rel_mean'] for obj in objs]
    ax.bar(x - width/2, in_sample_rel, width, label='In-sample', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, oos_rel_mean, width, label='Out-of-sample', alpha=0.8, color='coral')
    ax.set_ylabel('RMSE (%)', fontweight='bold')
    ax.set_title('Relative Error: RMSE (%)', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(obj_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 5: Generalization gap (Relative %)
    ax = axes[1, 1]
    gen_gaps_rel = [results[obj]['oos_rmse_rel_mean'] - results[obj]['in_sample_rmse_rel'] for obj in objs]
    colors = ['green' if gap < 0.5 else 'orange' if gap < 1.5 else 'red' for gap in gen_gaps_rel]
    bars = ax.bar(x, gen_gaps_rel, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('OOS - In-Sample (%)', fontweight='bold')
    ax.set_title('Generalization Gap (Relative %)', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(obj_labels, rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')
    for bar, gap in zip(bars, gen_gaps_rel):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{gap:.2f}%',
               ha='center', va='bottom' if gap > 0 else 'top', fontsize=9)
    
    # Plot 6: Parameter stability
    ax = axes[1, 2]
    param_names = ['ν0', 'θ', 'ρ', 'κ', 'σ']
    for obj, color in zip(objs, colors_objs):
        stability = results[obj]['param_stability']
        ax.plot(param_names, stability, 'o-', label=OBJECTIVE_NAMES[obj], linewidth=2, markersize=8, color=color)
    ax.set_ylabel('Coefficient of Variation', fontweight='bold')
    ax.set_title('Parameter Stability', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# 6. MAIN
# ============================================================================

if __name__ == "__main__":
    
    csv_path = "tsla_2019_2022.csv"
    
    print("\n" + "="*80)
    print("HESTON CALIBRATION: In-Sample vs Out-of-Sample Analysis")
    print("="*80)
    
    try:
        results, K, prices, spot, tau = run_analysis(csv_path, r=0.05, q=0.0)
        
        # Summary table
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        df_summary = pd.DataFrame({
            'Objective': [OBJECTIVE_NAMES[obj] for obj in results.keys()],
            'In-Sample RMSE': [results[obj]['in_sample_rmse'] for obj in results.keys()],
            'In-Sample MAE': [results[obj]['in_sample_mae'] for obj in results.keys()],
            'OOS RMSE': [results[obj]['oos_rmse_mean'] for obj in results.keys()],
            'OOS Std': [results[obj]['oos_rmse_std'] for obj in results.keys()],
            'Gen. Gap': [results[obj]['oos_rmse_mean'] - results[obj]['in_sample_rmse'] 
                        for obj in results.keys()],
            'Param CV (ρ)': [results[obj]['param_stability'][2] for obj in results.keys()]
        })
        print("\n" + df_summary.to_string(index=False))
        
        # Parameter table
        print("\n" + "="*80)
        print("CALIBRATED PARAMETERS")
        print("="*80)
        
        param_names = ['ν0', 'θ', 'ρ', 'κ', 'σ']
        df_params = pd.DataFrame({
            'Objective': [OBJECTIVE_NAMES[obj] for obj in results.keys()],
            **{name: [results[obj]['theta'][i] for obj in results.keys()] 
               for i, name in enumerate(param_names)}
        })
        print("\n" + df_params.to_string(index=False))
        
        # Plots
        fig = plot_comparison(results, K, prices, spot)
        plt.savefig('heston_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved: heston_comparison.png")
        plt.show()
        
        # Additional analysis: Best objective by metric
        print("\n" + "="*80)
        print("BEST PERFORMERS")
        print("="*80)
        
        best_in_rmse = df_summary.loc[df_summary['In-Sample RMSE'].idxmin()]
        best_oos_rmse = df_summary.loc[df_summary['OOS RMSE'].idxmin()]
        best_gen_gap = df_summary.loc[df_summary['Gen. Gap'].idxmin()]
        best_stability = df_summary.loc[df_summary['Param CV (ρ)'].idxmin()]
        
        print(f"\n✓ Best In-Sample RMSE: {best_in_rmse['Objective']} (${best_in_rmse['In-Sample RMSE']:.4f})")
        print(f"✓ Best OOS RMSE: {best_oos_rmse['Objective']} (${best_oos_rmse['OOS RMSE']:.4f})")
        print(f"✓ Smallest Gen. Gap: {best_gen_gap['Objective']} (${best_gen_gap['Gen. Gap']:.4f})")
        print(f"✓ Most Stable Params: {best_stability['Objective']} (CV={best_stability['Param CV (ρ)']:.4f})")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()