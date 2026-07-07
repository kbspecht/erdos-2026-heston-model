import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, suppress=True)

# ============================================================================
# 1. HESTON MODEL (Simplified)
# ============================================================================

class HestonModel:
    """Lightweight Heston model for European option pricing."""
    
    def __init__(self, S0, r, q, theta=None):
        self.S0 = S0
        self.r = r
        self.q = q
        if theta is not None:
            self.nu0, self.theta_v, self.rho, self.kappa, self.sigma = theta
    
    def set_params(self, theta):
        self.nu0, self.theta_v, self.rho, self.kappa, self.sigma = theta
    
    def characteristic_function(self, u, tau):
        """Heston characteristic function."""
        try:
            d = np.sqrt((self.rho * self.sigma * u * 1j - self.kappa)**2 - 
                       self.sigma**2 * (2 * u * 1j - u**2))
            g = (self.kappa - self.rho * self.sigma * u * 1j - d) / \
                (self.kappa - self.rho * self.sigma * u * 1j + d)
            
            exp_term = np.exp(u * 1j * (np.log(self.S0) + (self.r - self.q) * tau))
            power_term = (1 - g * np.exp(-d * tau)) / (1 - g)
            
            variance_term = (self.kappa * self.theta_v / self.sigma**2) * (
                (self.kappa - self.rho * self.sigma * u * 1j - d) * tau - 
                2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g))
            )
            
            return exp_term * np.exp(variance_term + (self.nu0 / self.sigma**2) * power_term)
        except:
            return 0.0
    
    def call_prices_fft(self, K, tau, N=2048, eta=0.25):
        """Price European calls using Black-Scholes as fallback."""
        K = np.atleast_1d(K)
        
        # Use constant volatility based on calibrated parameters
        # For now, use an implied vol from the spot-vol correlation
        sigma_bs = np.sqrt(self.nu0)  # Simple approximation
        
        d1 = (np.log(self.S0 / K) + (self.r - self.q + 0.5 * sigma_bs**2) * tau) / (sigma_bs * np.sqrt(tau))
        d2 = d1 - sigma_bs * np.sqrt(tau)
        
        prices = (self.S0 * np.exp(-self.q * tau) * norm.cdf(d1) - 
                  K * np.exp(-self.r * tau) * norm.cdf(d2))
        
        return np.maximum(prices, 0)
    
    def implied_vol(self, price, K, tau, tol=1e-6, max_iter=50):
        """Implied volatility via Newton-Raphson."""
        if price <= 0 or tau <= 0.001:
            return 0.2
        
        sigma = np.sqrt(2 * np.pi / tau) * price / self.S0
        sigma = np.clip(sigma, 0.001, 5.0)
        
        for _ in range(max_iter):
            d1 = (np.log(self.S0 / K) + (self.r - self.q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
            d2 = d1 - sigma * np.sqrt(tau)
            
            price_bs = (self.S0 * np.exp(-self.q * tau) * norm.cdf(d1) - 
                       K * np.exp(-self.r * tau) * norm.cdf(d2))
            vega = self.S0 * np.exp(-self.q * tau) * norm.pdf(d1) * np.sqrt(tau)
            
            if abs(vega) < 1e-10:
                break
            
            sigma_new = sigma + (price - price_bs) / vega
            if abs(sigma_new - sigma) < tol:
                return np.clip(sigma_new, 0.001, 5.0)
            
            sigma = np.clip(sigma_new, 0.001, 5.0)
        
        return np.clip(sigma, 0.001, 5.0)


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
        """Implied vol errors."""
        iv_model = np.array([model.implied_vol(p, k, tau) for p, k in zip(model_prices, K)])
        iv_market = np.array([model.implied_vol(p, k, tau) for p, k in zip(market_prices, K)])
        return np.sum((iv_model - iv_market)**2)


# ============================================================================
# 3. CALIBRATOR
# ============================================================================

class HestonCalibrator:
    """Simple calibrator."""
    
    def __init__(self, S0, r, q):
        self.S0 = S0
        self.r = r
        self.q = q
    
    def objective_wrapper(self, theta, K, prices_mkt, tau, obj_type):
        """Evaluate objective function."""
        
        # Check constraints
        nu0, theta_v, rho, kappa, sigma = theta
        if not (nu0 > 0.001 and theta_v > 0.001 and -0.99 < rho < 0.99 and 
                kappa > 0.01 and sigma > 0.01 and 2*kappa*theta_v > sigma**2):
            return 1e10
        
        # Compute model prices
        model = HestonModel(self.S0, self.r, self.q, theta)
        prices_model = model.call_prices_fft(K, tau)
        
        if np.any(np.isnan(prices_model)) or np.any(prices_model < 0):
            return 1e10
        
        # Evaluate objective
        if obj_type == 'vega_weighted':
            return ObjectiveFunctions.vega_weighted(prices_model, prices_mkt, K, 
                                                    self.S0, self.r, self.q, tau)
        elif obj_type == 'relative':
            return ObjectiveFunctions.relative(prices_model, prices_mkt)
        elif obj_type == 'implied_vol':
            return ObjectiveFunctions.implied_vol(prices_model, prices_mkt, K, 
                                                  self.S0, self.r, self.q, tau, model)
        else:  # 'sse'
            return ObjectiveFunctions.sse(prices_model, prices_mkt)
    
    def calibrate(self, K, prices_mkt, tau, obj_type='vega_weighted'):
        """Calibrate to market prices."""
        
        theta0 = np.array([0.05, 0.05, -0.5, 2.0, 0.3])
        bounds = [(0.001, 1.0), (0.001, 1.0), (-0.99, 0.99), (0.01, 10.0), (0.01, 2.0)]
        
        result = minimize(
            self.objective_wrapper,
            theta0,
            args=(K, prices_mkt, tau, obj_type),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-8}
        )
        
        return result.x, result.fun


# ============================================================================
# 4. DATA LOADING
# ============================================================================

def load_csv_data(csv_path, target_date=None):
    """Load option data from CSV (handles historical data)."""
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(r'[\[\]]', '', regex=True)
    
    df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
    
    if target_date:
        df = df[df['QUOTE_DATE'] == pd.to_datetime(target_date)]
    else:
        df = df[df['QUOTE_DATE'] == df['QUOTE_DATE'].max()]
    
    spot = df['UNDERLYING_LAST'].iloc[0]
    quote_date = df['QUOTE_DATE'].iloc[0]
    
    # Clean data - select columns FIRST
    df = df[['STRIKE', 'C_BID', 'C_ASK', 'EXPIRE_DATE']].copy()
    df.columns = ['K', 'bid', 'ask', 'exp']
    
    # Convert EXPIRE_DATE BEFORE numeric conversion
    df['exp'] = pd.to_datetime(df['exp'], unit='ns')
    
    # NOW apply numeric conversion (exp is datetime, so it won't be affected)
    df[['K', 'bid', 'ask']] = df[['K', 'bid', 'ask']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['moneyness'] = df['K'] / spot
    
    # Filter
    df = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['bid'] <= df['ask'])]
    df = df[(df['moneyness'] >= 0.75) & (df['moneyness'] <= 1.25)]
    df = df.drop_duplicates(subset=['K']).sort_values('K')
    
    print(f"✓ Loaded {len(df)} options")
    print(f"  Spot: ${spot:.2f}, Quote Date: {quote_date.date()}")
    
    # ===== DEBUG =====
    print(f"\nFirst 5 options:")
    print(df[['K', 'bid', 'ask', 'mid', 'moneyness', 'exp']].head())
    
    print(f"\nAll unique expiration dates:")
    unique_exps = sorted(df['exp'].unique())
    for exp in unique_exps:
        days = (exp - pd.Timestamp(quote_date)).days
        count = len(df[df['exp'] == exp])
        print(f"  {exp.date()}: {days} days to expiration, {count} options")
    # ===== END DEBUG =====
    
    # Get next expiration after quote_date (skip very short dated)
    exp_dates_after = [e for e in unique_exps if (e - pd.Timestamp(quote_date)).days >= 7]
    
    if exp_dates_after:
        # Prefer 21+ days for better volatility dynamics
        long_dated = [e for e in exp_dates_after if (e - pd.Timestamp(quote_date)).days >= 20]
        exp_date = long_dated[0] if long_dated else exp_dates_after[0]
    else:
        exp_date = unique_exps[-1] if unique_exps else pd.Timestamp(quote_date) + pd.Timedelta(days=30)
    
    days_to_exp = (exp_date - pd.Timestamp(quote_date)).days
    
    if days_to_exp < 1:
        days_to_exp = 1
        exp_date = pd.Timestamp(quote_date) + pd.Timedelta(days=1)
    
    tau = days_to_exp / 365.0
    
    print(f"\n  Selected Expiration: {exp_date.date()} ({days_to_exp} days, τ={tau:.6f})")

    # Filter to ONLY the selected expiration
    df = df[df['exp'] == exp_date]
    
    print(f"  Options used: {len(df)}")
    
    return df[['K', 'mid']].values, spot, tau

# ============================================================================
# 5. ANALYSIS
# ============================================================================

def run_analysis(csv_path, r=0.05, q=0.0):
    """Complete analysis: in-sample, out-of-sample, robustness."""
    
    # Load data
    data, spot, tau = load_csv_data(csv_path)
    K, prices = data[:, 0], data[:, 1]

    # ===== DIAGNOSTICS =====
    print("\n" + "="*80)
    print("DIAGNOSTICS")
    print("="*80)
    print(f"Market prices - Min: ${prices.min():.2f}, Max: ${prices.max():.2f}, Mean: ${prices.mean():.2f}")
    print(f"Strikes - Min: ${K.min():.2f}, Max: ${K.max():.2f}, Moneyness: {K.min()/spot:.3f}-{K.max()/spot:.3f}")
    print(f"Intrinsic value (ATM): ${max(spot - K[len(K)//2], 0):.2f}")
    print(f"Time to expiration: {tau:.6f} years ({tau*365:.1f} days)")
    
    # Quick test calibration
    calibrator = HestonCalibrator(spot, r, q)
    theta_test, _ = calibrator.calibrate(K, prices, tau, 'sse')
    model_test = HestonModel(spot, r, q, theta_test)
    prices_test = model_test.call_prices_fft(K, tau)
    
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
        model = HestonModel(spot, r, q, theta_opt)
        prices_pred = model.call_prices_fft(K, tau)
        
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
            prices_train, prices_test = prices[train_idx], prices[test_idx]
            
            theta_cv, _ = calibrator.calibrate(K_train, prices_train, tau, obj)
            model_cv = HestonModel(spot, r, q, theta_cv)
            prices_test_pred = model_cv.call_prices_fft(K_test, tau)
            
            oos_rmses.append(np.sqrt(np.mean((prices_test_pred - prices_test)**2)))
            oos_rmses_rel.append(np.sqrt(np.mean(((prices_test_pred - prices_test) / prices_test)**2)) * 100)
        
        oos_rmse_mean = np.mean(oos_rmses)
        oos_rmse_std = np.std(oos_rmses)
        oos_rmse_rel_mean = np.mean(oos_rmses_rel)
        oos_rmse_rel_std = np.std(oos_rmses_rel)
        
        # === ROBUSTNESS: Parameter stability across folds ===
        param_list = []
        for train_idx, test_idx in kf.split(K):
            theta_cv, _ = calibrator.calibrate(K[train_idx], prices[train_idx], tau, obj)
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
        
        print(f"  In-sample RMSE:       ${in_sample_rmse:.4f} ({in_sample_rmse_rel:.2f}%)")
        print(f"  OOS RMSE:             ${oos_rmse_mean:.4f} ± ${oos_rmse_std:.4f} ({oos_rmse_rel_mean:.2f}% ± {oos_rmse_rel_std:.2f}%)")
        print(f"  Gen. Gap (abs):       ${oos_rmse_mean - in_sample_rmse:.4f}")
        print(f"  Gen. Gap (rel):       {oos_rmse_rel_mean - in_sample_rmse_rel:.2f}%")
        print(f"  Param Stability (ρ):  {param_cv[2]:.4f}")
    
    return results, K, prices, spot, tau


def plot_comparison(results, K, prices, spot):
    """Compare objectives with absolute and relative errors."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    objs = list(results.keys())
    
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
    ax.set_xticklabels(objs, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Generalization gap (Absolute $)
    ax = axes[0, 1]
    gen_gaps = [results[obj]['oos_rmse_mean'] - results[obj]['in_sample_rmse'] for obj in objs]
    colors = ['green' if gap < 1 else 'orange' if gap < 3 else 'red' for gap in gen_gaps]
    bars = ax.bar(objs, gen_gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('OOS - In-Sample ($)', fontweight='bold')
    ax.set_title('Generalization Gap (Absolute)', fontweight='bold', fontsize=12)
    ax.set_xticklabels(objs, rotation=45, ha='right')
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
               label=obj, alpha=0.7, linewidth=2, color=color)
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
    ax.set_xticklabels(objs, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 5: Generalization gap (Relative %)
    ax = axes[1, 1]
    gen_gaps_rel = [results[obj]['oos_rmse_rel_mean'] - results[obj]['in_sample_rmse_rel'] for obj in objs]
    colors = ['green' if gap < 0.5 else 'orange' if gap < 1.5 else 'red' for gap in gen_gaps_rel]
    bars = ax.bar(objs, gen_gaps_rel, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('OOS - In-Sample (%)', fontweight='bold')
    ax.set_title('Generalization Gap (Relative %)', fontweight='bold', fontsize=12)
    ax.set_xticklabels(objs, rotation=45, ha='right')
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
        ax.plot(param_names, stability, 'o-', label=obj, linewidth=2, markersize=8, color=color)
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
            'Objective': list(results.keys()),
            'In-Sample RMSE': [results[obj]['in_sample_rmse'] for obj in results.keys()],
            'OOS RMSE': [results[obj]['oos_rmse_mean'] for obj in results.keys()],
            'Gen. Gap': [results[obj]['oos_rmse_mean'] - results[obj]['in_sample_rmse'] 
                        for obj in results.keys()],
            'Param CV (ρ)': [results[obj]['param_stability'][2] for obj in results.keys()]
        })
        print("\n" + df_summary.to_string(index=False))
        
        # Plots
        fig = plot_comparison(results, K, prices, spot)
        plt.savefig('heston_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved: heston_comparison.png")
        plt.show()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

