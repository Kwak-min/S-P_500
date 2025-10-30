"""
리스크 측정 지표 계산 스크립트
VaR, CVaR, Sharpe Ratio, Maximum Drawdown 등을 계산합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data(filename='../part1_data_collection/sp500_returns.csv'):
    """
    데이터를 로드합니다.
    """
    print("=" * 60)
    print("데이터 로드")
    print("=" * 60)
    
    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"✓ 데이터 로드 완료: {len(df)} 행")
        return df
    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {filename}")
        raise

def calculate_var_historical(returns, confidence_level=0.95):
    """
    Historical VaR을 계산합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    confidence_level : float
        신뢰수준
    
    Returns:
    --------
    float
        VaR 값
    """
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

def calculate_var_parametric(returns, confidence_level=0.95):
    """
    Parametric VaR을 계산합니다 (정규분포 가정).
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    confidence_level : float
        신뢰수준
    
    Returns:
    --------
    float
        VaR 값
    """
    mu = returns.mean()
    sigma = returns.std()
    var = stats.norm.ppf(1 - confidence_level, mu, sigma)
    return var

def calculate_var_monte_carlo(returns, confidence_level=0.95, n_simulations=10000):
    """
    Monte Carlo VaR을 계산합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    confidence_level : float
        신뢰수준
    n_simulations : int
        시뮬레이션 횟수
    
    Returns:
    --------
    float
        VaR 값
    """
    mu = returns.mean()
    sigma = returns.std()
    
    # Monte Carlo 시뮬레이션
    simulated_returns = np.random.normal(mu, sigma, n_simulations)
    var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    
    return var

def calculate_cvar(returns, confidence_level=0.95):
    """
    Conditional VaR (CVaR 또는 Expected Shortfall)을 계산합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    confidence_level : float
        신뢰수준
    
    Returns:
    --------
    float
        CVaR 값
    """
    var = calculate_var_historical(returns, confidence_level)
    # VaR을 초과하는 손실의 평균
    cvar = returns[returns <= var].mean()
    return cvar

def calculate_all_var_metrics(returns):
    """
    모든 VaR 지표를 계산합니다.
    """
    print("\n" + "=" * 60)
    print("Value at Risk (VaR) 계산")
    print("=" * 60)
    
    confidence_levels = [0.95, 0.99]
    results = {}
    
    for conf in confidence_levels:
        print(f"\n신뢰수준: {conf*100:.0f}%")
        
        # Historical VaR
        var_hist = calculate_var_historical(returns, conf)
        print(f"  Historical VaR:   {var_hist:.6f} ({var_hist*100:.4f}%)")
        
        # Parametric VaR
        var_param = calculate_var_parametric(returns, conf)
        print(f"  Parametric VaR:   {var_param:.6f} ({var_param*100:.4f}%)")
        
        # Monte Carlo VaR
        var_mc = calculate_var_monte_carlo(returns, conf)
        print(f"  Monte Carlo VaR:  {var_mc:.6f} ({var_mc*100:.4f}%)")
        
        # CVaR
        cvar = calculate_cvar(returns, conf)
        print(f"  CVaR (ES):        {cvar:.6f} ({cvar*100:.4f}%)")
        
        results[f'VaR_Historical_{int(conf*100)}'] = var_hist
        results[f'VaR_Parametric_{int(conf*100)}'] = var_param
        results[f'VaR_MonteCarlo_{int(conf*100)}'] = var_mc
        results[f'CVaR_{int(conf*100)}'] = cvar
    
    return results

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Sharpe Ratio를 계산합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    risk_free_rate : float
        무위험 이자율 (연율)
    
    Returns:
    --------
    float
        Sharpe Ratio
    """
    # 일별 무위험 이자율
    rf_daily = risk_free_rate / 252
    
    # 초과 수익률
    excess_returns = returns - rf_daily
    
    # Sharpe Ratio
    sharpe = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
    """
    Sortino Ratio를 계산합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    risk_free_rate : float
        무위험 이자율 (연율)
    target_return : float
        목표 수익률
    
    Returns:
    --------
    float
        Sortino Ratio
    """
    # 일별 무위험 이자율
    rf_daily = risk_free_rate / 252
    
    # 초과 수익률
    excess_returns = returns - rf_daily
    
    # 하방 편차 (downside deviation)
    downside_returns = returns[returns < target_return]
    downside_std = downside_returns.std()
    
    # Sortino Ratio
    sortino = (excess_returns.mean() * 252) / (downside_std * np.sqrt(252))
    
    return sortino

def calculate_maximum_drawdown(prices):
    """
    Maximum Drawdown을 계산합니다.
    
    Parameters:
    -----------
    prices : pd.Series
        가격 데이터
    
    Returns:
    --------
    tuple
        (최대 낙폭, 고점 날짜, 저점 날짜, 회복 날짜)
    """
    # 누적 최대값
    cummax = prices.cummax()
    
    # Drawdown 계산
    drawdown = (prices - cummax) / cummax
    
    # Maximum Drawdown
    max_dd = drawdown.min()
    
    # 최대 낙폭 지점
    max_dd_date = drawdown.idxmin()
    
    # 고점 날짜
    peak_date = cummax[:max_dd_date].idxmax()
    
    # 회복 날짜 (고점을 회복한 날짜)
    recovery_dates = prices[max_dd_date:][prices[max_dd_date:] >= cummax[max_dd_date]]
    recovery_date = recovery_dates.index[0] if len(recovery_dates) > 0 else None
    
    return max_dd, peak_date, max_dd_date, recovery_date

def calculate_calmar_ratio(returns, prices):
    """
    Calmar Ratio를 계산합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    prices : pd.Series
        가격 데이터
    
    Returns:
    --------
    float
        Calmar Ratio
    """
    # 연율화 수익률
    annual_return = returns.mean() * 252
    
    # Maximum Drawdown
    max_dd, _, _, _ = calculate_maximum_drawdown(prices)
    
    # Calmar Ratio
    calmar = annual_return / abs(max_dd)
    
    return calmar

def calculate_all_performance_metrics(returns, prices):
    """
    모든 성과 지표를 계산합니다.
    """
    print("\n" + "=" * 60)
    print("성과 지표 계산")
    print("=" * 60)
    
    results = {}
    
    # 기본 통계
    results['Mean_Return_Daily'] = returns.mean()
    results['Mean_Return_Annual'] = returns.mean() * 252
    results['Volatility_Daily'] = returns.std()
    results['Volatility_Annual'] = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    results['Sharpe_Ratio'] = calculate_sharpe_ratio(returns)
    print(f"Sharpe Ratio:        {results['Sharpe_Ratio']:.4f}")
    
    # Sortino Ratio
    results['Sortino_Ratio'] = calculate_sortino_ratio(returns)
    print(f"Sortino Ratio:       {results['Sortino_Ratio']:.4f}")
    
    # Maximum Drawdown
    max_dd, peak_date, trough_date, recovery_date = calculate_maximum_drawdown(prices)
    results['Max_Drawdown'] = max_dd
    print(f"Maximum Drawdown:    {max_dd:.4f} ({max_dd*100:.2f}%)")
    print(f"  고점 날짜:         {peak_date.strftime('%Y-%m-%d')}")
    print(f"  저점 날짜:         {trough_date.strftime('%Y-%m-%d')}")
    if recovery_date:
        print(f"  회복 날짜:         {recovery_date.strftime('%Y-%m-%d')}")
        recovery_days = (recovery_date - peak_date).days
        print(f"  회복 기간:         {recovery_days} 일")
    else:
        print(f"  회복 날짜:         미회복")
    
    # Calmar Ratio
    results['Calmar_Ratio'] = calculate_calmar_ratio(returns, prices)
    print(f"Calmar Ratio:        {results['Calmar_Ratio']:.4f}")
    
    return results

def save_risk_metrics(var_results, performance_results, filename='risk_metrics.csv'):
    """
    리스크 지표를 CSV로 저장합니다.
    """
    all_results = {**var_results, **performance_results}
    df = pd.DataFrame([all_results])
    df.to_csv(filename, index=False)
    print(f"\n✓ 리스크 지표 저장: {filename}")

def save_performance_report(var_results, performance_results, filename='performance_metrics.txt'):
    """
    성과 리포트를 텍스트 파일로 저장합니다.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("S&P 500 리스크 및 성과 지표 리포트\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. Value at Risk (VaR)\n")
        f.write("-" * 60 + "\n")
        for key, value in var_results.items():
            f.write(f"{key:.<40} {value:.6f} ({value*100:.4f}%)\n")
        
        f.write("\n2. 성과 지표\n")
        f.write("-" * 60 + "\n")
        for key, value in performance_results.items():
            if 'Ratio' in key or 'Return_Annual' in key:
                f.write(f"{key:.<40} {value:.4f}\n")
            else:
                f.write(f"{key:.<40} {value:.6f}\n")
    
    print(f"✓ 성과 리포트 저장: {filename}")

def plot_var_analysis(returns):
    """
    VaR 분석 차트를 그립니다.
    """
    print("\n" + "=" * 60)
    print("VaR 분석 차트 생성")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 수익률 히스토그램 + VaR
    axes[0, 0].hist(returns, bins=100, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black', linewidth=0.5)
    
    # VaR 선
    var_95_hist = calculate_var_historical(returns, 0.95)
    var_99_hist = calculate_var_historical(returns, 0.99)
    
    axes[0, 0].axvline(var_95_hist, color='orange', linestyle='--', 
                      linewidth=2, label=f'VaR 95%: {var_95_hist*100:.2f}%')
    axes[0, 0].axvline(var_99_hist, color='red', linestyle='--', 
                      linewidth=2, label=f'VaR 99%: {var_99_hist*100:.2f}%')
    
    axes[0, 0].set_title('Returns Distribution with VaR', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Returns', fontsize=10)
    axes[0, 0].set_ylabel('Density', fontsize=10)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. VaR 방법론 비교
    conf_levels = [0.90, 0.95, 0.99]
    var_hist = [calculate_var_historical(returns, c) * 100 for c in conf_levels]
    var_param = [calculate_var_parametric(returns, c) * 100 for c in conf_levels]
    var_mc = [calculate_var_monte_carlo(returns, c) * 100 for c in conf_levels]
    
    x = np.arange(len(conf_levels))
    width = 0.25
    
    axes[0, 1].bar(x - width, var_hist, width, label='Historical', alpha=0.8)
    axes[0, 1].bar(x, var_param, width, label='Parametric', alpha=0.8)
    axes[0, 1].bar(x + width, var_mc, width, label='Monte Carlo', alpha=0.8)
    
    axes[0, 1].set_title('VaR Comparison by Method', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Confidence Level', fontsize=10)
    axes[0, 1].set_ylabel('VaR (%)', fontsize=10)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'{c*100:.0f}%' for c in conf_levels])
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. CVaR vs VaR
    var_values = [calculate_var_historical(returns, c) * 100 for c in conf_levels]
    cvar_values = [calculate_cvar(returns, c) * 100 for c in conf_levels]
    
    axes[1, 0].plot(conf_levels, var_values, marker='o', linewidth=2, 
                   markersize=8, label='VaR', color='blue')
    axes[1, 0].plot(conf_levels, cvar_values, marker='s', linewidth=2, 
                   markersize=8, label='CVaR', color='red')
    
    axes[1, 0].set_title('VaR vs CVaR', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Confidence Level', fontsize=10)
    axes[1, 0].set_ylabel('Risk Measure (%)', fontsize=10)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].invert_yaxis()
    
    # 4. 꼬리 위험 분석
    tail_returns = returns[returns <= var_95_hist]
    axes[1, 1].hist(tail_returns, bins=30, density=True, alpha=0.7, 
                   color='red', edgecolor='black', linewidth=0.5)
    axes[1, 1].axvline(var_95_hist, color='orange', linestyle='--', 
                      linewidth=2, label=f'VaR 95%')
    axes[1, 1].axvline(calculate_cvar(returns, 0.95), color='darkred', 
                      linestyle='--', linewidth=2, label=f'CVaR 95%')
    
    axes[1, 1].set_title('Tail Risk Analysis', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Returns', fontsize=10)
    axes[1, 1].set_ylabel('Density', fontsize=10)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('var_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ VaR 분석 차트 저장: var_analysis.png")

def plot_drawdown_analysis(prices):
    """
    Drawdown 분석 차트를 그립니다.
    """
    print("\n" + "=" * 60)
    print("Drawdown 분석 차트 생성")
    print("=" * 60)
    
    # Drawdown 계산
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. 가격과 고점
    axes[0].plot(prices.index, prices.values, linewidth=1.5, 
                color='blue', label='S&P 500 Price', alpha=0.8)
    axes[0].plot(cummax.index, cummax.values, linewidth=1.5, 
                color='red', linestyle='--', label='Peak', alpha=0.7)
    axes[0].fill_between(prices.index, prices.values, cummax.values, 
                         alpha=0.3, color='red')
    axes[0].set_title('S&P 500 Price and Drawdown', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price', fontsize=10)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Drawdown 시계열
    axes[1].fill_between(drawdown.index, drawdown.values * 100, 0, 
                         alpha=0.6, color='red', label='Drawdown')
    axes[1].plot(drawdown.index, drawdown.values * 100, 
                linewidth=1, color='darkred', alpha=0.8)
    
    # Maximum Drawdown 표시
    max_dd, peak_date, trough_date, _ = calculate_maximum_drawdown(prices)
    axes[1].axhline(max_dd * 100, color='black', linestyle=':', 
                   linewidth=2, label=f'Max DD: {max_dd*100:.2f}%')
    axes[1].plot(trough_date, max_dd * 100, 'ro', markersize=10, 
                label='Max DD Point')
    
    axes[1].set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=10)
    axes[1].set_ylabel('Drawdown (%)', fontsize=10)
    axes[1].legend(loc='lower left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drawdown_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Drawdown 분석 차트 저장: drawdown_analysis.png")

def plot_risk_return_profile(returns, prices):
    """
    리스크-수익 프로파일 차트를 그립니다.
    """
    print("\n" + "=" * 60)
    print("리스크-수익 프로파일 차트 생성")
    print("=" * 60)
    
    # 롤링 통계
    windows = [60, 126, 252]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 롤링 Sharpe Ratio
    for window in windows:
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        axes[0, 0].plot(rolling_sharpe.index, rolling_sharpe.values, 
                       linewidth=1.5, label=f'{window}-day', alpha=0.7)
    
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 0].set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Sharpe Ratio', fontsize=10)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 연율화 수익률 vs 변동성
    rolling_return_252 = returns.rolling(252).mean() * 252
    rolling_vol_252 = returns.rolling(252).std() * np.sqrt(252)
    
    axes[0, 1].scatter(rolling_vol_252, rolling_return_252, 
                      alpha=0.5, s=10, c=range(len(rolling_vol_252)), cmap='viridis')
    axes[0, 1].set_title('Risk-Return Profile (252-day Rolling)', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Volatility (Annual)', fontsize=10)
    axes[0, 1].set_ylabel('Return (Annual)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 롤링 최대 낙폭
    rolling_max = prices.rolling(252, min_periods=1).max()
    rolling_dd = (prices - rolling_max) / rolling_max
    
    axes[1, 0].fill_between(rolling_dd.index, rolling_dd.values * 100, 0, 
                           alpha=0.6, color='red')
    axes[1, 0].plot(rolling_dd.index, rolling_dd.values * 100, 
                   linewidth=1, color='darkred', alpha=0.8)
    axes[1, 0].set_title('Rolling Drawdown (252-day)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Date', fontsize=10)
    axes[1, 0].set_ylabel('Drawdown (%)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 성과 지표 비교 (바 차트)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    calmar = calculate_calmar_ratio(returns, prices)
    
    metrics = ['Sharpe\nRatio', 'Sortino\nRatio', 'Calmar\nRatio']
    values = [sharpe, sortino, calmar]
    colors = ['blue', 'green', 'orange']
    
    axes[1, 1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Performance Metrics Comparison', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Ratio', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('risk_return_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 리스크-수익 프로파일 차트 저장: risk_return_profile.png")

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 60)
    print("리스크 측정 지표 계산 프로그램")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        data = load_data()
        returns = data['Log_Return'].dropna()
        prices = data['Close']
        
        # 2. VaR 계산
        var_results = calculate_all_var_metrics(returns)
        
        # 3. 성과 지표 계산
        performance_results = calculate_all_performance_metrics(returns, prices)
        
        # 4. 결과 저장
        print("\n" + "=" * 60)
        print("데이터 저장")
        print("=" * 60)
        save_risk_metrics(var_results, performance_results)
        save_performance_report(var_results, performance_results)
        
        # 5. 시각화
        plot_var_analysis(returns)
        plot_drawdown_analysis(prices)
        plot_risk_return_profile(returns, prices)
        
        print("\n" + "=" * 60)
        print("리스크 측정 완료!")
        print("=" * 60)
        print("\n생성된 파일:")
        print("  - risk_metrics.csv")
        print("  - performance_metrics.txt")
        print("  - var_analysis.png")
        print("  - drawdown_analysis.png")
        print("  - risk_return_profile.png")
        
        print("\n다음 단계:")
        print("  python portfolio_analysis.py")
        
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
