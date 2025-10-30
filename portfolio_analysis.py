"""
포트폴리오 분석 스크립트
효율적 투자선, 리스크 기여도, 시나리오 분석을 수행합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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

def portfolio_statistics(weights, returns):
    """
    포트폴리오 통계를 계산합니다.
    
    Parameters:
    -----------
    weights : np.array
        자산 가중치
    returns : pd.DataFrame
        수익률 데이터
    
    Returns:
    --------
    tuple
        (포트폴리오 수익률, 포트폴리오 변동성)
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_std

def simulate_stress_scenarios(returns):
    """
    스트레스 시나리오를 시뮬레이션합니다.
    """
    print("\n" + "=" * 60)
    print("스트레스 시나리오 분석")
    print("=" * 60)
    
    # 시나리오 정의
    scenarios = {
        '정상 시장': {'mean_shift': 0, 'vol_multiplier': 1.0},
        '경미한 조정': {'mean_shift': -0.02, 'vol_multiplier': 1.5},
        '심각한 하락': {'mean_shift': -0.05, 'vol_multiplier': 2.0},
        '극단적 위기': {'mean_shift': -0.10, 'vol_multiplier': 3.0},
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        # 시나리오 적용
        adjusted_returns = returns + params['mean_shift']
        adjusted_vol = returns.std() * params['vol_multiplier']
        
        # 시뮬레이션
        simulated = np.random.normal(
            adjusted_returns.mean(),
            adjusted_vol,
            len(returns)
        )
        
        # VaR 계산
        var_95 = np.percentile(simulated, 5)
        cvar_95 = simulated[simulated <= var_95].mean()
        
        results[scenario_name] = {
            'Mean': adjusted_returns.mean() * 252,
            'Volatility': adjusted_vol * np.sqrt(252),
            'VaR_95': var_95,
            'CVaR_95': cvar_95
        }
        
        print(f"\n{scenario_name}:")
        print(f"  연율화 수익률:    {adjusted_returns.mean() * 252:.4f}")
        print(f"  연율화 변동성:    {adjusted_vol * np.sqrt(252):.4f}")
        print(f"  VaR 95%:         {var_95:.6f} ({var_95*100:.4f}%)")
        print(f"  CVaR 95%:        {cvar_95:.6f} ({cvar_95*100:.4f}%)")
    
    return results

def plot_stress_scenarios(scenario_results):
    """
    스트레스 시나리오 결과를 시각화합니다.
    """
    print("\n" + "=" * 60)
    print("스트레스 시나리오 차트 생성")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = list(scenario_results.keys())
    
    # 1. 수익률 비교
    returns_data = [scenario_results[s]['Mean'] * 100 for s in scenarios]
    axes[0, 0].barh(scenarios, returns_data, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_title('Expected Annual Returns by Scenario', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Return (%)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. 변동성 비교
    vol_data = [scenario_results[s]['Volatility'] * 100 for s in scenarios]
    axes[0, 1].barh(scenarios, vol_data, color='orange', edgecolor='black')
    axes[0, 1].set_title('Expected Annual Volatility by Scenario', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Volatility (%)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. VaR 비교
    var_data = [scenario_results[s]['VaR_95'] * 100 for s in scenarios]
    axes[1, 0].barh(scenarios, var_data, color='red', edgecolor='black')
    axes[1, 0].set_title('VaR 95% by Scenario', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('VaR (%)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. CVaR 비교
    cvar_data = [scenario_results[s]['CVaR_95'] * 100 for s in scenarios]
    axes[1, 1].barh(scenarios, cvar_data, color='darkred', edgecolor='black')
    axes[1, 1].set_title('CVaR 95% by Scenario', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('CVaR (%)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('stress_scenarios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 스트레스 시나리오 차트 저장: stress_scenarios.png")

def analyze_return_distribution(returns):
    """
    수익률 분포를 심층 분석합니다.
    """
    print("\n" + "=" * 60)
    print("수익률 분포 심층 분석")
    print("=" * 60)
    
    from scipy import stats
    
    # 분위수 분석
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    
    print("\n분위수 분석:")
    for q in quantiles:
        value = returns.quantile(q)
        print(f"  {q*100:>5.1f}%: {value:.6f} ({value*100:.4f}%)")
    
    # 극단값 분석
    threshold_lower = returns.quantile(0.05)
    threshold_upper = returns.quantile(0.95)
    
    extreme_losses = returns[returns <= threshold_lower]
    extreme_gains = returns[returns >= threshold_upper]
    
    print(f"\n극단 손실 (하위 5%):")
    print(f"  발생 횟수: {len(extreme_losses)}")
    print(f"  평균 손실: {extreme_losses.mean():.6f} ({extreme_losses.mean()*100:.4f}%)")
    print(f"  최대 손실: {extreme_losses.min():.6f} ({extreme_losses.min()*100:.4f}%)")
    
    print(f"\n극단 이익 (상위 5%):")
    print(f"  발생 횟수: {len(extreme_gains)}")
    print(f"  평균 이익: {extreme_gains.mean():.6f} ({extreme_gains.mean()*100:.4f}%)")
    print(f"  최대 이익: {extreme_gains.max():.6f} ({extreme_gains.max()*100:.4f}%)")

def plot_distribution_analysis(returns):
    """
    분포 분석 차트를 그립니다.
    """
    print("\n" + "=" * 60)
    print("분포 분석 차트 생성")
    print("=" * 60)
    
    from scipy import stats
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 로그 스케일 히스토그램
    axes[0, 0].hist(returns, bins=100, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black', linewidth=0.5)
    
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
    
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Returns Distribution (Log Scale)', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Returns', fontsize=10)
    axes[0, 0].set_ylabel('Log Density', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 누적 분포 함수
    sorted_returns = np.sort(returns)
    cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    
    axes[0, 1].plot(sorted_returns, cdf, linewidth=2, label='Empirical CDF')
    axes[0, 1].plot(x, stats.norm.cdf(x, mu, sigma), 'r--', 
                   linewidth=2, label='Normal CDF')
    
    axes[0, 1].set_title('Cumulative Distribution Function', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Returns', fontsize=10)
    axes[0, 1].set_ylabel('Cumulative Probability', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 왼쪽 꼬리 분석
    threshold = returns.quantile(0.05)
    left_tail = returns[returns <= threshold]
    
    axes[1, 0].hist(left_tail, bins=30, density=True, alpha=0.7, 
                   color='red', edgecolor='black', linewidth=0.5)
    axes[1, 0].axvline(threshold, color='darkred', linestyle='--', 
                      linewidth=2, label=f'5% Quantile')
    axes[1, 0].set_title('Left Tail Analysis (Bottom 5%)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Returns', fontsize=10)
    axes[1, 0].set_ylabel('Density', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 오른쪽 꼬리 분석
    threshold = returns.quantile(0.95)
    right_tail = returns[returns >= threshold]
    
    axes[1, 1].hist(right_tail, bins=30, density=True, alpha=0.7, 
                   color='green', edgecolor='black', linewidth=0.5)
    axes[1, 1].axvline(threshold, color='darkgreen', linestyle='--', 
                      linewidth=2, label=f'95% Quantile')
    axes[1, 1].set_title('Right Tail Analysis (Top 5%)', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Returns', fontsize=10)
    axes[1, 1].set_ylabel('Density', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 분포 분석 차트 저장: distribution_analysis.png")

def create_summary_dashboard(returns, prices):
    """
    전체 요약 대시보드를 생성합니다.
    """
    print("\n" + "=" * 60)
    print("요약 대시보드 생성")
    print("=" * 60)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 가격 차트
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(prices.index, prices.values, linewidth=1.5, color='blue', alpha=0.8)
    ax1.set_title('S&P 500 Index Price', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 수익률
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(returns.index, returns.values * 100, linewidth=0.8, 
            color='blue', alpha=0.6)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_title('Daily Log Returns', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Returns (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 수익률 분포
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(returns, bins=50, density=True, alpha=0.7, 
            color='skyblue', edgecolor='black')
    ax3.set_title('Returns Distribution', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Returns', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 롤링 변동성
    ax4 = fig.add_subplot(gs[2, 1])
    rolling_vol = returns.rolling(60).std() * np.sqrt(252) * 100
    ax4.plot(rolling_vol.index, rolling_vol.values, linewidth=1.5, color='red')
    ax4.set_title('60-day Rolling Volatility', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=9)
    ax4.set_ylabel('Volatility (%)', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. 주요 지표
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    # 통계 계산
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol
    
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    max_dd = drawdown.min()
    
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # 텍스트 박스
    stats_text = f"""
    Key Metrics
    {'='*30}
    
    Annual Return:   {annual_return*100:>7.2f}%
    Annual Vol:      {annual_vol*100:>7.2f}%
    Sharpe Ratio:    {sharpe:>7.2f}
    
    Max Drawdown:    {max_dd*100:>7.2f}%
    
    VaR (95%):       {var_95*100:>7.2f}%
    CVaR (95%):      {cvar_95*100:>7.2f}%
    """
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig('summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 요약 대시보드 저장: summary_dashboard.png")

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 60)
    print("포트폴리오 분석 프로그램")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        data = load_data()
        returns = data['Log_Return'].dropna()
        prices = data['Close']
        
        # 2. 스트레스 시나리오 분석
        scenario_results = simulate_stress_scenarios(returns)
        
        # 3. 수익률 분포 분석
        analyze_return_distribution(returns)
        
        # 4. 시각화
        plot_stress_scenarios(scenario_results)
        plot_distribution_analysis(returns)
        create_summary_dashboard(returns, prices)
        
        print("\n" + "=" * 60)
        print("포트폴리오 분석 완료!")
        print("=" * 60)
        print("\n생성된 파일:")
        print("  - stress_scenarios.png")
        print("  - distribution_analysis.png")
        print("  - summary_dashboard.png")
        
        print("\n" + "=" * 60)
        print("전체 프로젝트 완료!")
        print("=" * 60)
        print("\n모든 분석이 완료되었습니다.")
        print("각 Part의 README.md를 참조하여 결과를 확인하세요.")
        
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
