"""
GARCH 모델 분석 스크립트
GARCH(1,1) 모델을 추정하고 변동성을 예측합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
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

def estimate_garch_model(returns, p=1, q=1):
    """
    GARCH 모델을 추정합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    p : int
        GARCH 차수
    q : int
        ARCH 차수
    
    Returns:
    --------
    model_result
        추정된 GARCH 모델
    """
    print("\n" + "=" * 60)
    print(f"GARCH({p},{q}) 모델 추정")
    print("=" * 60)
    
    # 수익률을 퍼센트로 변환 (수치 안정성을 위해)
    returns_percent = returns * 100
    
    # GARCH 모델 정의
    model = arch_model(returns_percent, vol='Garch', p=p, q=q, dist='normal')
    
    # 모델 추정
    print("모델 추정 중...")
    model_result = model.fit(disp='off', show_warning=False)
    
    print("✓ 모델 추정 완료")
    
    return model_result

def print_garch_results(model_result):
    """
    GARCH 모델 결과를 출력합니다.
    """
    print("\n" + "=" * 60)
    print("GARCH 모델 추정 결과")
    print("=" * 60)
    
    print("\n" + str(model_result.summary()))
    
    # 파라미터 추출
    params = model_result.params
    
    print("\n" + "=" * 60)
    print("모델 파라미터")
    print("=" * 60)
    
    print(f"\nω (omega):     {params['omega']:.6f}")
    print(f"α (alpha[1]):  {params['alpha[1]']:.6f}")
    print(f"β (beta[1]):   {params['beta[1]']:.6f}")
    
    # 변동성 지속성
    persistence = params['alpha[1]'] + params['beta[1]']
    print(f"\n변동성 지속성 (α + β): {persistence:.6f}")
    
    if persistence < 1:
        print("✓ 정상성 조건 만족 (α + β < 1)")
    else:
        print("✗ 정상성 조건 불만족 (α + β >= 1)")
    
    # 장기 평균 분산
    long_run_var = params['omega'] / (1 - persistence)
    long_run_vol = np.sqrt(long_run_var)
    print(f"\n장기 평균 변동성: {long_run_vol:.4f}% (일별)")
    print(f"장기 평균 변동성: {long_run_vol * np.sqrt(252):.4f}% (연율화)")

def save_garch_results(model_result, filename='garch_results.txt'):
    """
    GARCH 모델 결과를 텍스트 파일로 저장합니다.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("GARCH(1,1) 모델 추정 결과\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(model_result.summary()))
        
        params = model_result.params
        persistence = params['alpha[1]'] + params['beta[1]']
        long_run_var = params['omega'] / (1 - persistence)
        long_run_vol = np.sqrt(long_run_var)
        
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("주요 지표\n")
        f.write("=" * 60 + "\n")
        f.write(f"\n변동성 지속성 (α + β): {persistence:.6f}\n")
        f.write(f"장기 평균 변동성 (연율화): {long_run_vol * np.sqrt(252):.4f}%\n")
    
    print(f"✓ GARCH 결과 저장: {filename}")

def forecast_volatility(model_result, horizon=30):
    """
    GARCH 모델로 변동성을 예측합니다.
    
    Parameters:
    -----------
    model_result : ARCHModelResult
        추정된 GARCH 모델
    horizon : int
        예측 기간
    
    Returns:
    --------
    pd.DataFrame
        예측 결과
    """
    print("\n" + "=" * 60)
    print(f"변동성 예측 ({horizon}일)")
    print("=" * 60)
    
    # 변동성 예측
    forecasts = model_result.forecast(horizon=horizon)
    forecast_variance = forecasts.variance.values[-1, :]
    forecast_volatility = np.sqrt(forecast_variance)
    
    print(f"✓ {horizon}일 변동성 예측 완료")
    
    return forecast_volatility

def plot_conditional_volatility(model_result, returns):
    """
    조건부 변동성을 시각화합니다.
    """
    print("\n" + "=" * 60)
    print("조건부 변동성 차트 생성")
    print("=" * 60)
    
    # 조건부 변동성 추출
    conditional_vol = model_result.conditional_volatility
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 수익률
    axes[0].plot(returns.index, returns.values * 100, 
                linewidth=0.8, color='blue', alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_title('Daily Log Returns', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Returns (%)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. 조건부 변동성
    axes[1].plot(conditional_vol.index, conditional_vol.values, 
                linewidth=1.5, color='darkred', alpha=0.8)
    axes[1].set_title('Conditional Volatility (GARCH)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volatility (%)', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 3. 수익률과 조건부 변동성 밴드
    axes[2].plot(returns.index, returns.values * 100, 
                linewidth=0.8, color='blue', alpha=0.6, label='Returns')
    axes[2].fill_between(conditional_vol.index, 
                         conditional_vol.values, 
                         -conditional_vol.values, 
                         alpha=0.2, color='red', label='±1 Std Dev')
    axes[2].fill_between(conditional_vol.index, 
                         2*conditional_vol.values, 
                         -2*conditional_vol.values, 
                         alpha=0.1, color='orange', label='±2 Std Dev')
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[2].set_title('Returns with Volatility Bands', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].set_ylabel('Returns (%)', fontsize=10)
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('garch_conditional_volatility.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 조건부 변동성 차트 저장: garch_conditional_volatility.png")

def plot_forecast(model_result, returns, horizon=60):
    """
    변동성 예측을 시각화합니다.
    """
    print("\n" + "=" * 60)
    print("변동성 예측 차트 생성")
    print("=" * 60)
    
    # 과거 조건부 변동성
    conditional_vol = model_result.conditional_volatility
    
    # 미래 변동성 예측
    forecast_vol = forecast_volatility(model_result, horizon=horizon)
    
    # 날짜 생성 (예측 기간)
    last_date = conditional_vol.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=horizon+1, freq='D')[1:]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 과거 조건부 변동성 (최근 1년)
    recent_vol = conditional_vol.iloc[-252:]
    ax.plot(recent_vol.index, recent_vol.values, 
           linewidth=1.5, color='blue', label='Historical Conditional Volatility', alpha=0.8)
    
    # 예측 변동성
    ax.plot(forecast_dates, forecast_vol, 
           linewidth=2, color='red', label=f'{horizon}-day Forecast', alpha=0.8, linestyle='--')
    
    # 장기 평균
    params = model_result.params
    persistence = params['alpha[1]'] + params['beta[1]']
    long_run_var = params['omega'] / (1 - persistence)
    long_run_vol = np.sqrt(long_run_var)
    
    ax.axhline(y=long_run_vol, color='green', linestyle=':', 
              linewidth=2, label=f'Long-run Mean: {long_run_vol:.2f}%', alpha=0.7)
    
    ax.set_title('GARCH Volatility Forecast', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volatility (%)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('garch_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 변동성 예측 차트 저장: garch_forecast.png")

def plot_residual_diagnostics(model_result):
    """
    잔차 진단 차트를 그립니다.
    """
    print("\n" + "=" * 60)
    print("잔차 진단 차트 생성")
    print("=" * 60)
    
    # 표준화 잔차
    std_resid = model_result.std_resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 표준화 잔차 시계열
    axes[0, 0].plot(std_resid.index, std_resid.values, 
                   linewidth=0.8, color='blue', alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 0].set_title('Standardized Residuals', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Std. Residuals', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 잔차 히스토그램
    axes[0, 1].hist(std_resid.dropna(), bins=50, density=True, 
                   alpha=0.7, color='skyblue', edgecolor='black')
    x = np.linspace(std_resid.min(), std_resid.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
    axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Std. Residuals', fontsize=10)
    axes[0, 1].set_ylabel('Density', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    stats.probplot(std_resid.dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ACF of Squared Residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(std_resid.dropna()**2, lags=40, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title('ACF of Squared Std. Residuals', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Lag', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('garch_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 잔차 진단 차트 저장: garch_diagnostics.png")

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 60)
    print("GARCH 모델 분석 프로그램")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        data = load_data()
        returns = data['Log_Return'].dropna()
        
        # 2. GARCH(1,1) 모델 추정
        garch_result = estimate_garch_model(returns, p=1, q=1)
        
        # 3. 결과 출력 및 저장
        print_garch_results(garch_result)
        save_garch_results(garch_result)
        
        # 4. 시각화
        plot_conditional_volatility(garch_result, returns)
        plot_forecast(garch_result, returns, horizon=60)
        plot_residual_diagnostics(garch_result)
        
        print("\n" + "=" * 60)
        print("GARCH 분석 완료!")
        print("=" * 60)
        print("\n생성된 파일:")
        print("  - garch_results.txt")
        print("  - garch_conditional_volatility.png")
        print("  - garch_forecast.png")
        print("  - garch_diagnostics.png")
        
        print("\n다음 단계:")
        print("  cd ../part3_risk_measurement")
        print("  python risk_metrics.py")
        
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
