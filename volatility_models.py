"""
변동성 모델 분석 스크립트
다양한 방법으로 변동성을 계산하고 비교합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def calculate_historical_volatility(returns, window=252):
    """
    히스토리컬 변동성을 계산합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    window : int
        계산 윈도우 (기본값: 252 거래일 = 1년)
    
    Returns:
    --------
    float
        연율화 변동성
    """
    volatility = returns.std() * np.sqrt(252)
    return volatility

def calculate_rolling_volatility(returns, windows=[20, 60, 252]):
    """
    이동평균 변동성을 계산합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    windows : list
        윈도우 크기 리스트
    
    Returns:
    --------
    pd.DataFrame
        이동평균 변동성 데이터프레임
    """
    print("\n" + "=" * 60)
    print("이동평균 변동성 계산")
    print("=" * 60)
    
    volatility_df = pd.DataFrame(index=returns.index)
    
    for window in windows:
        col_name = f'Volatility_{window}d'
        # 이동 표준편차를 연율화
        volatility_df[col_name] = returns.rolling(window=window).std() * np.sqrt(252)
        print(f"✓ {window}일 이동평균 변동성 계산 완료")
    
    return volatility_df

def calculate_ewma_volatility(returns, lambda_param=0.94):
    """
    지수가중이동평균(EWMA) 변동성을 계산합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    lambda_param : float
        감쇠 파라미터 (기본값: 0.94, RiskMetrics 권장값)
    
    Returns:
    --------
    pd.Series
        EWMA 변동성
    """
    print("\n" + "=" * 60)
    print("EWMA 변동성 계산")
    print("=" * 60)
    print(f"Lambda 파라미터: {lambda_param}")
    
    # 제곱 수익률
    squared_returns = returns ** 2
    
    # EWMA 분산
    ewma_variance = squared_returns.ewm(alpha=1-lambda_param, adjust=False).mean()
    
    # EWMA 변동성 (연율화)
    ewma_volatility = np.sqrt(ewma_variance) * np.sqrt(252)
    
    print("✓ EWMA 변동성 계산 완료")
    
    return ewma_volatility

def calculate_parkinson_volatility(data, window=20):
    """
    Parkinson 변동성을 계산합니다.
    고가-저가 정보를 활용한 변동성 추정
    
    Parameters:
    -----------
    data : pd.DataFrame
        가격 데이터 (High, Low 컬럼 필요)
    window : int
        계산 윈도우
    
    Returns:
    --------
    pd.Series
        Parkinson 변동성
    """
    print("\n" + "=" * 60)
    print("Parkinson 변동성 계산")
    print("=" * 60)
    
    hl_ratio = np.log(data['High'] / data['Low'])
    parkinson_vol = np.sqrt(hl_ratio ** 2 / (4 * np.log(2)))
    
    # 이동평균
    rolling_parkinson = parkinson_vol.rolling(window=window).mean() * np.sqrt(252)
    
    print(f"✓ Parkinson 변동성 계산 완료 (윈도우: {window}일)")
    
    return rolling_parkinson

def print_volatility_summary(returns, volatility_measures):
    """
    변동성 요약 통계를 출력합니다.
    """
    print("\n" + "=" * 60)
    print("변동성 측정값 요약")
    print("=" * 60)
    
    print(f"\n전체 기간 히스토리컬 변동성: {calculate_historical_volatility(returns):.4f}")
    print(f"  (연율화 값, 약 {calculate_historical_volatility(returns)*100:.2f}%)")
    
    print("\n최근 변동성 (마지막 관측값):")
    for col in volatility_measures.columns:
        last_value = volatility_measures[col].iloc[-1]
        if not np.isnan(last_value):
            print(f"  {col:.<40} {last_value:.4f} ({last_value*100:.2f}%)")

def plot_volatility_comparison(returns, volatility_measures, ewma_vol):
    """
    다양한 변동성 측정값을 비교하는 차트를 그립니다.
    """
    print("\n" + "=" * 60)
    print("변동성 비교 차트 생성")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 수익률 차트
    axes[0].plot(returns.index, returns.values, linewidth=0.8, color='blue', alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_title('Daily Log Returns', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Returns', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. 이동평균 변동성
    for col in volatility_measures.columns:
        axes[1].plot(volatility_measures.index, volatility_measures[col], 
                    linewidth=1.5, label=col, alpha=0.8)
    axes[1].set_title('Rolling Volatility (Annualized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volatility', fontsize=10)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # 3. EWMA 변동성
    axes[2].plot(ewma_vol.index, ewma_vol.values, linewidth=1.5, 
                color='darkred', label='EWMA Volatility (λ=0.94)', alpha=0.8)
    axes[2].plot(volatility_measures.index, volatility_measures['Volatility_60d'], 
                linewidth=1.5, color='blue', label='60-day Rolling Vol', alpha=0.6, linestyle='--')
    axes[2].set_title('EWMA vs Rolling Volatility', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].set_ylabel('Volatility', fontsize=10)
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('volatility_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 변동성 비교 차트 저장: volatility_comparison.png")

def plot_rolling_volatility_detailed(volatility_measures):
    """
    이동평균 변동성 상세 차트를 그립니다.
    """
    print("\n" + "=" * 60)
    print("이동평균 변동성 상세 차트 생성")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['red', 'blue', 'green']
    
    for i, col in enumerate(volatility_measures.columns):
        ax.plot(volatility_measures.index, volatility_measures[col], 
               linewidth=1.5, label=col, color=colors[i], alpha=0.7)
    
    # 평균선
    for i, col in enumerate(volatility_measures.columns):
        mean_val = volatility_measures[col].mean()
        ax.axhline(y=mean_val, color=colors[i], linestyle=':', linewidth=1.5, 
                  alpha=0.5, label=f'{col} Mean: {mean_val:.4f}')
    
    ax.set_title('Rolling Volatility Analysis (Annualized)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volatility', fontsize=12)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rolling_volatility.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 이동평균 변동성 차트 저장: rolling_volatility.png")

def plot_volatility_clustering(returns):
    """
    변동성 클러스터링을 시각화합니다.
    """
    print("\n" + "=" * 60)
    print("변동성 클러스터링 차트 생성")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. 수익률 절대값
    axes[0].plot(returns.index, np.abs(returns.values), 
                linewidth=0.8, color='purple', alpha=0.6)
    axes[0].set_title('Absolute Returns (Volatility Proxy)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('|Returns|', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. 제곱 수익률 (변동성 대리변수)
    squared_returns = returns ** 2
    axes[1].plot(squared_returns.index, squared_returns.values, 
                linewidth=0.8, color='darkred', alpha=0.6)
    axes[1].set_title('Squared Returns (Volatility Clustering)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=10)
    axes[1].set_ylabel('Returns²', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('volatility_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 변동성 클러스터링 차트 저장: volatility_clustering.png")

def save_volatility_measures(volatility_measures, ewma_vol, filename='volatility_measures.csv'):
    """
    변동성 측정값을 CSV로 저장합니다.
    """
    # 모든 변동성 측정값 결합
    all_volatility = volatility_measures.copy()
    all_volatility['EWMA_Volatility'] = ewma_vol
    
    all_volatility.to_csv(filename)
    print(f"✓ 변동성 측정값 저장: {filename}")

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 60)
    print("변동성 모델 분석 프로그램")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        data = load_data()
        returns = data['Log_Return'].dropna()
        
        # 2. 히스토리컬 변동성
        hist_vol = calculate_historical_volatility(returns)
        print(f"\n전체 기간 히스토리컬 변동성: {hist_vol:.4f} ({hist_vol*100:.2f}%)")
        
        # 3. 이동평균 변동성
        rolling_vol = calculate_rolling_volatility(returns, windows=[20, 60, 252])
        
        # 4. EWMA 변동성
        ewma_vol = calculate_ewma_volatility(returns, lambda_param=0.94)
        
        # 5. Parkinson 변동성
        parkinson_vol = calculate_parkinson_volatility(data, window=20)
        
        # 6. 요약 통계
        print_volatility_summary(returns, rolling_vol)
        
        # 7. 시각화
        plot_volatility_comparison(returns, rolling_vol, ewma_vol)
        plot_rolling_volatility_detailed(rolling_vol)
        plot_volatility_clustering(returns)
        
        # 8. 데이터 저장
        print("\n" + "=" * 60)
        print("데이터 저장")
        print("=" * 60)
        save_volatility_measures(rolling_vol, ewma_vol)
        
        print("\n" + "=" * 60)
        print("변동성 분석 완료!")
        print("=" * 60)
        print("\n생성된 파일:")
        print("  - volatility_measures.csv")
        print("  - volatility_comparison.png")
        print("  - rolling_volatility.png")
        print("  - volatility_clustering.png")
        
        print("\n다음 단계:")
        print("  python garch_analysis.py")
        
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
