"""
S&P 500 기초 통계 분석 스크립트
수익률 분석 및 시각화를 수행합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data(filename='sp500_returns.csv'):
    """
    저장된 데이터를 로드합니다.
    
    Parameters:
    -----------
    filename : str
        파일명
    
    Returns:
    --------
    pd.DataFrame
        데이터프레임
    """
    print("=" * 60)
    print("데이터 로드")
    print("=" * 60)
    
    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"✓ 데이터 로드 완료: {filename}")
        print(f"✓ 데이터 크기: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {filename}")
        print("먼저 data_collector.py를 실행하세요.")
        raise

def calculate_statistics(returns):
    """
    기초 통계량을 계산합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    
    Returns:
    --------
    dict
        통계량 딕셔너리
    """
    print("\n" + "=" * 60)
    print("기초 통계량 계산")
    print("=" * 60)
    
    stats_dict = {
        '평균 (Mean)': returns.mean(),
        '표준편차 (Std Dev)': returns.std(),
        '분산 (Variance)': returns.var(),
        '왜도 (Skewness)': stats.skew(returns.dropna()),
        '첨도 (Kurtosis)': stats.kurtosis(returns.dropna()),
        '최소값 (Min)': returns.min(),
        '최대값 (Max)': returns.max(),
        '25% 분위수': returns.quantile(0.25),
        '중앙값 (Median)': returns.median(),
        '75% 분위수': returns.quantile(0.75),
    }
    
    # 연율화 통계량 (252 거래일 가정)
    stats_dict['연율화 수익률 (Ann. Return)'] = returns.mean() * 252
    stats_dict['연율화 변동성 (Ann. Volatility)'] = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (무위험 이자율 0% 가정)
    stats_dict['Sharpe Ratio'] = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    print("✓ 통계량 계산 완료")
    
    return stats_dict

def print_statistics(stats_dict):
    """
    통계량을 출력합니다.
    
    Parameters:
    -----------
    stats_dict : dict
        통계량 딕셔너리
    """
    print("\n" + "=" * 60)
    print("기초 통계량 결과")
    print("=" * 60)
    
    for key, value in stats_dict.items():
        if 'Ratio' in key or '수익률' in key:
            print(f"{key:.<40} {value:.4f}")
        else:
            print(f"{key:.<40} {value:.6f}")

def save_statistics(stats_dict, filename='basic_statistics.txt'):
    """
    통계량을 텍스트 파일로 저장합니다.
    
    Parameters:
    -----------
    stats_dict : dict
        통계량 딕셔너리
    filename : str
        파일명
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("S&P 500 기초 통계량 리포트\n")
        f.write("=" * 60 + "\n\n")
        
        for key, value in stats_dict.items():
            if 'Ratio' in key or '수익률' in key:
                f.write(f"{key:.<40} {value:.4f}\n")
            else:
                f.write(f"{key:.<40} {value:.6f}\n")
    
    print(f"✓ 통계량 저장 완료: {filename}")

def plot_price_chart(data):
    """
    가격 차트를 그립니다.
    
    Parameters:
    -----------
    data : pd.DataFrame
        데이터프레임
    """
    print("\n" + "=" * 60)
    print("가격 차트 생성")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 종가 차트
    axes[0].plot(data.index, data['Close'], linewidth=1.5, color='blue', alpha=0.8)
    axes[0].set_title('S&P 500 Index Price', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Price', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(['Close Price'], loc='upper left')
    
    # 거래량 차트
    axes[1].bar(data.index, data['Volume'], width=1, color='gray', alpha=0.6)
    axes[1].set_title('Trading Volume', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Volume', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('price_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 가격 차트 저장: price_chart.png")

def plot_returns_distribution(returns):
    """
    수익률 분포를 시각화합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    """
    print("\n" + "=" * 60)
    print("수익률 분포 차트 생성")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 시계열 차트
    axes[0, 0].plot(returns.index, returns.values, linewidth=0.8, color='blue', alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 0].set_title('Daily Log Returns Time Series', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date', fontsize=10)
    axes[0, 0].set_ylabel('Log Returns', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 히스토그램 + 정규분포
    axes[0, 1].hist(returns.dropna(), bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 정규분포 곡선
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
    axes[0, 1].set_title('Returns Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Log Returns', fontsize=10)
    axes[0, 1].set_ylabel('Density', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    stats.probplot(returns.dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 박스플롯
    axes[1, 1].boxplot(returns.dropna(), vert=True)
    axes[1, 1].set_title('Returns Box Plot', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Log Returns', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('returns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 수익률 분포 차트 저장: returns_distribution.png")

def plot_returns_histogram(returns):
    """
    수익률 히스토그램을 그립니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    """
    print("\n" + "=" * 60)
    print("수익률 히스토그램 생성")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 히스토그램
    n, bins, patches = ax.hist(returns.dropna(), bins=100, density=True, 
                                alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # 통계량 표시
    mu = returns.mean()
    sigma = returns.std()
    skewness = stats.skew(returns.dropna())
    kurt = stats.kurtosis(returns.dropna())
    
    # 정규분포 곡선
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2.5, label='Normal Distribution')
    
    # 평균선
    ax.axvline(mu, color='green', linestyle='--', linewidth=2, label=f'Mean: {mu:.6f}')
    
    # 텍스트 박스
    textstr = f'Mean: {mu:.6f}\nStd Dev: {sigma:.6f}\nSkewness: {skewness:.4f}\nKurtosis: {kurt:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.set_title('S&P 500 Daily Log Returns Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Log Returns', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('returns_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 히스토그램 저장: returns_histogram.png")

def perform_normality_tests(returns):
    """
    정규성 검정을 수행합니다.
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    """
    print("\n" + "=" * 60)
    print("정규성 검정")
    print("=" * 60)
    
    # Jarque-Bera 검정
    jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
    print(f"\nJarque-Bera Test:")
    print(f"  통계량: {jb_stat:.4f}")
    print(f"  p-value: {jb_pvalue:.6f}")
    print(f"  결과: {'정규분포가 아님' if jb_pvalue < 0.05 else '정규분포일 가능성'} (α=0.05)")
    
    # Shapiro-Wilk 검정
    sw_stat, sw_pvalue = stats.shapiro(returns.dropna()[:5000])  # 샘플 크기 제한
    print(f"\nShapiro-Wilk Test:")
    print(f"  통계량: {sw_stat:.4f}")
    print(f"  p-value: {sw_pvalue:.6f}")
    print(f"  결과: {'정규분포가 아님' if sw_pvalue < 0.05 else '정규분포일 가능성'} (α=0.05)")

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 60)
    print("S&P 500 기초 통계 분석 프로그램")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        data = load_data('sp500_returns.csv')
        
        # 2. 로그 수익률 추출
        returns = data['Log_Return']
        
        # 3. 기초 통계량 계산
        stats_dict = calculate_statistics(returns)
        print_statistics(stats_dict)
        save_statistics(stats_dict)
        
        # 4. 정규성 검정
        perform_normality_tests(returns)
        
        # 5. 시각화
        plot_price_chart(data)
        plot_returns_distribution(returns)
        plot_returns_histogram(returns)
        
        print("\n" + "=" * 60)
        print("분석 완료!")
        print("=" * 60)
        print("\n생성된 파일:")
        print("  - basic_statistics.txt")
        print("  - price_chart.png")
        print("  - returns_distribution.png")
        print("  - returns_histogram.png")
        
        print("\n다음 단계:")
        print("  cd ../part2_volatility_analysis")
        print("  python volatility_models.py")
        
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
