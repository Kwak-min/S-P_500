"""
S&P 500 데이터 수집 스크립트
Yahoo Finance API를 사용하여 S&P 500 지수 데이터를 다운로드합니다.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def download_sp500_data(period='5y', interval='1d'):
    """
    S&P 500 지수 데이터를 다운로드합니다.
    
    Parameters:
    -----------
    period : str
        데이터 기간 ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    interval : str
        데이터 간격 ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
    --------
    pd.DataFrame
        S&P 500 데이터프레임
    """
    print("=" * 60)
    print("S&P 500 데이터 다운로드 시작")
    print("=" * 60)
    
    # S&P 500 티커 심볼
    ticker = "^GSPC"
    
    try:
        # 데이터 다운로드
        print(f"\n티커: {ticker}")
        print(f"기간: {period}")
        print(f"간격: {interval}")
        print("\n데이터 다운로드 중...")
        
        sp500 = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if sp500.empty:
            raise ValueError("데이터를 다운로드할 수 없습니다.")
        
        print(f"✓ 다운로드 완료: {len(sp500)} 행")
        print(f"✓ 기간: {sp500.index[0].strftime('%Y-%m-%d')} ~ {sp500.index[-1].strftime('%Y-%m-%d')}")
        
        return sp500
    
    except Exception as e:
        print(f"✗ 에러 발생: {str(e)}")
        raise

def calculate_returns(data):
    """
    수익률을 계산합니다.
    
    Parameters:
    -----------
    data : pd.DataFrame
        가격 데이터
    
    Returns:
    --------
    pd.DataFrame
        수익률이 추가된 데이터프레임
    """
    print("\n" + "=" * 60)
    print("수익률 계산")
    print("=" * 60)
    
    df = data.copy()
    
    # 단순 수익률 (Simple Returns)
    df['Simple_Return'] = df['Close'].pct_change()
    
    # 로그 수익률 (Log Returns)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 결측치 제거
    df = df.dropna()
    
    print(f"✓ 단순 수익률 계산 완료")
    print(f"✓ 로그 수익률 계산 완료")
    print(f"✓ 유효 데이터: {len(df)} 행")
    
    return df

def save_data(data, filename):
    """
    데이터를 CSV 파일로 저장합니다.
    
    Parameters:
    -----------
    data : pd.DataFrame
        저장할 데이터
    filename : str
        파일명
    """
    try:
        data.to_csv(filename)
        print(f"✓ 데이터 저장 완료: {filename}")
    except Exception as e:
        print(f"✗ 저장 실패: {str(e)}")

def display_data_info(data):
    """
    데이터 정보를 출력합니다.
    
    Parameters:
    -----------
    data : pd.DataFrame
        데이터프레임
    """
    print("\n" + "=" * 60)
    print("데이터 정보")
    print("=" * 60)
    
    print(f"\n데이터 형태: {data.shape}")
    print(f"\n컬럼:\n{data.columns.tolist()}")
    
    print("\n기초 통계량:")
    print(data[['Close', 'Volume']].describe())
    
    print("\n첫 5행:")
    print(data.head())
    
    print("\n마지막 5행:")
    print(data.tail())

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 60)
    print("S&P 500 데이터 수집 프로그램")
    print("=" * 60)
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 데이터 다운로드
        sp500_data = download_sp500_data(period='5y', interval='1d')
        
        # 2. 데이터 정보 출력
        display_data_info(sp500_data)
        
        # 3. 수익률 계산
        sp500_with_returns = calculate_returns(sp500_data)
        
        # 4. 데이터 저장
        print("\n" + "=" * 60)
        print("데이터 저장")
        print("=" * 60)
        save_data(sp500_data, 'sp500_data.csv')
        save_data(sp500_with_returns, 'sp500_returns.csv')
        
        print("\n" + "=" * 60)
        print("데이터 수집 완료!")
        print("=" * 60)
        print("\n다음 단계:")
        print("  python basic_analysis.py")
        
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
