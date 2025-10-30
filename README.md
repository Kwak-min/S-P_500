# S&P 500 변동성 분석 및 리스크 측정 프로젝트

## 📊 프로젝트 개요
S&P 500 지수 데이터를 활용하여 금융 시장의 변동성을 분석하고 다양한 리스크 측정 지표를 계산하는 프로젝트입니다.

## 🎯 프로젝트 목표
- 공개 금융 데이터를 활용한 실전 금융공학 분석
- 변동성 모델링 및 예측
- 리스크 관리 지표 계산 및 시각화

## 📁 프로젝트 구조
```
sp500_risk_analysis/
├── part1_data_collection/          # Part 1: 데이터 수집 및 기초 분석
│   ├── data_collector.py
│   ├── basic_analysis.py
│   └── README.md
├── part2_volatility_analysis/      # Part 2: 변동성 분석
│   ├── volatility_models.py
│   ├── garch_analysis.py
│   └── README.md
├── part3_risk_measurement/         # Part 3: 리스크 측정
│   ├── risk_metrics.py
│   ├── portfolio_analysis.py
│   └── README.md
└── requirements.txt
```

## 🚀 시작하기

### 필요 라이브러리 설치
```bash
pip install -r requirements.txt
```

### Part 1: 데이터 수집 및 기초 분석
```bash
cd part1_data_collection
python data_collector.py
python basic_analysis.py
```

### Part 2: 변동성 분석
```bash
cd part2_volatility_analysis
python volatility_models.py
python garch_analysis.py
```

### Part 3: 리스크 측정
```bash
cd part3_risk_measurement
python risk_metrics.py
python portfolio_analysis.py
```

## 📈 주요 기능

### Part 1: 데이터 수집 및 기초 분석
- Yahoo Finance API를 통한 S&P 500 데이터 수집
- 기초 통계량 분석 (평균, 표준편차, 왜도, 첨도)
- 수익률 분포 시각화
- 시계열 트렌드 분석

### Part 2: 변동성 분석
- 히스토리컬 변동성 계산
- 이동평균 변동성
- GARCH(1,1) 모델을 통한 변동성 예측
- 변동성 클러스터링 분석

### Part 3: 리스크 측정
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Conditional Value at Risk (CVaR)
- Sharpe Ratio, Sortino Ratio
- Maximum Drawdown
- 베타 계수

## 📊 분석 결과
각 파트별로 다음과 같은 결과물이 생성됩니다:
- 분석 그래프 (PNG 파일)
- 통계 리포트 (CSV/TXT 파일)
- 시각화된 대시보드

## 🛠️ 기술 스택
- **Python 3.8+**
- **pandas**: 데이터 처리
- **numpy**: 수치 계산
- **matplotlib / seaborn**: 시각화
- **yfinance**: 금융 데이터 수집
- **scipy**: 통계 분석
- **arch**: GARCH 모델링

## 📝 라이선스
MIT License

## 👤 작성자
금융공학 분석 프로젝트

## 📧 문의
프로젝트 관련 문의사항은 Issues를 통해 남겨주세요.

---
**Note**: 이 프로젝트는 교육 목적으로 작성되었으며, 실제 투자 결정에 사용하기 전에 전문가의 조언을 구하시기 바랍니다.
