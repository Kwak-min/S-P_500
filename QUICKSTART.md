# 빠른 시작 가이드 (Quick Start Guide)

## 🚀 프로젝트 설정

### 1. 저장소 클론
```bash
git clone <repository-url>
cd sp500_risk_analysis
```

### 2. 가상환경 생성 (권장)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 필요 라이브러리 설치
```bash
pip install -r requirements.txt
```

## 📊 프로젝트 실행

### Part 1: 데이터 수집 및 기초 분석 (10분)
```bash
cd part1_data_collection
python data_collector.py
python basic_analysis.py
```

**결과물:**
- `sp500_data.csv`: 원본 데이터
- `sp500_returns.csv`: 수익률 데이터
- `basic_statistics.txt`: 기초 통계량
- `price_chart.png`: 가격 차트
- `returns_distribution.png`: 수익률 분포
- `returns_histogram.png`: 히스토그램

### Part 2: 변동성 분석 (15분)
```bash
cd ../part2_volatility_analysis
python volatility_models.py
python garch_analysis.py
```

**결과물:**
- `volatility_measures.csv`: 변동성 측정값
- `volatility_comparison.png`: 변동성 비교
- `rolling_volatility.png`: 이동평균 변동성
- `volatility_clustering.png`: 변동성 클러스터링
- `garch_results.txt`: GARCH 모델 결과
- `garch_conditional_volatility.png`: 조건부 변동성
- `garch_forecast.png`: 변동성 예측
- `garch_diagnostics.png`: 모델 진단

### Part 3: 리스크 측정 (15분)
```bash
cd ../part3_risk_measurement
python risk_metrics.py
python portfolio_analysis.py
```

**결과물:**
- `risk_metrics.csv`: 리스크 지표
- `performance_metrics.txt`: 성과 지표
- `var_analysis.png`: VaR 분석
- `drawdown_analysis.png`: Drawdown 분석
- `risk_return_profile.png`: 리스크-수익 프로파일
- `stress_scenarios.png`: 스트레스 시나리오
- `distribution_analysis.png`: 분포 분석
- `summary_dashboard.png`: 요약 대시보드

## 🎯 단계별 학습 가이드

### 초보자 (금융공학 입문)
1. Part 1 먼저 실행 → 기초 통계 이해
2. 결과 파일 확인 및 해석
3. Part 1 README.md 정독

### 중급자 (변동성 모델링)
1. Part 1 & 2 순차 실행
2. GARCH 모델 결과 분석
3. 변동성 예측 정확도 평가

### 고급자 (리스크 관리)
1. 전체 파트 실행
2. 리스크 지표 해석
3. 포트폴리오 최적화 적용

## 💡 자주 묻는 질문 (FAQ)

### Q1: 데이터가 다운로드되지 않아요
**A:** 인터넷 연결을 확인하세요. yfinance는 실시간으로 Yahoo Finance에서 데이터를 가져옵니다.

### Q2: 그래프가 깨져 보여요
**A:** 한글 폰트 문제일 수 있습니다. 시스템에 맞는 폰트를 설정하세요.

### Q3: GARCH 모델이 수렴하지 않아요
**A:** 데이터 기간이 너무 짧거나 초기값 문제일 수 있습니다. 데이터 기간을 늘려보세요.

### Q4: 특정 기간만 분석하고 싶어요
**A:** `data_collector.py`의 `period` 파라미터를 수정하세요 (예: '1y', '2y', '5y').

### Q5: 다른 지수도 분석할 수 있나요?
**A:** 네! `data_collector.py`의 ticker를 변경하세요:
- NASDAQ: `^IXIC`
- Dow Jones: `^DJI`
- KOSPI: `^KS11`

## 🔧 문제 해결

### ImportError 발생 시
```bash
pip install --upgrade -r requirements.txt
```

### Memory Error 발생 시
```python
# data_collector.py에서 period를 줄이기
download_sp500_data(period='1y')  # 5y → 1y
```

### Permission Error 발생 시
```bash
# 관리자 권한으로 실행 (Windows)
# 또는 쓰기 권한 확인
chmod +w .
```

## 📚 추가 학습 자료

### 추천 도서
1. "Options, Futures, and Other Derivatives" - John Hull
2. "Value at Risk" - Philippe Jorion
3. "Quantitative Risk Management" - McNeil, Frey, Embrechts

### 온라인 강의
1. Coursera - Financial Engineering and Risk Management
2. MIT OpenCourseWare - Finance Theory
3. Khan Academy - Finance and Capital Markets

### 관련 자료
- [Yahoo Finance API 문서](https://pypi.org/project/yfinance/)
- [ARCH 라이브러리 문서](https://arch.readthedocs.io/)
- [pandas 금융 분석 튜토리얼](https://pandas.pydata.org/docs/user_guide/timeseries.html)

## 🤝 기여하기

이 프로젝트는 교육 목적으로 공개되었습니다. 개선 사항이나 버그를 발견하시면:

1. Issue를 생성하거나
2. Pull Request를 제출해주세요

## ⚠️ 면책 조항

이 프로젝트는 **교육 및 연구 목적**으로만 제작되었습니다. 

- 실제 투자 결정에 사용하지 마세요
- 과거 성과는 미래 수익을 보장하지 않습니다
- 투자 전 전문가의 조언을 구하세요

## 📞 연락처

프로젝트 관련 문의: GitHub Issues

---
**Happy Coding! 📈**
