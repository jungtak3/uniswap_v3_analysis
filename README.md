# Uniswap V3 전략 분석 도구

간단하게 만듬..
fee, spread 등 확실한 데이터를 구하기 어려움이 있었음
Ethusdt는 중간에 비어 있는 데이터가 없는데
ethusdc는 있음 바이낸스 문제

## 📁 프로젝트 구조

```
uniswap_v3_analysis_repo/
├── README.md                          # 프로젝트 설명
├── requirements.txt                   # Python 의존성 라이브러리
├── .gitignore                         # Git 무시 파일
├── main.py                           # 메인 실행 스크립트
├── src/                              # 소스 코드
│   ├── uniswapv3_strategy_fixed.py   # Uniswap V3 전략 분석
│   └── create_testing_period_chart.py # 테스트 기간 차트 생성
└── data/                             # 데이터 폴더 (CSV 파일)
```

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/jungtak3/uniswap_v3_analysis
cd uniswap_v3_analysis_repo
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 데이터 준비

`ETHUSDC_20181215_20250430.csv` 파일을 프로젝트 루트 디렉토리에 배치하세요.

### 4. 실행

```bash
python main.py
```

## 📊 기능

### 전략 분석 (`uniswapv3_strategy_fixed.py`)
- 4가지 Uniswap V3 유동성 공급 전략 구현
- 백테스팅 및 성과 평가
- 시각화 및 결과 저장

### 테스트 기간 차트 (`create_testing_period_chart.py`)
- 테스트 기간 시장 분석
- 가격 움직임 시각화
- 시장 체제 분석

## 🔄 자동 실행

`main.py`는 두 스크립트를 순차적으로 실행합니다:
1. 테스트 기간 차트 생성
2. Uniswap V3 전략 분석 실행

## 📈 출력 파일

- `uniswap_v3_testing_period_analysis.png` - 테스트 기간 분석 차트
- `uniswap_v3_market_regimes.png` - 시장 체제 분석
- `uniswap_v3_strategy_analysis_fixed.png` - 전략 성과 비교
- `uniswap_v3_strategy_performance_fixed.csv` - 상세 성과 데이터
