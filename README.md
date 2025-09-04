# 클라우드 악성코드 예측 감지 웹앱

Flask + Supabase + RandomForest 기반 CSV 업로드 악성 탐지 웹앱.

## 기능 요약
- Supabase 이메일/비밀번호 회원가입/로그인, JWT 세션 보관
- CSV 업로드 후 예측 및 비율 차트 시각화 (Chart.js)
- 업로드 로그 SQLite 저장, 악성 비율 30% 이상 Slack/이메일 알림
- 관리자 대시보드: 로그 조회, 통계/차트, 보고서 이메일 발송
- /predict JSON API
- CSRF 보호, JWT 만료 확인, 기본 XSS 방지(Jinja 자동 이스케이프)

## 폴더 구조
```
project/
├── app.py
├── model/
│   ├── train.py
│   └── model.pkl
├── static/
│   └── chart.js
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── reset.html
│   ├── admin.html
│   └── error.html
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 사전 준비
- Supabase 프로젝트 생성 후 `SUPABASE_URL`, `SUPABASE_KEY` 확보
- 관리자 이메일 설정: `ADMIN_EMAIL`
- Slack Incoming Webhook (선택): `SLACK_WEBHOOK`
- SMTP 설정: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`

## 로컬 실행
1) 의존성 설치
```
pip install -r requirements.txt
```
2) 모델 학습 파일 생성 (CSV 없으면 synthetic 데이터 사용)
```
python model/train.py --csv path/to/train.csv  # 선택
```
3) 환경변수 설정 (PowerShell 예)
```
$env:SUPABASE_URL = "..."
$env:SUPABASE_KEY = "..."
$env:ADMIN_EMAIL = "admin@example.com"
$env:SLACK_WEBHOOK = "https://hooks.slack.com/services/..."
$env:SMTP_HOST = "smtp.example.com"
$env:SMTP_PORT = "587"
$env:SMTP_USER = "no-reply@example.com"
$env:SMTP_PASS = "yourpass"
$env:FLASK_SECRET_KEY = "change_me"
```
4) Flask 실행
```
python app.py
```
5) 접속: `http://localhost:5000`

## Docker 실행
1) `.env` 파일(옵션)
```
SUPABASE_URL=...
SUPABASE_KEY=...
ADMIN_EMAIL=admin@example.com
SLACK_WEBHOOK=https://hooks.slack.com/services/...
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=no-reply@example.com
SMTP_PASS=yourpass
FLASK_SECRET_KEY=change_me
```
2) 빌드 및 실행
```
docker compose up --build -d
```
3) 접속: `http://localhost:5000`

## 사용 방법
- 회원가입 → 이메일 인증 → 로그인
- 업로드: CSV에 `duration, protocol, src_bytes, dst_bytes` 포함
- 결과 파이차트 및 악성 상위 10개 테이블 확인
- 관리자(`ADMIN_EMAIL`)는 `/admin` 접근 가능, 보고서 이메일 발송 가능

## 보안 메모
- CSRF 토큰 검사(`/upload`, `/predict`, 폼 POST)
- JWT 유효성: Supabase `get_user` 호출 또는 `exp` 확인
- XSS: Jinja2 기본 이스케이프, 사용자 입력은 템플릿에 직접 삽입 지양

## API
- POST `/predict`
```
{
  "duration": 1.23,
  "protocol": "tcp",
  "src_bytes": 123,
  "dst_bytes": 456
}
```
응답: `{ "prediction": "normal|malicious", "pred": 0|1 }`

## 주의사항
- `model/model.pkl`은 `train.py`를 실행해 생성하세요.
- SMTP/Slack 미설정 시 관련 기능은 무시됩니다.
- 본 예제는 데모용이며, 운영 환경에서는 추가 보안/로깅/에러처리가 필요합니다.
