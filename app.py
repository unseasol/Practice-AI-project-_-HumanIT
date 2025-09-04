import os
import json
import sqlite3
import secrets
import datetime as dt
import math
from functools import wraps
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    abort,
    flash,
)
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
import joblib
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import requests

from supabase import create_client, Client
from jose import jwt, JWTError


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

    # Environment configuration
    app.config.update(
        SUPABASE_URL=os.environ.get("SUPABASE_URL", ""),
        SUPABASE_KEY=os.environ.get("SUPABASE_KEY", ""),
        SLACK_WEBHOOK=os.environ.get("SLACK_WEBHOOK", ""),
        ADMIN_EMAIL=os.environ.get("ADMIN_EMAIL", ""),
        SMTP_HOST=os.environ.get("SMTP_HOST", ""),
        SMTP_PORT = int(os.environ.get("SMTP_PORT") or 587),
        SMTP_USER=os.environ.get("SMTP_USER", ""),
        SMTP_PASS=os.environ.get("SMTP_PASS", ""),
        REPORT_CRON=os.environ.get("REPORT_CRON", "0 9 * * *"),  # default 09:00 daily
        MAX_CONTENT_LENGTH=500 * 1024 * 1024,  # 500 MB uploads (increased from 16MB)
        UPLOAD_EXTENSIONS={".csv"},
        DATABASE_PATH=os.path.join(os.path.dirname(__file__), "logs.db"),
        MODEL_PATH=os.path.join(os.path.dirname(__file__), "model", "model.pkl"),
        FEATURE_NAMES_PATH=os.path.join(os.path.dirname(__file__), "model", "feature_names.pkl"),
    )

    # Initialize Supabase client
    supabase: Client | None = None
    if app.config["SUPABASE_URL"] and app.config["SUPABASE_KEY"]:
        supabase = create_client(app.config["SUPABASE_URL"], app.config["SUPABASE_KEY"])  # type: ignore

    # Initialize DB
    _init_db(app.config["DATABASE_PATH"])

    # Load model and meta (pipeline + schema info)
    model = _load_model(app.config["MODEL_PATH"])
    meta_path = os.path.join(os.path.dirname(__file__), "model", "model_meta.pkl")
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else None

    # Store in app.config so it can be reused in routes
    app.config["MODEL_OBJ"] = model
    app.config["MODEL_META"] = meta
    
    # Debug information
    print(f"Model loaded: {model is not None}")

    if model is None:
        return jsonify({"error": "Model not loaded. Please train model first."}), 500
        print("WARNING: Model not loaded! Please run model/train.py first.")

    used_cols = list((meta or {}).get("used_cols", []))

    # CSRF token setup
    @app.before_request
    def ensure_csrf_token() -> None:
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)

    def validate_csrf() -> bool:
        token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        return bool(token and token == session.get("csrf_token"))

    # Auth helpers
    def _get_current_user_email() -> str | None:
        return session.get("user_email")

    def _get_current_jwt() -> str | None:
        return session.get("access_token")

    def _jwt_is_valid(token: str | None) -> bool:
        if not token:
            return False
        # Supabase uses JWT; best is to call get_user
        try:
            if supabase is not None:
                res = supabase.auth.get_user(token)
                return res is not None and res.user is not None  # type: ignore[attr-defined]
        except Exception:
            pass
        # Fallback: decode exp only if public key unavailable
        try:
            payload = jwt.get_unverified_claims(token)
            exp = int(payload.get("exp", 0))
            return exp > int(dt.datetime.now(dt.timezone.utc).timestamp())
        except JWTError:
            return False

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            token = _get_current_jwt()
            if not _jwt_is_valid(token):
                return jsonify({"error": "로그인이 필요합니다. 먼저 로그인해주세요."}), 401
            return view(*args, **kwargs)
        return wrapped

    def admin_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            token = _get_current_jwt()
            if not _jwt_is_valid(token):
                return redirect(url_for("login"))
            email = _get_current_user_email()
            if not email or email.lower() != str(app.config["ADMIN_EMAIL"]).lower():
                abort(403)
            return view(*args, **kwargs)
        return wrapped

    # Routes
    @app.route("/ping_supabase")
    def ping_supabase():
        try:
            if supabase is None:
                return {"ok": False, "msg": "Supabase not configured"}, 500
        # auth 익명 호출 테스트 (프로젝트 정보)
            user = supabase.auth.get_user(session.get("access_token"))  # 로그인 전이면 None
            return {"ok": True, "user": bool(getattr(user, "user", None))}
        except Exception as e:
            return {"ok": False, "error": str(e)}, 500

    @app.route("/")
    def index():
        return render_template("index.html", csrf_token=session.get("csrf_token"))

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "POST":
            if not validate_csrf():
                abort(400)
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            nickname = request.form.get("nickname", "").strip()
            try:
                if supabase is None:
                    raise RuntimeError("Supabase not configured")
                supabase.auth.sign_up({
                    "email": email,
                    "password": password,
                    "options": {"data": {"nickname": nickname}},
                })
                flash("회원가입 완료. 이메일 인증을 확인하세요.", "success")
                return redirect(url_for("login"))
            except Exception as e:
                return render_template("error.html", message=str(e)), 400
        return render_template("signup.html", csrf_token=session.get("csrf_token"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            if not validate_csrf():
                abort(400)
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            try:
                if supabase is None:
                    raise RuntimeError("Supabase not configured")
                result = supabase.auth.sign_in_with_password({"email": email, "password": password})
                access_token = result.session.access_token  # type: ignore[attr-defined]
                session["access_token"] = access_token
                session["user_email"] = email
                flash("로그인 성공", "success")
                return redirect(url_for("index"))
            except Exception as e:
                return render_template("error.html", message=str(e)), 400
        return render_template("login.html", csrf_token=session.get("csrf_token"))

    @app.route("/mypage")
    @login_required
    def mypage():
        email = session.get("user_email")
        with sqlite3.connect(app.config["DATABASE_PATH"]) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT id, uploaded_at, filename, total_rows, malicious_rows, malicious_ratio, details_json
                FROM uploads
                WHERE email = ?
                ORDER BY uploaded_at DESC
            """, (email,)).fetchall()

        logs = [dict(r) for r in rows]

        # 요약 카드용 집계
        total_uploads = len(logs)
        total_rows = sum(r["total_rows"] for r in logs) if logs else 0
        total_mal = sum(r["malicious_rows"] for r in logs) if logs else 0
        avg_ratio = (sum(r["malicious_ratio"] for r in logs) / total_uploads) if total_uploads else 0.0
        last_time = logs[0]["uploaded_at"] if logs else None

        # 클래스 분포 통합(여러 업로드의 malicious_details 합치기)
        by_class: dict[str, int] = {}
        import json as _json
        for r in logs:
            if r.get("details_json"):
                try:
                    d = _json.loads(r["details_json"])
                    for k, v in (d.get("malicious_details") or {}).items():
                        by_class[k] = by_class.get(k, 0) + int(v)
                except Exception:
                    pass

        # 전체 정상/악성 합계(파이차트용)
        total_normal = total_rows - total_mal

        return render_template(
            "mypage.html",
            logs=logs,
            summary=dict(
                total_uploads=total_uploads,
                total_rows=total_rows,
                total_mal=total_mal,
                total_normal=total_normal,
                avg_ratio=avg_ratio,
                last_time=last_time,
            ),
            by_class=by_class,
            csrf_token=session.get("csrf_token"),
        )

    @app.route("/mypage/<int:upload_id>")
    @login_required
    def upload_detail(upload_id: int):
        email = session.get("user_email")
        with sqlite3.connect(app.config["DATABASE_PATH"]) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT id, email, uploaded_at, filename, total_rows, malicious_rows, malicious_ratio, details_json
                FROM uploads
                WHERE id = ? AND email = ?
            """, (upload_id, email)).fetchone()
        if not row:
            abort(404)

        details = {}
        try:
            if row["details_json"]:
                details = json.loads(row["details_json"])
        except Exception:
            details = {}

        # 파이차트용 합계
        total_rows = row["total_rows"] or 0
        mal_rows = row["malicious_rows"] or 0
        normal_rows = total_rows - mal_rows

        # by-class 분포
        by_class = (details.get("malicious_details") or {})

        return render_template(
            "upload_detail.html",
            meta=dict(row),
            details=details,
            totals=dict(normal=normal_rows, malicious=mal_rows),
            by_class=by_class,
            csrf_token=session.get("csrf_token"),
        )


    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("index"))

    @app.route("/reset", methods=["GET", "POST"])
    def reset():
        if request.method == "POST":
            if not validate_csrf():
                abort(400)
            email = request.form.get("email", "").strip()
            try:
                if supabase is None:
                    raise RuntimeError("Supabase not configured")
                redirect_to = request.url_root.rstrip("/") + url_for("login")
                supabase.auth.reset_password_for_email(email, {"redirect_to": redirect_to})
                flash("비밀번호 재설정 이메일을 확인하세요.", "info")
                return redirect(url_for("login"))
            except Exception as e:
                return render_template("error.html", message=str(e)), 400
        return render_template("reset.html", csrf_token=session.get("csrf_token"))

    from flask import jsonify
    import traceback

    @app.errorhandler(500)
    def server_error(e):
        app.logger.exception("500 on %s", request.path)
        # 업로드/예측 API는 항상 JSON
        if request.path.startswith("/upload") or request.path.startswith("/predict"):
            return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500
        return render_template("error.html", message=f"서버 오류: {e}"), 500

    @app.route("/upload", methods=["POST"])
    @login_required
    def upload():
        import re, math, numpy as np, pandas as pd

        def _read_csv_upload(file_storage, **pd_kwargs):
            UNSW_HEADERS_45 = [
                "id","dur","proto","service","state","spkts","dpkts","sbytes","dbytes","rate",
                "sttl","dttl","sload","dload","sloss","dloss","sinpkt","dinpkt","sjit","djit",
                "swin","stcpb","dtcpb","dwin","tcprtt","synack","ackdat","smean","dmean",
                "trans_depth","response_body_len","ct_srv_src","ct_state_ttl","ct_dst_ltm",
                "ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm","is_ftp_login",
                "ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm","ct_srv_dst","is_sm_ips_ports",
                "attack_cat","label"
            ]
            UNSW_HEADERS_49 = [
                "srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes","sttl",
                "dttl","sloss","dloss","service","sload","dload","spkts","dpkts","swin","dwin",
                "stcpb","dtcpb","smean","dmean","trans_depth","response_body_len","sjit","djit",
                "sinpkt","dinpkt","tcprtt","synack","ackdat","is_sm_ips_ports","ct_state_ttl",
                "ct_flw_http_mthd","ct_srv_dst","ct_dst_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
                "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_src_ltm","attack_cat","label",
                "extra1","extra2","extra3","extra4","extra5"
            ][:49]

            # 내부 공용 리더: 매 호출마다 파일 포인터를 되감아 읽습니다.
            def _read_with(**kw):
                file_storage.stream.seek(0)
                merged = {**pd_kwargs, **kw}
                return pd.read_csv(file_storage, **merged)

            # 1) 1차 읽기(일반 헤더 가정)
            df = _read_with(low_memory=False)
            cols_lower = {str(c).strip().lower() for c in df.columns}
            expected_any = {"label","attack_cat","dur","spkts","dpkts","sbytes","dbytes","rate"}
            if cols_lower & expected_any:
                return df

            # 2) 헤더가 데이터로 들어간 듯하면 형식 복구
            def _looks_like_value(s: str) -> bool:
                s = str(s).strip()
                return bool(re.match(r"^\d+(\.\d+){3}$", s)) or bool(re.match(r"^\d+(\.\d+)?$", s))
            suspicious = sum(_looks_like_value(c) for c in list(df.columns)[:6])

            if suspicious >= 3:
                ncols = df.shape[1]
                if ncols == 45:
                    return _read_with(header=None, names=UNSW_HEADERS_45, low_memory=False)
                elif ncols == 49:
                    return _read_with(header=None, names=UNSW_HEADERS_49, low_memory=False)
                else:
                    names = UNSW_HEADERS_49[:ncols]
                    return _read_with(header=None, names=names, low_memory=False)

            # 3) 판단 불가 → 1차 읽기 유지
            return df        
        try:
            model = app.config.get("MODEL_OBJ")
            meta = app.config.get("MODEL_META")
            print("=== UPLOAD REQUEST STARTED ===")

            # --- CSRF & 파일 ---
            if not validate_csrf():
                return jsonify({"error": "Invalid CSRF"}), 400
            if "file" not in request.files:
                return jsonify({"error": "No file"}), 400

            f = request.files["file"]
            filename = secure_filename(f.filename or "")
            print(f"File received: {filename}")
            if os.path.splitext(filename)[1].lower() not in app.config["UPLOAD_EXTENSIONS"]:
                return jsonify({"error": "Only CSV allowed"}), 400

            # --- CSV 로드: DtypeWarning 방지 ---
            df = _read_csv_upload(f)
            print(f"CSV loaded successfully, shape={df.shape}")
            print(f"Columns(head): {list(df.columns)[:10]}")

            # --- 모델/메타 확인 ---
            if model is None or meta is None:
                return jsonify({"error": "Model not loaded. Please train model first."}), 500

            # ===== 전처리 (당신이 쓰고 있는 버전 유지) =====
            def _norm(s: str) -> str:
                return re.sub(r"[^a-z0-9]", "", str(s).lower())

            used_cols = list(meta.get("used_cols", []))
            aliases_map = {
                "flowduration": ["Flow Duration", "duration", "dur"],
                "totalfwdpackets": ["Total Fwd Packets", "spkts", "fwd pkts", "fwd_packets"],
                "totalbackwardpackets": ["Total Backward Packets", "dpkts", "bwd pkts", "bwd_packets"],
                "flowbytes/s": ["Flow Bytes/s", "sload", "bytes per second", "flow bytes s"],
                "flowpackets/s": ["Flow Packets/s", "rate", "packets per second", "flow packets s"],
            }

            norm2orig = {_norm(c): c for c in df.columns}

            def get_series_by_alias(aliases, default_val=0):
                for a in aliases:
                    key = _norm(a)
                    if key in norm2orig:
                        s = pd.to_numeric(df[norm2orig[key]], errors="coerce")
                        return s.fillna(0)
                return pd.Series([default_val] * len(df), index=df.index)

            df_proc = pd.DataFrame(index=df.index)
            for target_col in used_cols:
                key = _norm(target_col)
                cand_aliases = aliases_map.get(key, [target_col])
                series = get_series_by_alias(cand_aliases, default_val=0)

                if (series is None) or (series.sum() == 0 and series.max() == 0):
                    if key == "flowduration":
                        series = get_series_by_alias(["dur", "duration", "Flow Duration"], default_val=0)
                    elif key == "flowbytes/s":
                        sbytes = get_series_by_alias(["sbytes", "src_bytes"], default_val=0)
                        dbytes = get_series_by_alias(["dbytes", "dst_bytes"], default_val=0)
                        dur = get_series_by_alias(["dur", "Flow Duration", "duration"], default_val=0)
                        series = (sbytes.add(dbytes, fill_value=0) / dur.replace(0, pd.NA)).fillna(0)
                    elif key == "flowpackets/s":
                        spkts = get_series_by_alias(["spkts", "Total Fwd Packets"], default_val=0)
                        dpkts = get_series_by_alias(["dpkts", "Total Backward Packets"], default_val=0)
                        dur = get_series_by_alias(["dur", "Flow Duration", "duration"], default_val=0)
                        series = (spkts.add(dpkts, fill_value=0) / dur.replace(0, pd.NA)).fillna(0)

                df_proc[target_col] = pd.to_numeric(series, errors="coerce").fillna(0)

            # ✅ 타입·순서·결측 고정 (간헐 500의 주범 예방)
            df_proc = df_proc.reindex(columns=used_cols, fill_value=0.0)
            df_proc = df_proc.apply(pd.to_numeric, errors="coerce").astype("float64").fillna(0.0)
            df_proc[df_proc < 0] = 0.0
            print(f"[DEBUG] df_proc shape={df_proc.shape}")
            print("[DEBUG] dtypes:", df_proc.dtypes.to_dict())

            # --- 예측 ---
            preds = model.predict(df_proc)
            preds_labels = preds  # (멀티클래스 파이프라인이면 바로 문자열일 것)
            print(f"Prediction successful, sample={preds_labels[:10]}")

            preds_series = pd.Series(preds_labels, name="pred").astype(str)
            normal_mask = preds_series.str.lower().isin(["benign", "normal"])
            malicious_mask = ~normal_mask

            malicious_count = int(malicious_mask.sum())
            normal_count = int(normal_mask.sum())
            total_count = int(len(preds_series))
            malicious_ratio = (malicious_count / total_count) if total_count else 0.0
            print(f"Results: normal={normal_count}, malicious={malicious_count}, ratio={malicious_ratio:.4f}")

            # --- 상위 악성 10개 (메모리 안전하게 힙으로 선별) ---
            import heapq
            def _safe_num(v, default=0):
                try: return float(v)
                except: return default
            def _safe_str(v, default="N/A"):
                try:
                    s = str(v).strip()
                    return default if s == "" or s.lower() == "nan" else s
                except: return default

            _colmap = {_norm(c): c for c in df.columns}
            def _pick(series, candidates, default=None):
                for cand in candidates:
                    col = _colmap.get(_norm(cand))
                    if col is not None:
                        return series.get(col, default)
                return default
            _PROTO_NUM = {"6": "tcp", "17": "udp", "1": "icmp"}
            def _map_proto(v):
                try:
                    s = str(v).strip().lower()
                    if s in ("tcp", "udp", "icmp"): return s
                    return _PROTO_NUM.get(str(int(float(s))), s)
                except: return "n/a"

            CAND_DURATION = ["Flow Duration", "duration", "dur"]
            CAND_PROTOCOL = ["Protocol", "protocol", "proto", "protocol_type"]
            CAND_SRCB = ["src_bytes","sbytes","Total Length of Fwd Packets","TotLen Fwd Pkts","TotLenFwdPkts","Fwd Packets Length Total","Avg Fwd Segment Size"]
            CAND_DSTB = ["dst_bytes","dbytes","Total Length of Bwd Packets","TotLen Bwd Pkts","TotLenBwdPkts","Bwd Packets Length Total","Avg Bwd Segment Size"]

            heap = []  # (score, duration, dict)
            mask_arr = malicious_mask.values if isinstance(malicious_mask, pd.Series) else malicious_mask
            
            # Malicious prediction별 집계 추가
            malicious_by_type = {}
            
            for i in np.flatnonzero(mask_arr):
                r = df.iloc[i]
                duration  = _safe_num(_pick(r, CAND_DURATION, 0))
                proto_raw = _pick(r, CAND_PROTOCOL, "n/a")
                protocol  = _safe_str(_map_proto(proto_raw))
                src_bytes = _safe_num(_pick(r, CAND_SRCB, 0))
                dst_bytes = _safe_num(_pick(r, CAND_DSTB, 0))
                prediction = _safe_str(preds_series.iloc[i])
                
                # Malicious 타입별 집계
                malicious_by_type[prediction] = malicious_by_type.get(prediction, 0) + 1
                
                row = {
                    "duration": duration, "protocol": protocol,
                    "src_bytes": src_bytes, "dst_bytes": dst_bytes,
                    "prediction": prediction,
                }
                score = (src_bytes or 0) + (dst_bytes or 0)
                if len(heap) < 10:
                    heapq.heappush(heap, (score, duration, row))
                else:
                    heapq.heappushpop(heap, (score, duration, row))

            top_mal_out = [t[2] for t in sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)]

            # === 요약 결과 JSON 구성 ===
            result_payload = {
                "success": True,
                "total_records": total_count,       # len(df) 대신 total_count 사용
                "normal": normal_count,
                "normal_count": normal_count,       # 프론트 호환용 중복 키 유지
                "malicious": malicious_count,
                "malicious_count": malicious_count, # 프론트 호환용 중복 키 유지
                "ratio": float(malicious_ratio),
                "malicious_details": malicious_by_type,  # 파이차트용 상세 분류
                "top_malicious": top_mal_out,
                "malicious_by_type": malicious_by_type   # (호환을 위해 중복 유지)
            }

            # === 로그 저장(파일명 + 상세 JSON까지 저장) ===
            email = _get_current_user_email() or "unknown"
            _insert_log(
                app.config["DATABASE_PATH"],
                email,
                dt.datetime.utcnow(),
                malicious_ratio,
                total_count,
                malicious_count,
                filename=filename,                                            # ← 추가 저장
                details_json=json.dumps(result_payload, ensure_ascii=False)   # ← 추가 저장
            )

            # === 응답 ===
            return jsonify(result_payload)


            #return jsonify({
            #    "success": True,  # 프론트엔드 호환성을 위해 추가
            #    "total_records": total_count,  # 기존: len(df) 대신 total_count 사용
            #    "normal": normal_count,
            #    "normal_count": normal_count,  # 프론트엔드 호환성을 위해 추가
            #    "malicious": malicious_count,
            #    "malicious_count": malicious_count,  # 프론트엔드 호환성을 위해 추가
            #    "ratio": float(malicious_ratio),
            #    "malicious_details": malicious_by_type,  # 파이차트용 상세 분류 (기존 malicious_by_type 사용)
            #    "top_malicious": top_mal_out,
            #    "malicious_by_type": malicious_by_type  # 추가: 중복이지만 호환성을 위해
            #})

        except Exception as e:
            # 어떤 예외든 JSON으로 반환 + 스택로그
            app.logger.exception("Upload failed")
            return jsonify({"error": f"Internal error: {e.__class__.__name__}: {e}"}), 500

    @app.route("/predict", methods=["POST"])
    @login_required
    def predict_api():
        model = app.config.get("MODEL_OBJ")
        meta = app.config.get("MODEL_META")
        if not request.is_json:
            return jsonify({"error": "JSON required"}), 400
        if not validate_csrf():
            return jsonify({"error": "Invalid CSRF"}), 400

        payload = request.get_json(silent=True) or {}

        if model is None or meta is None:
            return jsonify({"error": "Model or meta not loaded"}), 500

        # --- JSON payload → DataFrame 변환 ---
        X = pd.DataFrame([{c: payload.get(c, 0) for c in meta["used_cols"]}])
        for c in X.columns:
            if X[c].dtype == "object":
                X[c] = X[c].astype(str).fillna("0")
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

        try:
            pred = model.predict(X)[0]
            label = "malicious" if str(pred).lower() not in ("benign", "normal") else "normal"
            return jsonify({"prediction": label, "pred": str(pred)})
        except Exception as e:
            return jsonify({"error": f"예측 실패: {str(e)}"}), 400

    @app.route("/dashboard")
    @login_required
    def dashboard():
        # Alias of index for authenticated users
        return redirect(url_for("index"))

    @app.route("/admin")
    @admin_required
    def admin():
        rows = _fetch_all_logs(app.config["DATABASE_PATH"]) or []
        # Prepare aggregates
        df = pd.DataFrame(rows, columns=["id", "email", "uploaded_at", "malicious_ratio", "total_rows", "malicious_rows"]) if rows else pd.DataFrame(columns=["id","email","uploaded_at","malicious_ratio","total_rows","malicious_rows"])
        user_counts = (
            df.groupby("email")["id"].count().sort_values(ascending=False).to_dict()
            if not df.empty else {}
        )
        time_series = (
            df.assign(ts=pd.to_datetime(df["uploaded_at"]))
              .sort_values("ts")
              .assign(malicent=(df["malicious_ratio"] * 100.0))
              [["ts", "malicent"]]
              .to_dict(orient="records") if not df.empty else []
        )
        totals = {
            "malicious": int(df["malicious_rows"].sum()) if not df.empty else 0,
            "normal": int((df["total_rows"].sum() - df["malicious_rows"].sum())) if not df.empty else 0,
        }
        return render_template(
            "admin.html",
            rows=rows,
            user_counts=user_counts,
            time_series=time_series,
            totals=totals,
            csrf_token=session.get("csrf_token"),
        )

    @app.route("/admin/send_report", methods=["POST"])
    @admin_required
    def admin_send_report():
        if not validate_csrf():
            abort(400)
        csv_path = _generate_report_csv(app.config["DATABASE_PATH"])  # returns path
        _send_email(
            host=app.config["SMTP_HOST"],
            port=app.config["SMTP_PORT"],
            username=app.config["SMTP_USER"],
            password=app.config["SMTP_PASS"],
            subject="악성 탐지 보고서",
            sender=app.config["SMTP_USER"],
            recipients=[app.config["ADMIN_EMAIL"]] if app.config["ADMIN_EMAIL"] else [],
            body="첨부된 CSV를 확인하세요.",
            attachments=[csv_path] if csv_path else None,
        )
        flash("보고서가 전송되었습니다.", "success")
        return redirect(url_for("admin"))

    @app.errorhandler(403)
    def forbidden(_):
        return render_template("error.html", message="접근이 거부되었습니다."), 403

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", message="페이지를 찾을 수 없습니다."), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template("error.html", message=f"서버 오류: {e}"), 500

    @app.template_filter()
    def comma(value):
        try:
            return "{:,}".format(int(value))
        except (ValueError, TypeError):
            return value


    # Scheduler for automatic reports
    scheduler = BackgroundScheduler(daemon=True)
    try:
        minute, hour, day, month, weekday = _parse_cron(app.config["REPORT_CRON"])  # type: ignore
        scheduler.add_job(
            func=lambda: _send_scheduled_report(app),
            trigger=CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=weekday),
            id="daily_report",
            replace_existing=True,
        )
        scheduler.start()
    except Exception:
        # Ignore scheduler failures in constrained environments
        pass

    return app


def _parse_cron(expr: str) -> tuple[str, str, str, str, str]:
    parts = (expr or "").split()
    if len(parts) != 5:
        return ("0", "9", "*", "*", "*")
    return tuple(parts)  # type: ignore[return-value]


def _send_scheduled_report(app: Flask) -> None:
    try:
        csv_path = _generate_report_csv(app.config["DATABASE_PATH"])  # type: ignore[index]
        _send_email(
            host=app.config["SMTP_HOST"],
            port=app.config["SMTP_PORT"],
            username=app.config["SMTP_USER"],
            password=app.config["SMTP_PASS"],
            subject="[자동] 악성 탐지 일일 보고서",
            sender=app.config["SMTP_USER"],
            recipients=[app.config["ADMIN_EMAIL"]] if app.config["ADMIN_EMAIL"] else [],
            body="자동 발송 보고서입니다. 첨부 CSV를 확인하세요.",
            attachments=[csv_path] if csv_path else None,
        )
    except Exception:
        pass


def _init_db(db_path: str) -> None:
    import sqlite3, os
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA busy_timeout=3000;")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            malicious_ratio REAL NOT NULL,
            total_rows INTEGER NOT NULL,
            malicious_rows INTEGER NOT NULL,
            filename TEXT,
            details_json TEXT
        )
        """)
        # 안전하게 컬럼 추가(이미 있으면 무시)
        for add in ("ALTER TABLE uploads ADD COLUMN filename TEXT",
                    "ALTER TABLE uploads ADD COLUMN details_json TEXT"):
            try: cur.execute(add)
            except Exception: pass
        conn.commit()

def _insert_log(db_path: str, email: str, uploaded_at: dt.datetime,
                ratio: float, total: int, mal: int,
                filename: str | None = None,
                details_json: str | None = None) -> None:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO uploads
              (email, uploaded_at, malicious_ratio, total_rows, malicious_rows, filename, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (email, uploaded_at.isoformat(), float(ratio), int(total), int(mal),
              filename, details_json))
        conn.commit()



def _fetch_all_logs(db_path: str) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, email, uploaded_at, malicious_ratio, total_rows, malicious_rows FROM uploads ORDER BY uploaded_at DESC")
        return cur.fetchall()


def _generate_report_csv(db_path: str) -> str | None:
    rows = _fetch_all_logs(db_path)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["id", "email", "uploaded_at", "malicious_ratio", "total_rows", "malicious_rows"])
    out_path = os.path.join(os.path.dirname(db_path), f"report_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def _load_feature_names(feature_names_path: str) -> list[str] | None:
    """Load feature names from the training script"""
    try:
        print(f"Attempting to load feature names from: {feature_names_path}")
        if os.path.exists(feature_names_path):
            print(f"Feature names file exists, loading...")
            feature_names = joblib.load(feature_names_path)
            print(f"Feature names loaded successfully: {feature_names}")
            return feature_names
        else:
            print(f"Feature names file does not exist: {feature_names_path}")
            return None
    except Exception as e:
        print(f"Error loading feature names: {str(e)}")
        return None


def _preprocess_flexible(df: pd.DataFrame, feature_names: list[str] | None) -> pd.DataFrame:
    """Flexible preprocessing that works with different column formats"""
    if feature_names is None:
        # Fallback to original preprocessing
        return _preprocess(df)
    
    # Create a mapping for common column variations
    column_mappings = {
        # Duration variations
        "dur": ["dur", "duration", "Flow Duration"],
        "proto": ["proto", "protocol", "protocol_type"],
        "sbytes": ["sbytes", "src_bytes", "Total Length of Fwd Packets"],
        "dbytes": ["dbytes", "dst_bytes", "Total Length of Bwd Packets"],
        "spkts": ["spkts", "Total Fwd Packets"],
        "dpkts": ["dpkts", "Total Backward Packets"],
        "rate": ["rate", "Flow Packets/s"],
        "sload": ["sload", "Flow Bytes/s"],
        "dload": ["dload"],
    }
    
    # Specific mappings for expected features
    feature_mappings = {
        "Flow Duration": ["dur", "duration", "Flow Duration"],
        "Total Fwd Packets": ["spkts", "Total Fwd Packets"],
        "Total Backward Packets": ["dpkts", "Total Backward Packets"],
        "Flow Packets/s": ["rate", "Flow Packets/s"],
        "Flow Bytes/s": ["sload", "Flow Bytes/s"],
    }
    
    df_proc = pd.DataFrame()
    
    # Try to map available columns to expected features
    for feature in feature_names:
        found = False
        
        # First, try direct match
        if feature in df.columns:
            df_proc[feature] = df[feature]
            found = True
            print(f"Direct match found for {feature}")
        else:
            # Try feature-specific mappings
            if feature in feature_mappings:
                for possible_col in feature_mappings[feature]:
                    if possible_col in df.columns:
                        df_proc[feature] = df[possible_col]
                        found = True
                        print(f"Mapped {possible_col} -> {feature}")
                        break
            
            # If still not found, try general column mappings
            if not found:
                for col in df.columns:
                    # Check variations
                    for variations in column_mappings.values():
                        if col in variations and feature in variations:
                            df_proc[feature] = df[col]
                            found = True
                            print(f"General mapping: {col} -> {feature}")
                            break
                    if found:
                        break
            
            # If still not found, try fuzzy matching
            if not found:
                for col in df.columns:
                    if feature.lower() in col.lower() or col.lower() in feature.lower():
                        df_proc[feature] = df[col]
                        found = True
                        print(f"Fuzzy match: {col} -> {feature}")
                        break
        
        # If still not found, fill with 0
        if not found:
            df_proc[feature] = 0
            print(f"No mapping found for {feature}, using 0")
    
    # Ensure numeric conversion
    for col in df_proc.columns:
        df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce").fillna(0)
    
    return df_proc


def _create_dataframe_from_payload(payload: dict, feature_names: list[str] | None) -> pd.DataFrame:
    """Create DataFrame from API payload with flexible column mapping"""
    if feature_names is None:
        # Fallback to original format
        return pd.DataFrame([{
            "duration": payload.get("duration", 0),
            "protocol": payload.get("protocol", "tcp"),
            "src_bytes": payload.get("src_bytes", 0),
            "dst_bytes": payload.get("dst_bytes", 0),
        }])
    
    # Create DataFrame with available features
    data = {}
    for feature in feature_names:
        # Try to find matching key in payload
        found = False
        for key in payload.keys():
            if key == feature or key.lower() == feature.lower():
                data[feature] = payload[key]
                found = True
                break
        
        # If not found, use default values
        if not found:
            if "dur" in feature or "duration" in feature:
                data[feature] = payload.get("duration", 0)
            elif "proto" in feature:
                data[feature] = payload.get("protocol", "tcp")
            elif "sbytes" in feature or "src" in feature:
                data[feature] = payload.get("src_bytes", 0)
            elif "dbytes" in feature or "dst" in feature:
                data[feature] = payload.get("dst_bytes", 0)
            else:
                data[feature] = 0
    
    return pd.DataFrame([data])


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Map protocol to integers; unknown -> -1
    mapping = {"tcp": 0, "udp": 1, "icmp": 2}
    df_proc = df.copy()
    df_proc["protocol"] = df_proc["protocol"].astype(str).str.lower().map(mapping).fillna(-1).astype(int)
    for col in ["duration", "src_bytes", "dst_bytes"]:
        df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce").fillna(0)
    return df_proc[["duration", "protocol", "src_bytes", "dst_bytes"]]

import numpy as np
def safe_log1p(a):
    a = np.asarray(a, dtype=float)
    a = np.where(~np.isfinite(a), 0.0, a)
    a = np.where(a < -0.999999, -0.999999, a)
    return np.log1p(a)

def _load_model(model_path: str):
    """Load the trained model from file"""
    try:
        print(f"Attempting to load model from: {model_path}")
        if os.path.exists(model_path):
            print("Model file exists, loading...")
            model = joblib.load(model_path)
            print(f"Model loaded successfully: {type(model)}")
            return model
        else:
            print(f"Model file does not exist: {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def _notify_slack(webhook: str, email: str, ratio: float, total: int) -> None:
    if not webhook:
        return
    try:
        text = f":rotating_light: 경보: 사용자 {email}의 업로드에서 악성 비율 {ratio:.2%} (총 {total}행)"
        requests.post(webhook, json={"text": text}, timeout=5)
    except Exception:
        pass


def _send_email(host: str, port: int, username: str, password: str, subject: str, sender: str, recipients: list[str], body: str, attachments: list[str] | None = None) -> None:
    if not (host and port and username and password and recipients):
        return
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        for path in (attachments or []):
            try:
                part = MIMEBase('application', 'octet-stream')
                with open(path, 'rb') as f:
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}"')
                msg.attach(part)
            except Exception:
                continue

        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(sender, recipients, msg.as_string())
    except Exception:
        pass


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


