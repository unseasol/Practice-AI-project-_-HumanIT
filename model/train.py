# project/model/train.py
import os, glob
import numpy as np
import pandas as pd
import unicodedata
import unicodedata
import re
import numpy as np
from io import StringIO
from collections import Counter

from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
import joblib

# =========================
# 설정
# =========================
# 업로드 코드와 동일한 "표준 피처" 5개만 사용(호환성 유지)
CANON_COLS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
]

def safe_log1p(X):
    """log1p 전에 NaN/Inf/음수 방어해서 피클 가능한 전역 함수로 사용"""
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = 0.0  # NaN/±Inf -> 0
    X[X < 0] = 0.0            # 음수 -> 0
    return np.log1p(X)

def _read_csv_robust(path: str) -> pd.DataFrame:
    """
    인코딩/구분자/깨진 줄에 강한 CSV 로더.
    - 여러 인코딩 시도 (utf-8 계열, MS 계열 등)
    - 기본은 콤마 구분자로 시도 → 실패 시 sep=None로 자동 추정
    - on_bad_lines(TypeError 시 error_bad_lines)로 깨진 줄 스킵
    - 마지막엔 'replace' 디코딩으로라도 읽기
    """
    def _try_read(enc: str, use_auto_sep: bool):
        # pandas 버전에 따라 on_bad_lines 지원/미지원 대응
        read_kwargs = dict(encoding=enc)
        if use_auto_sep:
            read_kwargs.update(sep=None, engine="python")  # 자동 구분자 추정(이때만 python 엔진)
        try:
            # 최신 판다스
            return pd.read_csv(path, on_bad_lines="skip", **read_kwargs)
        except TypeError:
            # 구버전 호환
            return pd.read_csv(path, error_bad_lines=False, warn_bad_lines=False, **read_kwargs)

    encodings = ("utf-8", "utf-8-sig", "cp949", "euc-kr", "cp1252", "latin1")
    # 1) 콤마 구분자 우선
    for enc in encodings:
        try:
            return _try_read(enc, use_auto_sep=False)
        except UnicodeDecodeError:
            continue
        except Exception:
            # 구분자 문제일 수 있으니 자동 추정으로 재시도
            try:
                return _try_read(enc, use_auto_sep=True)
            except Exception:
                continue

    # 2) 최후의 수단: 바이트 → UTF-8 replace 디코딩 → 파싱
    with open(path, "rb") as fh:
        raw = fh.read().decode("utf-8", errors="replace")
    try:
        return pd.read_csv(StringIO(raw), on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(StringIO(raw), error_bad_lines=False, warn_bad_lines=False)

def _clean_text(s) -> str:
    """라벨/문자열 정규화: 깨진 문자/다중 공백/유니코드 정리."""
    try:
        s = str(s)
    except Exception:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("�", " ")
           .replace("–", "-").replace("—", "-")
           .replace("“", '"').replace("”", '"')
           .replace("’", "'").replace("‚", "'"))
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 선택: "hgb" | "rf" | "xgb"
MODEL_KIND = os.environ.get("MODEL_KIND", "hgb").lower()


# =========================
# 유틸
# =========================
def _safe_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _detect_schema_by_cols(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if "Label" in cols:
        return "cicids"
    if "attack_cat" in cols:
        return "unsw_multi"
    if "label" in cols:
        return "unsw_bin"
    raise ValueError("Unknown schema: no Label/attack_cat/label")

def _to_canonical(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    """스키마별 → 공통 5피처 + 멀티클래스 라벨('target')"""
    out = pd.DataFrame(index=df.index, columns=CANON_COLS)

    if schema == "cicids":
        # 숫자 피처(있으면 그대로, 없으면 0)
        out["Flow Duration"]         = _safe_num(df.get("Flow Duration", 0))
        out["Total Fwd Packets"]     = _safe_num(df.get("Total Fwd Packets", 0))
        out["Total Backward Packets"]= _safe_num(df.get("Total Backward Packets", 0))
        fbps = _safe_num(df.get("Flow Bytes/s", 0))
        fps  = _safe_num(df.get("Flow Packets/s", 0))

        # 없는 경우 파생
        if (fbps == 0).all():
            fwd_len = _safe_num(df.get("Total Length of Fwd Packets", 0))
            bwd_len = _safe_num(df.get("Total Length of Bwd Packets", 0))
            dur = out["Flow Duration"].replace(0, np.nan)
            fbps = (fwd_len + bwd_len) / dur

        if (fps == 0).all():
            sp = out["Total Fwd Packets"]
            dp = out["Total Backward Packets"]
            dur = out["Flow Duration"].replace(0, np.nan)
            fps = (sp + dp) / dur

        out["Flow Bytes/s"]   = fbps.fillna(0)
        out["Flow Packets/s"] = fps.fillna(0)

        # 라벨: BENIGN은 Benign, 나머지 원래 이름 유지
        y = df["Label"].astype(str).map(_clean_text)
        target = np.where(y.str.upper() == "BENIGN", "Benign", y)

    elif schema in ("unsw_multi", "unsw_bin"):
        dur    = _safe_num(df.get("dur", 0))
        spkts  = _safe_num(df.get("spkts", 0))
        dpkts  = _safe_num(df.get("dpkts", 0))
        sbytes = _safe_num(df.get("sbytes", 0))
        dbytes = _safe_num(df.get("dbytes", 0))
        rate   = _safe_num(df.get("rate", 0))

        out["Flow Duration"]          = dur
        out["Total Fwd Packets"]      = spkts
        out["Total Backward Packets"] = dpkts

        # Flow Packets/s: rate 우선, 없으면 (spkts+dpkts)/dur
        fps = rate.copy()
        if (fps == 0).any():
            dur_safe = dur.replace(0, np.nan)
            fps = fps.mask(fps == 0, (spkts + dpkts) / dur_safe)
        out["Flow Packets/s"] = fps.fillna(0)

        # Flow Bytes/s: (sbytes+dbytes)/dur
        fbps = (sbytes + dbytes) / dur.replace(0, np.nan)
        out["Flow Bytes/s"] = fbps.fillna(0)

        if schema == "unsw_multi":
            lab = df["attack_cat"].astype(str).map(_clean_text)
            # 'Normal'은 Benign으로 통일, 나머지는 원래 카테고리 유지(대문자/소문자 혼합 방지)
            target = np.where(lab.str.lower() == "normal", "Benign", lab)
        else:  # unsw_bin
            lab = df["label"].astype(str)
            target = np.where(lab == "1", "Attack", "Benign")

    else:
        raise ValueError("Unsupported schema")

    # 숫자 보정
    for c in CANON_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out["target"] = pd.Series(target).astype(str)
    return out

def _iter_dataframes(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet"))) + \
            sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No parquet/csv found in {data_dir}")

    for p in files:
        if p.lower().endswith(".parquet"):
            df = pd.read_parquet(p)
        else:
            df = _read_csv_robust(p)   # ← 앞서 넣은 안전 로더 사용

        # 1) 무한대만 NaN으로
        df = df.replace([np.inf, -np.inf], np.nan)

        # 2) 숫자형만 0으로 채우기 (범주형/문자형은 건드리지 않음)
        num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
        if num_cols:
            df[num_cols] = df[num_cols].fillna(0)

        # 3) 범주형/문자형은 그대로 두되, 파이프라인에서 처리되도록 둠
        #    (원하면 다음 2줄로 안전 전환 가능)
        # for c in df.columns:
        #     if not is_numeric_dtype(df[c]):
        #         df[c] = df[c].astype(object)

        yield p, df



# =========================
# 메인
# =========================
def main():
    root      = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_dir  = os.path.join(root, "data")
    out_model = os.path.join(os.path.dirname(__file__), "model.pkl")
    out_meta  = os.path.join(os.path.dirname(__file__), "model_meta.pkl")

    canon_list = []
    for path, df in _iter_dataframes(data_dir):
        try:
            schema = _detect_schema_by_cols(df)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(path)}: {e}")
            continue
        print(f"[LOAD] {os.path.basename(path)} → schema={schema}, rows={len(df)}")
        cdf = _to_canonical(df, schema)
        print(f"  labels: {Counter(cdf['target']).most_common(10)}")
        canon_list.append(cdf)

    if not canon_list:
        raise RuntimeError("No usable training data.")
    data = pd.concat(canon_list, ignore_index=True)

    # X, y
    X = data[CANON_COLS].copy()
    y = data["target"].astype(str)

    print("[INFO] merged rows =", len(X))
    dist = Counter(y)
    print("[INFO] class distribution =", dist.most_common(20))

    # 로그 변환으로 스케일 안정화
    # pre = ColumnTransformer([
    #     ("num_log1p", Pipeline(steps=[
    #         ("imp",   SimpleImputer(strategy="constant", fill_value=0.0)),
    #         ("log1p", FunctionTransformer(safe_log1p, validate=False)),
    #     ]), CANON_COLS),
    # ], remainder="drop") 모델 생성 후 오류 방지를 위해 아래 코드 사용

    pre = ColumnTransformer([
    ("num_log1p", FunctionTransformer(np.log1p, validate=False), CANON_COLS),
    ])


    # 클래스 불균형 대응(샘플 가중치: 1/freq)
    inv_freq = {c: 1.0 / cnt for c, cnt in dist.items()}
    sample_weight = y.map(inv_freq).values

    # ===== 모델 선택 =====
    if MODEL_KIND == "rf":
        model = RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=42,
            class_weight="balanced_subsample"
        )
    elif MODEL_KIND == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError("xgboost가 설치되어 있지 않습니다. pip install xgboost 후 MODEL_KIND=xgb 로 실행하세요.") from e
        model = XGBClassifier(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            objective="multi:softprob",
        )
    else:  # 기본: HistGradientBoosting (대개 RF보다 성능 좋고 빠름)
        model = HistGradientBoostingClassifier(
            max_depth=None,
            learning_rate=0.06,
            max_iter=650,
            l2_regularization=0.0,
            random_state=42
        )

    pipe = Pipeline([("pre", pre), ("clf", model)])

    X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
        X, y, sample_weight, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_tr, y_tr, clf__sample_weight=sw_tr)  # HGB/RF는 sample_weight 지원, XGB는 내부에서 무시됨

    y_pr = pipe.predict(X_te)
    print(classification_report(y_te, y_pr, zero_division=0))

    joblib.dump(pipe, out_model)
    joblib.dump(
        {
            "used_cols": CANON_COLS,
            "task": "multiclass",
            "classes": sorted(list(set(y))),
            "notes": f"MODEL_KIND={MODEL_KIND}, canonicalized CICIDS+UNSW, log1p"
        },
        out_meta
    )
    print(f"[OK] saved: {out_model}")
    print(f"[OK] saved: {out_meta}")

if __name__ == "__main__":
    main()
