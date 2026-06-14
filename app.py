# app.py (UPDATED — adds password reset endpoints; otherwise unchanged)
import os
import io
import json
import joblib
import traceback
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    session, jsonify, send_file
)
from flask_sqlalchemy import SQLAlchemy
# ------------------ EMAIL SUPPORT ------------------
from flask_mail import Mail, Message

mail = Mail()      # create mail object

from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import bcrypt
import sqlite3
import threading
import psutil
from scapy.all import sniff, IP
import matplotlib.pyplot as plt
# Paste these with the other imports near top of app.py
import time
import threading
import socket
from collections import defaultdict, deque
from flask import session, current_app

# scapy (use local imports inside functions too if you prefer)
from scapy.all import sniff, IP, TCP, UDP, Raw

SEVERITY_RULES = {
    "benign": ("Low", "Normal traffic.", "No action needed."),
    "portscan": ("High", "Port scanning activity detected.", "Block the source IP and check firewall logs."),
    "hostsweep_Pn": ("Medium", "Host sweep probing detected.", "Monitor the host and restrict ICMP if needed."),
    "bruteforce_http": ("High", "HTTP brute-force detected.", "Enable rate limiting and review login attempts."),
    "bruteforce_https": ("High", "HTTPS brute-force detected.", "Enable CAPTCHA or MFA."),
    "sql_injection_http": ("High", "SQL Injection attempt over HTTP.", "Filter user input and inspect web server logs."),
    "sql_injection_https": ("High", "SQL Injection attempt over HTTPS.", "Activate WAF and sanitize SQL inputs."),
    "dos_http": ("High", "Possible HTTP DoS attack.", "Rate-limit traffic and block attacker IP."),
    "dos_https": ("High", "Possible HTTPS DoS attack.", "Analyze TLS flood attempts."),
    "ftp_login": ("High", "FTP brute-force login attempts detected.", "Disable anonymous login and enforce strong passwords."),
    "ssh_login": ("Medium", "SSH login attempt.", "Check for unauthorized SSH attempts."),
    "ssh_login_successful": ("High", "Suspicious successful SSH login.", "Verify credentials and check access logs."),
    "ssrf_http": ("High", "Potential SSRF attack detected.", "Block internal resource access."),
    "ssrf_https": ("High", "Potential SSRF over HTTPS.", "Validate URLs and block private IP access."),
    "xss_http": ("Medium", "Possible XSS attack.", "Sanitize HTML inputs."),
    "xss_https": ("Medium", "Possible XSS over HTTPS.", "Enable content security policy."),
    "revshell_http": ("High", "Reverse shell detected.", "Immediately isolate the host!"),
    "smtp_version": ("Low", "SMTP version request.", "Normal unless repeated."),
    "ftp_version": ("Low", "FTP version probe.", "Normal unless repeated."),
    "smtp_enum": ("High", "SMTP enumeration attempt.", "Block attacker IP."),
}

def classify_packet(pkt):
    try:
        payload = len(bytes(pkt))
    except:
        payload = 0

    if payload > 1200:
        return "DoS"
    if payload < 80 and pkt.haslayer("TCP"):
        return "PortScan"
    if pkt.haslayer("UDP") and payload < 60:
        return "probe"
    if pkt.haslayer("ICMP") and payload > 500:
        return "Botnet"

    return "Normal"

# itsdangerous for token generation/validation (commonly available with Flask installs)
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

# Optional PDF: try import reportlab
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ----------------------------
# Helpers
# ----------------------------
def safe_read_csv(path):
    """Try reading CSV robustly with fallbacks."""
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, low_memory=True)
        except Exception:
            try:
                return pd.read_csv(path, engine="python")
            except Exception as e:
                raise e

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ----------------------------
# Universal converter: any CSV -> Web-IDS23-ish
# (keeps minimal assumptions and derives required features)
# ----------------------------
def convert_to_webids23(df):
    df = df.copy()

    # Normalize column names to snake_case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Automap common names -> webids names
    column_map = {
        "forward_packets": "fwd_pkts_tot",
        "fwd_packets": "fwd_pkts_tot",
        "total_fwd_packets": "fwd_pkts_tot",
        "backward_packets": "bwd_pkts_tot",
        "bwd_packets": "bwd_pkts_tot",
        "total_bwd_packets": "bwd_pkts_tot",
        "duration": "flow_duration",
        "flow_duration_ms": "flow_duration",
        "flow_duration_ns": "flow_duration",
        "time": "flow_duration",
        "payload": "payload_bytes",
        "bytes": "payload_bytes",
        "total_bytes": "payload_bytes",
        "len": "payload_bytes",
        "length": "payload_bytes",
    }
    for col in list(df.columns):
        if col in column_map:
            df.rename(columns={col: column_map[col]}, inplace=True)

    # Ensure baseline numeric columns exist
    baseline = [
        "fwd_pkts_tot", "bwd_pkts_tot", "fwd_header_size_tot", "bwd_header_size_tot",
        "payload_bytes", "flow_duration", "fwd_init_window_size", "bwd_init_window_size"
    ]
    for c in baseline:
        if c not in df.columns:
            df[c] = 0

    # Convert numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

    # Fix flow_duration units and zeros
    df["flow_duration"] = df["flow_duration"].replace(np.inf, np.nan)
    df["flow_duration"] = pd.to_numeric(df["flow_duration"], errors="coerce").fillna(0.001)
    max_dur = df["flow_duration"].max() if len(df) > 0 else 0
    if pd.notna(max_dur):
        if max_dur > 1_000_000:     # likely nanoseconds
            df["flow_duration"] = df["flow_duration"] / 1e9
        elif max_dur > 1000:       # likely milliseconds
            df["flow_duration"] = df["flow_duration"] / 1000

    df["flow_duration"] = df["flow_duration"].replace(0, 0.001).fillna(0.001)

    # Derived features same as training
    df["flow_pkts_per_sec"] = (df["fwd_pkts_tot"].fillna(0) + df["bwd_pkts_tot"].fillna(0)) / df["flow_duration"]
    df["fwd_pkts_per_sec"] = df["fwd_pkts_tot"].fillna(0) / df["flow_duration"]
    df["bwd_pkts_per_sec"] = df["bwd_pkts_tot"].fillna(0) / df["flow_duration"]
    df["payload_bytes_per_second"] = df["payload_bytes"].fillna(0) / df["flow_duration"]

    # Safe numeric fallback
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df

# ----------------------------
# Load environment & initialize
# ----------------------------
load_dotenv()
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "uploads", "reports")
ensure_dir(UPLOAD_FOLDER)
ensure_dir(REPORTS_FOLDER)

app = Flask(__name__, static_folder="static", template_folder="templates")

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "my_secret")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "network_intrusion_app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Mail Config (optional)
app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", 465))
app.config["MAIL_USE_SSL"] = os.getenv("MAIL_USE_SSL", "True").lower() in ["true", "1", "yes"]
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")

db = SQLAlchemy(app)
mail.init_app(app)

# ----------------------------
# Database Models
# ----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(256), unique=True, nullable=False)
    password = db.Column(db.String(512), nullable=False)

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(64), nullable=False)
    attack_type = db.Column(db.String(128), nullable=False)
    src_ip = db.Column(db.String(128))
    dst_ip = db.Column(db.String(128))
    extra = db.Column(db.String(512))

with app.app_context():
    db.create_all()

# ----------------------------
# Strong Password Check
# ----------------------------
def strong_password(pw: str) -> bool:
    return (
        len(pw) >= 8
        and any(c.isalpha() for c in pw)
        and any(c.isdigit() for c in pw)
        and any(c in "@#$%^&*!?._-" for c in pw)
    )

# ----------------------------
# Token serializer for reset links
# ----------------------------
def get_serializer():
    return URLSafeTimedSerializer(app.config["SECRET_KEY"])

def make_reset_token(email):
    s = get_serializer()
    return s.dumps(email, salt="password-reset-salt")

def verify_reset_token(token, max_age_seconds=3600):
    s = get_serializer()
    try:
        email = s.loads(token, salt="password-reset-salt", max_age=max_age_seconds)
        return email
    except SignatureExpired:
        return None
    except BadSignature:
        return None

# ----------------------------
# Load ML MODELS
# ----------------------------
MODEL_DIR = os.path.join(BASE_DIR, "models")
binary_model = None
multiclass_model = None
supervised_scaler = None
model_features = []
multiclass_label_names = None

def load_models():
    global binary_model, multiclass_model, supervised_scaler, model_features, multiclass_label_names
    try:
        binary_model = joblib.load(os.path.join(MODEL_DIR, "rf_binary.pkl"))
    except Exception as e:
        print("❌ Could not load rf_binary.pkl:", e)
    try:
        multiclass_model = joblib.load(os.path.join(MODEL_DIR, "rf_multiclass.pkl"))
    except Exception as e:
        print("❌ Could not load rf_multiclass.pkl:", e)
    try:
        supervised_scaler = joblib.load(os.path.join(MODEL_DIR, "supervised_scaler.pkl"))
    except Exception as e:
        print("❌ Could not load supervised_scaler.pkl:", e)
    try:
        model_features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))
    except Exception as e:
        print("❌ Could not load model_features.pkl:", e)
    # Extract class labels if multiclass model exposes them
    try:
        if multiclass_model is not None and hasattr(multiclass_model, "classes_"):
            multiclass_label_names = list(multiclass_model.classes_)
            print("✅ Multiclass labels extracted:", multiclass_label_names)
    except Exception as e:
        print("❌ Could not extract multiclass labels:", e)

load_models()

# ----------------------------
# REALTIME FIXED GLOBAL STATE
# ----------------------------
# ----------------------------
# REALTIME FIXED GLOBAL STATE
# ----------------------------
realtime_running = False
realtime_alerts = []
realtime_interface = r"\Device\NPF_Loopback"
realtime_thread = None
current_user_email = None

import psutil
from scapy.all import sniff, IP

def select_best_interface():
    """Pick the interface with a valid IPv4 (not 169.x.x.x)."""
    best = None
    for iface, addrs in psutil.net_if_addrs().items():
        for a in addrs:
            if a.family == 2:  # IPv4
                ip = a.address
                if ip.startswith("192.") or ip.startswith("10.") or ip.startswith("172."):
                    best = iface
    return best

# ----------------------------
# REALTIME SNIFFER (FINAL WORKING VERSION)
# ----------------------------
# ----------------------------
# REALTIME SNIFFER (LOWERCASE VERSION)
# ----------------------------
# ===== Realtime ML sniffer (paste/replace existing sniffer) =====
# Assumes model_features, supervised_scaler, binary_model, multiclass_model, multiclass_label_names exist

_FLOW_TIMEOUT = 0.3   # seconds of inactivity to finalize a flow
_FLOW_MAX_PACKETS = 4

def _get_global(name_upper, name_lower):
    g = globals()
    return g.get(name_upper, g.get(name_lower))

def _set_global(name_upper, name_lower, value):
    g = globals()
    if name_upper in g:
        g[name_upper] = value
    else:
        g[name_lower] = value

def run_realtime_ml_sniffer():
    print("🔥 ML Sniffer STARTED")

    running = _get_global("REALTIME_RUNNING", "realtime_running")
    alerts_list = _get_global("REALTIME_ALERTS", "realtime_alerts")
    iface = _get_global("REALTIME_INTERFACE", "realtime_interface")

    # Flow store
    flows = {}
    last_cleanup = time.time()

    # ---------------------------------------------------------
    #                FINALIZE FLOW (CORRECT INDENTED)
    # ---------------------------------------------------------
    def finalize_flow(key):
        print("\n🔥 finalize_flow CALLED for key:", key)

        info = flows.pop(key, None)
        if not info:
            return

        pkts = info["pkts"]
        if not pkts:
            print("❌ EMPTY PACKET LIST")
            return

        # ---------------- BASIC META ----------------
        first_ts = pkts[0]["ts"]
        last_ts = pkts[-1]["ts"]
        duration = max(0.0001, last_ts - first_ts)

        fwd_pkts = sum(1 for p in pkts if p["dir"] == "fwd")
        bwd_pkts = sum(1 for p in pkts if p["dir"] == "bwd")
        total_payload = sum(p["payload_len"] for p in pkts)

        fwd_hdrs = [p["hdr_len"] for p in pkts if p["dir"] == "fwd"]
        bwd_hdrs = [p["hdr_len"] for p in pkts if p["dir"] == "bwd"]

        # TCP FLAGS
        syn = fin = rst = ack = cwr = ece = 0
        fwd_flags = defaultdict(int)
        bwd_flags = defaultdict(int)

        for p in pkts:
            flags = p.get("tcp_flags", "")
            if p["dir"] == "fwd":
                for c in flags:
                    fwd_flags[c] += 1
            else:
                for c in flags:
                    bwd_flags[c] += 1

            syn += ("S" in flags)
            fin += ("F" in flags)
            rst += ("R" in flags)
            ack += ("A" in flags)
            ece += ("E" in flags)
            cwr += ("C" in flags)

        # ---------------- FEATURE VECTOR ----------------
        feat = {
            "flow_pkts_per_sec": (fwd_pkts + bwd_pkts) / duration,
            "fwd_pkts_per_sec": fwd_pkts / duration,
            "bwd_pkts_per_sec": bwd_pkts / duration,
            "flow_duration": duration,
            "bwd_header_size_tot": sum(bwd_hdrs) if bwd_hdrs else 0,
            "bwd_pkts_tot": bwd_pkts,
            "down_up_ratio": (bwd_pkts / (fwd_pkts+1)) if fwd_pkts else 0,
            "bwd_data_pkts_tot": sum(1 for p in pkts if p["dir"]=="bwd" and p["payload_len"]>0),
            "flow_ack_flag_count": ack,
            "fwd_data_pkts_tot": sum(1 for p in pkts if p["dir"]=="fwd" and p["payload_len"]>0),
            "payload_bytes_per_second": total_payload / duration,
            "fwd_header_size_tot": sum(fwd_hdrs) if fwd_hdrs else 0,
            "fwd_pkts_tot": fwd_pkts,
            "flow_syn_flag_count": syn,
            "flow_fin_flag_count": fin,
            "flow_rst_flag_count": rst,
            "flow_ece_flag_count": ece,
            "flow_cwr_flag_count": cwr,
        }

        print("🔥 REALTIME FEATURES:", feat)

        # -------------------------------------------------
        #               RULE-BASED DETECTION
        # -------------------------------------------------
        src = info["src"]
        dst = info["dst"]
        sport = info["sport"]
        dport = info["dport"]
        proto = info["proto"]

        # PORTSCAN
        if proto == "TCP" and syn >= 1 and feat["flow_pkts_per_sec"] > 2000:
            return push_alert(info, "portscan", "High", 1, "Port scanning detected.")

        # HOSTSWEEP
        if proto == "ICMP" or dst.startswith("239."):
            return push_alert(info, "hostsweep_Pn", "Medium", 1, "Host sweep detected.")

        # BRUTEFORCE HTTP
        if dport in [80, 8080] and feat["flow_pkts_per_sec"] > 40:
            return push_alert(info, "bruteforce_http", "High", 1, "HTTP brute-force detected.")

        # BRUTEFORCE HTTPS
        if dport == 443 and feat["flow_pkts_per_sec"] > 40:
            return push_alert(info, "bruteforce_https", "High", 1, "HTTPS brute-force detected.")

        # SQL INJECTION
        if dport in [80, 8080] and feat["payload_bytes_per_second"] > 60000:
            return push_alert(info, "sql_injection_http", "High", 1, "SQL Injection HTTP.")

        if dport == 443 and feat["payload_bytes_per_second"] > 60000:
            return push_alert(info, "sql_injection_https", "High", 1, "SQL Injection HTTPS.")

        # DOS
        if feat["flow_pkts_per_sec"] > 8000:
            name = "dos_https" if dport == 443 else "dos_http"
            return push_alert(info, name, "High", 1, "DoS attack detected.")

        # FTP BRUTE
        if dport == 21 and feat["flow_pkts_per_sec"] > 30:
            return push_alert(info, "ftp_login", "High", 1, "FTP brute-force detected.")

        # REVERSE SHELL
        if sport in [4444, 4445, 1337] or dport in [4444, 4445, 1337]:
            return push_alert(info, "revshell_http", "High", 1, "Reverse shell detected.")

        # SSRF
        if proto == "TCP" and feat["fwd_data_pkts_tot"] > 20 and feat["bwd_data_pkts_tot"] == 0:
            return push_alert(info, "ssrf_http", "High", 1, "SSRF detected.")

        # XSS
        if feat["payload_bytes_per_second"] > 20000:
            return push_alert(info, "xss_http", "Medium", 1, "Possible XSS attack.")

        # -------------------------------------------------
        #             ML PREDICTION (fallback)
        # -------------------------------------------------
        cols = model_features
        row = {c: float(feat.get(c, 0)) for c in cols}

        X_row = pd.DataFrame([row], columns=cols)
        try:
            X_scaled = supervised_scaler.transform(X_row)
            bp = binary_model.predict(X_scaled)[0]
            mp = multiclass_model.predict(X_scaled)[0]
        except:
            bp = 0
            mp = 0

        label = multiclass_label_names[int(mp)]
        severity, desc, advice = SEVERITY_RULES.get(label, ("Low", "", ""))

        return push_alert(info, label, severity, bp, desc)

    # ---------------------------------------------------------
    #                PACKET SNIFF LOOP
    # ---------------------------------------------------------
    print("Sniffer running on:", iface)

    def process_packet(pkt):
        try:
            if IP in pkt:
                proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "ICMP"
                src = pkt[IP].src
                dst = pkt[IP].dst
                sport = pkt.sport if hasattr(pkt, "sport") else 0
                dport = pkt.dport if hasattr(pkt, "dport") else 0
                ts = time.time()

                key = (proto, src, sport, dst, dport)

                if key not in flows:
                    flows[key] = {
                        "pkts": [],
                        "src": src,
                        "dst": dst,
                        "sport": sport,
                        "dport": dport,
                        "proto": proto,
                    }

                flows[key]["pkts"].append({
                    "dir": "fwd",
                    "ts": ts,
                    "hdr_len": len(pkt),
                    "payload_len": len(pkt.payload) if hasattr(pkt, "payload") else 0,
                    "tcp_flags": pkt.sprintf("%TCP.flags%") if TCP in pkt else "",
                })

                # finalize after 0.7s idle
                if ts - flows[key]["pkts"][0]["ts"] > 0.7:
                    finalize_flow(key)

        except Exception as e:
            print("Packet error:", e)

    sniff(iface=iface, prn=process_packet, store=False)
def push_alert(info, label, severity, binary_flag, desc=""):
    global current_user_email

    alert = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "src": info.get("src", "unknown"),
        "dst": info.get("dst", "unknown"),
        "type": label,
        "sev": severity,
        "desc": desc,
        "advice": SEVERITY_RULES.get(label, ("", "", ""))[2],
        "binary": int(binary_flag) if hasattr(binary_flag, '__int__') else binary_flag,
        "multi": label
    }

    # in-memory list
    realtime_alerts.append(alert)
    if len(realtime_alerts) > 500:
        realtime_alerts.pop(0)

    # store alert in sqlite DB (notifications table)
    try:
        conn = sqlite3.connect("network_intrusion_flaskdb", check_same_thread=False)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO notifications (time, src, dst, attack_type, severity, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (alert["time"], alert["src"], alert["dst"], str(alert["type"]), alert["sev"], alert["desc"]))
        conn.commit()
        conn.close()
        print("💾 Stored alert in DB")
    except Exception as e:
        print("❌ DB ERROR:", e)

    # choose recipient: prefer logged-in user; fall back to configured MAIL_USERNAME (for testing)
    receiver = current_user_email or app.config.get("MAIL_USERNAME")
    print("DEBUG push_alert: current_user_email =", current_user_email)
    print("DEBUG push_alert: using receiver =", receiver)

    if receiver and app.config.get("MAIL_USERNAME") and app.config.get("MAIL_PASSWORD"):
        try:
            # ensure we have app context (sending from background thread)
            with app.app_context():
                msg = Message(
                    subject=f"⚠ Realtime Attack Detected: {label}",
                    sender=app.config.get("MAIL_USERNAME"),
                    recipients=[receiver]
                )
                msg.body = (
                    f"Realtime Attack Detected!\n\n"
                    f"Time: {alert['time']}\n"
                    f"Source: {alert['src']}\n"
                    f"Destination: {alert['dst']}\n"
                    f"Type: {label}\n"
                    f"Severity: {severity}\n"
                    f"Description: {desc}\n\n"
                    "— Network Intrusion Detection System"
                )
                print("DEBUG about to mail.send()")
                mail.send(msg)
                print("📧 Email sent to:", receiver)
        except Exception as e:
            print("❌ Email send error:", e)
    else:
        print("⚠ No logged-in user email available or mail credentials missing; skipping realtime email")

    print("🚨 ALERT →", alert)
    return alert
# ===== end of realtime ML sniffer =====
# ----------------------------
# Preprocess (replicates training pipeline)
# ----------------------------
def preprocess_new_file(df):
    df = df.copy()
    # normalize
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    # ensure duration
    if "flow_duration" not in df.columns:
        df["flow_duration"] = 0.001
    df["flow_duration"] = df["flow_duration"].replace(0, np.nan).fillna(0.001)
    # engineered
    if "flow_pkts_per_sec" not in df.columns:
        df["flow_pkts_per_sec"] = (df.get("fwd_pkts_tot", 0) + df.get("bwd_pkts_tot", 0)) / df["flow_duration"]
    if "fwd_pkts_per_sec" not in df.columns:
        df["fwd_pkts_per_sec"] = df.get("fwd_pkts_tot", 0) / df["flow_duration"]
    if "bwd_pkts_per_sec" not in df.columns:
        df["bwd_pkts_per_sec"] = df.get("bwd_pkts_tot", 0) / df["flow_duration"]
    if "payload_bytes_per_second" not in df.columns:
        df["payload_bytes_per_second"] = df.get("payload_bytes", 0) / df["flow_duration"]
    # fill numeric missing
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean() if not df[col].isna().all() else 0)
    # ensure model features present
    if model_features and len(model_features) > 0:
        for f in model_features:
            if f not in df.columns:
                df[f] = 0
        df_final = df[model_features]
    else:
        df_final = df.select_dtypes(include=[np.number])
    df_final = df_final.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df_final

# ----------------------------
# Prediction + mapping
# ----------------------------
def predict_uploaded_csv(df_raw):
    # 1) convert to webids-like
    df_conv = convert_to_webids23(df_raw)

    # 2) create feature matrix
    X = preprocess_new_file(df_conv)

    # 3) scale
    if supervised_scaler is not None:
        try:
            X_scaled = supervised_scaler.transform(X)
        except Exception as e:
            print("Scaler transform error:", e)
            X_scaled = X.values
    else:
        X_scaled = X.values

    # 4) predict
    if binary_model is None or multiclass_model is None:
        raise RuntimeError("Models not loaded. Check models/ folder.")

    try:
        binary_pred = binary_model.predict(X_scaled)
        multi_pred = multiclass_model.predict(X_scaled)
    except Exception as e:
        print("Prediction error:", e)
        # fallback: mark normal
        binary_pred = np.zeros(X.shape[0], dtype=int)
        multi_pred = np.zeros(X.shape[0], dtype=int)

    # 5) map multiclass numeric labels -> names if available
    multi_names = None
    if multiclass_label_names is not None:
        # if multiclass model returns indices (0..n-1) mapping via classes_ is correct
        try:
            # if multi_pred are ints and classes_ aligned:
            if multi_pred.dtype.kind in ("i", "u") or np.all(np.isin(multi_pred, np.arange(len(multiclass_label_names)))):
                multi_names = [multiclass_label_names[int(x)] if pd.notna(x) else "unknown" for x in multi_pred]
            else:
                # maybe multi_pred already are class labels (strings)
                multi_names = [str(x) for x in multi_pred]
        except Exception:
            multi_names = [str(x) for x in multi_pred]
    else:
        multi_names = [str(x) for x in multi_pred]

    # Attach predictions
    out = df_conv.reset_index(drop=True).copy()
    out["binary"] = binary_pred
    out["multi"] = multi_pred
    out["multi_name"] = multi_names
    out["predicted_label"] = out["binary"].apply(lambda x: "Attack" if int(x) == 1 else "Normal")
    return out

# ----------------------------
# REPORT / EMAIL helpers
# ----------------------------
def generate_pdf_report(breakdown, total, attacks, save_path):
    """Generate a simple PDF report. Uses reportlab if installed. Returns path or None."""
    if not REPORTLAB_AVAILABLE:
        return None
    try:
        c = canvas.Canvas(save_path, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, height - 60, "Network Intrusion Detection Report")
        c.setFont("Helvetica", 11)
        c.drawString(40, height - 90, f"Generated: {datetime.utcnow().isoformat()} UTC")
        c.drawString(40, height - 120, f"Total rows analyzed: {total}")
        c.drawString(40, height - 140, f"Attacks detected: {attacks}")
        c.drawString(40, height - 170, "Attack breakdown:")
        y = height - 200
        for k, v in breakdown.items():
            c.drawString(60, y, f"- {k}: {v}")
            y -= 16
            if y < 60:
                c.showPage()
                y = height - 60
        c.save()
        return save_path
    except Exception as e:
        print("PDF generation failed:", e)
        return None

def save_csv_report(df, save_path):
    try:
        df.to_csv(save_path, index=False)
        return save_path
    except Exception as e:
        print("CSV save failed:", e)
        return None

def send_attack_alert(email, attacks, total, breakdown=None, report_path=None):
    if not app.config.get("MAIL_USERNAME") or not app.config.get("MAIL_PASSWORD"):
        print("Mail credentials not configured; skipping email.")
        return False
    try:
        msg = Message(
            subject="🚨 Network IDS Alert: Attacks Detected",
            sender=app.config["MAIL_USERNAME"],
            recipients=[email],
        )
        body_lines = [
            "Network Intrusion Detection System - Alert",
            "",
            f"Total rows analyzed: {total}",
            f"Attacks detected: {attacks}",
            ""
        ]
        if breakdown:
            body_lines.append("Breakdown:")
            for k, v in breakdown.items():
                body_lines.append(f"  - {k}: {v}")
            body_lines.append("")
        body_lines.append("This is an automated message from your IDS.")
        msg.body = "\n".join(body_lines)
        # attach report if provided
        if report_path and os.path.exists(report_path):
            with open(report_path, "rb") as fh:
                filename = os.path.basename(report_path)
                msg.attach(filename, "application/octet-stream", fh.read())
        mail.send(msg)
        print("📧 Email sent to", email)
        return True
    except Exception as e:
        print("❌ Email send failed:", e)
        traceback.print_exc()
        return False

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/notifications")
def notifications_page():
    conn = sqlite3.connect("network_intrusion_flaskdb", check_same_thread=False)
    cur = conn.cursor()

    cur.execute("SELECT time, src, dst, attack_type, severity, description FROM notifications ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()

    notifications = [
        {
            "time": r[0],
            "src": r[1],
            "dst": r[2],
            "attack_type": r[3],
            "severity": r[4],
            "description": r[5]
        }
        for r in rows
    ]

    return render_template("notifications.html", notifications=notifications)

@app.route("/settings")
def settings():
    return render_template("settings.html")

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("profile.html", user_email=session["user"])

# ----------------------------
# AUTH ROUTES
# ----------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"].lower()
        pw = request.form["password"]
        confirm = request.form["confirm"]
        if pw != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("signup"))
        if not strong_password(pw):
            flash("Weak password.", "danger")
            return redirect(url_for("signup"))
        if User.query.filter_by(email=email).first():
            flash("Email already exists.", "warning")
            return redirect(url_for("signup"))
        hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
        db.session.add(User(email=email, password=hashed))
        db.session.commit()
        flash("Account created!", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    global current_user_email   # <-- required to update global email

    if request.method == "POST":
        email = request.form["email"].lower()
        pw = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if not user:
            flash("Account not found.", "danger")
            return redirect(url_for("login"))

        if bcrypt.checkpw(pw.encode(), user.password.encode()):
            # Save email in session for normal pages
            session["user"] = user.email
            session["email"] = user.email
            global current_user_email

            # Save email globally for REALTIME email thread
            current_user_email = user.email

            flash("Logged in!", "success")
            return redirect(url_for("home"))

        flash("Invalid password.", "danger")

    return render_template("login.html")
@app.route('/logout')
def logout():
    session.clear()
    global current_user_email
    current_user_email = None
    flash("Logged out", "info")
    return redirect(url_for('home'))

# ----------------------------
# PASSWORD RESET ROUTES (ADDED)
# ----------------------------
@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    """
    Displays a form where user enters their email to receive a reset link.
    If mail not configured, the reset link will be printed to console for local testing.
    """
    if request.method == "POST":
        email = request.form.get("email", "").lower()
        if not email:
            flash("Please provide your email.", "warning")
            return redirect(url_for("reset_password"))

        user = User.query.filter_by(email=email).first()
        if not user:
            # Do not reveal whether email exists — flash generic message
            flash("If an account with that email exists, a reset link will be sent.", "info")
            return redirect(url_for("login"))

        token = make_reset_token(email)
        reset_url = url_for("reset_password_token", token=token, _external=True)

        # send email if configured
        if app.config.get("MAIL_USERNAME") and app.config.get("MAIL_PASSWORD"):
            try:
                msg = Message(
                    subject="Password reset for your Network IDS",
                    sender=app.config["MAIL_USERNAME"],
                    recipients=[email],
                )
                msg.body = f"Use the following link to reset your password. The link expires in 1 hour.\n\n{reset_url}"
                mail.send(msg)
                flash("If an account with that email exists, a reset link will be sent.", "info")
            except Exception as e:
                print("Email send failed:", e)
                flash("Could not send email. Check server logs.", "danger")
        else:
            # local dev fallback: print the link and show a notice
            print("Password reset link (dev):", reset_url)
            flash("Reset link printed to console (dev). Check server logs.", "info")

        return redirect(url_for("login"))

    return render_template("reset_password.html")

@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password_token(token):
    """
    Link landed from email. If GET: show form to set new password.
    If POST: validate token and update password.
    """
    email = verify_reset_token(token)
    if not email:
        flash("Reset link is invalid or expired.", "danger")
        return redirect(url_for("reset_password"))

    user = User.query.filter_by(email=email).first()
    if not user:
        flash("Invalid user for this reset link.", "danger")
        return redirect(url_for("reset_password"))

    if request.method == "POST":
        pw = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if pw != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("reset_password_token", token=token))
        if not strong_password(pw):
            flash("Password does not meet requirements.", "danger")
            return redirect(url_for("reset_password_token", token=token))

        # update user password
        hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
        user.password = hashed
        db.session.commit()
        flash("Password updated. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password_set.html", token=token, email=email)

# ----------------------------
# UPLOAD & DETECT
# ----------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method == "GET":
        return render_template("upload.html")

    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # read dataset
    try:
        df_raw = safe_read_csv(path)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {e}"}), 400

    # Convert + Predict
    try:
        df_converted = convert_to_webids23(df_raw)
        result_df = predict_uploaded_csv(df_converted)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    total = len(result_df)
    attacks = int((result_df["predicted_label"] == "Attack").sum())
    normal = int(total - attacks)

    # breakdown by multi_name (friendly names) if available
    breakdown = {}
    if "multi_name" in result_df.columns:
        breakdown = result_df["multi_name"].value_counts().to_dict()
    elif "multi" in result_df.columns:
        breakdown = result_df["multi"].astype(str).value_counts().to_dict()

    # build simple trend if timestamp-like exists
    trend = None
    ts_col = None
    for cand in ["ts", "timestamp", "time", "date"]:
        if cand in result_df.columns:
            ts_col = cand
            break
    if ts_col:
        try:
            temp = result_df.copy()
            temp[ts_col] = pd.to_datetime(temp[ts_col], errors="coerce")
            temp = temp.dropna(subset=[ts_col])
            temp["hour"] = temp[ts_col].dt.strftime("%Y-%m-%d %H:00")
            hourly = temp[temp["predicted_label"] == "Attack"].groupby("hour").size()
            trend = {"timestamps": list(hourly.index.tolist()), "attacks": list(hourly.values.tolist())}
        except Exception:
            trend = None

    # generate report (pdf or csv)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_basename = f"report_{timestamp}_{secure_filename(filename)}"
    pdf_path = os.path.join(REPORTS_FOLDER, report_basename + ".pdf")
    csv_path = os.path.join(REPORTS_FOLDER, report_basename + ".csv")
    report_file = None
    if REPORTLAB_AVAILABLE:
        rp = generate_pdf_report(breakdown, total, attacks, pdf_path)
        if rp:
            report_file = rp
    if report_file is None:
        # fallback to CSV with predictions
        try:
            result_df.to_csv(csv_path, index=False)
            report_file = csv_path
        except Exception:
            report_file = None

    # send email to logged in user with report
    try:
        user_email = session.get("user")
        if user_email and report_file:
            send_attack_alert(user_email, attacks, total, breakdown=breakdown, report_path=report_file)
    except Exception as e:
        print("Email/report send failed:", e)

    resp = {
        "total": total,
        "attack": attacks,
        "normal": normal,
        "method": "supervised",
        "breakdown": breakdown,
        "trend": trend,
        "report": os.path.basename(report_file) if report_file else None
    }
    return jsonify(resp)

@app.route("/download_report/<path:filename>")
def download_report(filename):
    full = os.path.join(REPORTS_FOLDER, filename)
    if not os.path.exists(full):
        return "Not found", 404
    return send_file(full, as_attachment=True)

# ----------------------------
# REALTIME ROUTES
# ----------------------------
@app.route("/realtime")
def realtime():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("realtime.html")


@app.route("/start_realtime")
def start_realtime():
    global realtime_running, realtime_thread

    print("🚀 /start_realtime CALLED")
    print("Using Interface:", realtime_interface)

    # Fix: ensure both variable names are set for sniffer
    realtime_running = True
    globals()["REALTIME_RUNNING"] = True   # <-- IMPORTANT FIX

    # If already running, do not start a new thread
    if realtime_thread and realtime_thread.is_alive():
        print("⚠️ Realtime thread already running")
        return jsonify({"status": "already_running"})

    print("🔥 Starting realtime sniffer thread...")

    realtime_thread = threading.Thread(target=run_realtime_ml_sniffer, daemon=True)
    realtime_thread.start()

    print("✅ Thread started successfully!")

    return jsonify({"status": "started"})



@app.route("/realtime_status")
@app.route("/realtime_status")
def realtime_status():
    global realtime_alerts, realtime_running

    formatted_alerts = []
    for a in realtime_alerts[-100:]:
        time_val = a.get("time") or a.get("timestamp") or datetime.now().strftime("%H:%M:%S")
        src_val = a.get("src") or a.get("src_ip") or "unknown"
        dst_val = a.get("dst") or a.get("dst_ip") or "unknown"
        type_val = a.get("type") or a.get("attack_type") or a.get("multi_name") or "Normal"
        sev_val = a.get("sev") or a.get("severity") or ("High" if a.get("binary",0)==1 else "Low")
        desc_val = a.get("desc") or a.get("description") or f"Predicted class: {type_val}"
        advice_val = a.get("advice") or "No action required."

        formatted_alerts.append({
            "time": time_val,
            "src": src_val,
            "dst": dst_val,
            "type": type_val,
            "sev": sev_val,
            "desc": desc_val,
            "advice": advice_val
        })

    return jsonify({
        "running": realtime_running,
        "alerts": formatted_alerts
    })

# ----------------------------
# REALTIME CHART IMAGE ROUTES
# ----------------------------
@app.route("/realtime_img/donut")
def rt_img_donut():
    attack = sum(a["sev"] in ["Medium", "High", "Critical"] for a in realtime_alerts)
    normal = len(realtime_alerts) - attack

    plt.figure(figsize=(4,4))
    plt.pie([normal, attack], labels=["Normal", "Attack"], autopct="%1.1f%%")
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    return send_file(img, mimetype="image/png")


@app.route("/realtime_img/line")
def rt_img_line():
    y = list(range(1, len(realtime_alerts) + 1))
    plt.figure(figsize=(5,2))
    plt.plot(y)
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    return send_file(img, mimetype="image/png")


@app.route("/realtime_img/bar")
def rt_img_bar():
    types = {}
    for a in realtime_alerts:
        t = a["type"]
        types[t] = types.get(t, 0) + 1

    if not types:
        types = {"No Data": 1}

    plt.figure(figsize=(5,2))
    plt.bar(types.keys(), types.values())
    plt.xticks(rotation=45)
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    return send_file(img, mimetype="image/png")

# ----------------------------
# START
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)