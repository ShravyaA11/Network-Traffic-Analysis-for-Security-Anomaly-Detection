import os, argparse, time, threading, joblib, math, smtplib, traceback
from email.message import EmailMessage
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scapy.all import sniff, IP, TCP, UDP, Raw

try:
    import winsound
    HAS_SOUND = True
except:
    HAS_SOUND = False

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "realtime_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ALERT_CSV = os.path.join(OUT_DIR, "alerts.csv")

DEFAULT_FEATURES = [
    "flow_pkts_per_sec","fwd_pkts_per_sec","bwd_pkts_per_sec","flow_duration",
    "bwd_header_size_tot","bwd_pkts_tot","down_up_ratio","bwd_data_pkts_tot",
    "flow_ack_flag_count","fwd_last_window_size","fwd_data_pkts_tot","bwd_header_size_min",
    "bwd_psh_flag_count","flow_rst_flag_count","bwd_header_size_max","payload_bytes_per_second",
    "bwd_last_window_size","fwd_header_size_tot","fwd_pkts_tot","fwd_psh_flag_count",
    "fwd_header_size_min","fwd_init_window_size","fwd_header_size_max","bwd_init_window_size",
    "flow_syn_flag_count","flow_fin_flag_count","flow_cwr_flag_count","bwd_urg_flag_count",
    "flow_ece_flag_count","fwd_urg_flag_count"
]

def play_sound():
    try:
        if HAS_SOUND:
            winsound.Beep(1200, 200)
    except:
        pass

def send_mail(srv, port, user, pwd, to, subject, body, file=None):
    try:
        msg = EmailMessage()
        msg["From"] = user
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)

        if file and os.path.exists(file):
            with open(file, "rb") as f:
                msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=os.path.basename(file))

        with smtplib.SMTP_SSL(srv, port) as s:
            s.login(user, pwd)
            s.send_message(msg)
    except:
        pass

def load_models():
    m = {}
    try: m["binary"] = joblib.load(os.path.join(MODEL_DIR, "rf_binary.pkl"))
    except: m["binary"] = None
    try: m["multi"] = joblib.load(os.path.join(MODEL_DIR, "rf_multiclass.pkl"))
    except: m["multi"] = None
    try: m["scaler"] = joblib.load(os.path.join(MODEL_DIR, "supervised_scaler.pkl"))
    except: m["scaler"] = None
    try: m["features"] = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))
    except: m["features"] = DEFAULT_FEATURES
    return m

def pkt_key(pkt):
    try:
        if IP in pkt:
            ip = pkt[IP]
            sport = None
            dport = None
            proto = None
            if TCP in pkt:
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
                proto = "TCP"
            elif UDP in pkt:
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport
                proto = "UDP"
            else:
                proto = str(ip.proto)
            return (ip.src, ip.dst, sport, dport, proto)
    except:
        pass
    return None

def rev_key(t): 
    if t: 
        s,d,sp,dp,p = t
        return (d,s,dp,sp,p)
    return None

class Agg:
    def __init__(self):
        self.flows = defaultdict(lambda: {
            "first": None, "last": None,
            "fwd_p":0,"bwd_p":0,"fwd_b":0,"bwd_b":0,
            "fwd_h":0,"bwd_h":0,
            "fwd_psh":0,"bwd_psh":0,"fwd_urg":0,"bwd_urg":0,
            "ack":0,"syn":0,"fin":0,"rst":0,"ece":0,"cwr":0
        })
        self.lock = threading.Lock()

    def add(self, pkt):
        k = pkt_key(pkt)
        if not k: return
        rk = rev_key(k)
        ts = float(pkt.time)
        try:
            pl = len(pkt[Raw].load) if Raw in pkt else 0
        except: pl = 0
        try:
            hl = len(bytes(pkt)) - pl
        except: hl = 0

        with self.lock:
            if k in self.flows:
                f = self.flows[k]
                if f["first"] is None: f["first"] = ts
                f["last"] = ts
                f["fwd_p"] += 1
                f["fwd_b"] += pl
                f["fwd_h"] += hl
                if TCP in pkt:
                    fl = pkt[TCP].flags
                    if fl & 0x10: f["ack"]+=1
                    if fl & 0x02: f["syn"]+=1
                    if fl & 0x01: f["fin"]+=1
                    if fl & 0x04: f["rst"]+=1
                    if fl & 0x08: f["fwd_psh"]+=1
                    if fl & 0x20: f["fwd_urg"]+=1
                    if fl & 0x40: f["ece"]+=1
                    if fl & 0x80: f["cwr"]+=1
            elif rk in self.flows:
                f = self.flows[rk]
                if f["first"] is None: f["first"] = ts
                f["last"] = ts
                f["bwd_p"] += 1
                f["bwd_b"] += pl
                f["bwd_h"] += hl
                if TCP in pkt:
                    fl = pkt[TCP].flags
                    if fl & 0x10: f["ack"]+=1
                    if fl & 0x02: f["syn"]+=1
                    if fl & 0x01: f["fin"]+=1
                    if fl & 0x04: f["rst"]+=1
                    if fl & 0x08: f["bwd_psh"]+=1
                    if fl & 0x20: f["bwd_urg"]+=1
                    if fl & 0x40: f["ece"]+=1
                    if fl & 0x80: f["cwr"]+=1
            else:
                f = self.flows[k]
                f["first"] = ts
                f["last"] = ts
                f["fwd_p"] = 1
                f["fwd_b"] = pl
                f["fwd_h"] = hl

    def snap(self):
        with self.lock:
            c = dict(self.flows)
            self.flows = defaultdict(lambda: {
                "first": None,"last":None,
                "fwd_p":0,"bwd_p":0,"fwd_b":0,"bwd_b":0,
                "fwd_h":0,"bwd_h":0,
                "fwd_psh":0,"bwd_psh":0,"fwd_urg":0,"bwd_urg":0,
                "ack":0,"syn":0,"fin":0,"rst":0,"ece":0,"cwr":0
            })
            return c

def build_df(flows, feats):
    rows = []
    for k,f in flows.items():
        src,dst,sp,dp,p = k
        dur = (f["last"]-f["first"]) if f["first"] and f["last"] else 0.001
        row = {
            "flow_duration": dur,
            "fwd_pkts_tot": f["fwd_p"],
            "bwd_pkts_tot": f["bwd_p"],
            "fwd_data_pkts_tot": f["fwd_b"],
            "bwd_data_pkts_tot": f["bwd_b"],
            "flow_pkts_per_sec": (f["fwd_p"]+f["bwd_p"])/dur,
            "fwd_pkts_per_sec": f["fwd_p"]/dur,
            "bwd_pkts_per_sec": f["bwd_p"]/dur,
            "payload_bytes_per_second": (f["fwd_b"]+f["bwd_b"])/dur,
            "fwd_header_size_tot": f["fwd_h"],
            "bwd_header_size_tot": f["bwd_h"],
            "fwd_psh_flag_count": f["fwd_psh"],
            "bwd_psh_flag_count": f["bwd_psh"],
            "fwd_urg_flag_count": f["fwd_urg"],
            "bwd_urg_flag_count": f["bwd_urg"],
            "flow_ack_flag_count": f["ack"],
            "flow_syn_flag_count": f["syn"],
            "flow_fin_flag_count": f["fin"],
            "flow_rst_flag_count": f["rst"],
            "flow_ece_flag_count": f["ece"],
            "flow_cwr_flag_count": f["cwr"],
            "down_up_ratio": (f["fwd_b"]+1)/(f["bwd_b"]+1),
            "src_ip": src,
            "dst_ip": dst
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=feats+["src_ip","dst_ip"])

    df = pd.DataFrame(rows)
    for f in feats:
        if f not in df.columns: df[f] = 0
    return df[feats+["src_ip","dst_ip"]]

def save_charts(history, breakdown):
    try:
        if history:
            ts,att,total = history[-1]
            norm = total - att
            plt.figure(figsize=(4,4))
            plt.pie([att,norm],labels=["Attack","Normal"],autopct="%1.0f%%")
            plt.savefig(os.path.join(OUT_DIR,"donut.png"))
            plt.close()

        if breakdown:
            items = sorted(breakdown.items(),key=lambda x:x[1],reverse=True)[:10]
            labs = [i[0] for i in items]
            vals = [i[1] for i in items]
            plt.figure(figsize=(6,3))
            plt.bar(labs,vals)
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(OUT_DIR,"bar.png"))
            plt.close()

        xs=[h[0] for h in history]
        ys=[h[1] for h in history]
        plt.figure(figsize=(6,3))
        plt.plot(xs,ys,marker="o")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(OUT_DIR,"line.png"))
        plt.close()
    except:
        pass

def loop(iface, win, email, smtp):
    models = load_models()
    feats = models["features"]
    scaler = models["scaler"]
    bmod = models["binary"]
    mmod = models["multi"]

    agg = Agg()
    history = deque(maxlen=50)
    breakdown = defaultdict(int)

    def handler(pkt):
        agg.add(pkt)

    t = threading.Thread(target=lambda: sniff(iface=iface, prn=handler, store=False))
    t.daemon=True
    t.start()

    while True:
        time.sleep(win)
        flows = agg.snap()
        ts = datetime.utcnow().strftime("%H:%M:%S")
        df = build_df(flows,feats)
        if df.empty:
            history.append((ts,0,0))
            save_charts(history,breakdown)
            continue

        X = df[feats].values
        try:
            if scaler is not None: X = scaler.transform(X)
        except: pass

        try:
            bp = bmod.predict(X) if bmod else np.zeros(len(X))
            mp = mmod.predict(X) if mmod else np.zeros(len(X))
        except:
            bp = np.zeros(len(X))
            mp = np.zeros(len(X))

        df["bin"] = bp
        df["multi"] = mp

        atk = int((df["bin"]==1).sum())
        tot = len(df)
        history.append((ts,atk,tot))

        for k,v in df[df["bin"]==1]["multi"].value_counts().items():
            breakdown[str(k)] += int(v)

        win_csv = os.path.join(OUT_DIR, f"win_{datetime.utcnow().strftime('%H%M%S')}.csv")
        df.to_csv(win_csv,index=False)

        if atk>0:
            play_sound()
            if not os.path.exists(ALERT_CSV):
                df[df["bin"]==1].to_csv(ALERT_CSV,index=False)
            else:
                df[df["bin"]==1].to_csv(ALERT_CSV,mode="a",header=False,index=False)

            if email and smtp:
                subj=f"Realtime IDS Attack Alert ({atk})"
                body=f"{atk} attacks detected out of {tot} flows in last {win}s"
                send_mail(smtp["server"],smtp["port"],smtp["user"],smtp["pass"],smtp["to"],subj,body,win_csv)

        save_charts(history,breakdown)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--iface",required=True)
    p.add_argument("--window",type=int,default=5)
    p.add_argument("--email",action="store_true")
    return p.parse_args()

if __name__=="__main__":
    a=args()
    smtp=None
    if a.email:
        smtp={
            "server":os.environ.get("ALERT_SMTP_SERVER","smtp.gmail.com"),
            "port":int(os.environ.get("ALERT_SMTP_PORT",465)),
            "user":os.environ.get("ALERT_SMTP_USER"),
            "pass":os.environ.get("ALERT_SMTP_PASS"),
            "to":os.environ.get("ALERT_TO_EMAIL")
        }
    loop(a.iface,a.window,a.email,smtp)