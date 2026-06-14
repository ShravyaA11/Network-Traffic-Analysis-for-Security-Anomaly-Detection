import pandas as pd
import numpy as np

MODEL_FEATURES = [
    'flow_pkts_per_sec', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec',
    'flow_duration', 'bwd_header_size_tot', 'bwd_pkts_tot',
    'down_up_ratio', 'bwd_data_pkts_tot', 'flow_ack_flag_count',
    'fwd_last_window_size', 'fwd_data_pkts_tot', 'bwd_header_size_min',
    'bwd_psh_flag_count', 'flow_rst_flag_count', 'bwd_header_size_max',
    'payload_bytes_per_second', 'bwd_last_window_size',
    'fwd_header_size_tot', 'fwd_pkts_tot', 'fwd_psh_flag_count',
    'fwd_header_size_min', 'fwd_init_window_size', 'fwd_header_size_max',
    'bwd_init_window_size', 'flow_syn_flag_count', 'flow_fin_flag_count',
    'flow_cwr_flag_count', 'bwd_urg_flag_count', 'flow_ece_flag_count',
    'fwd_urg_flag_count'
]

def convert_to_webids23(df):
    df = df.copy()

    # Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Keep only numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Create fallback duration
    if "flow_duration" not in df.columns:
        df["flow_duration"] = 1  # assume 1 sec window

    df["flow_duration"] = df["flow_duration"].replace(0, 1)

    # Generate synthetic core packet features
    df["fwd_pkts_tot"] = df.get("fwd_pkts_tot", df.select_dtypes("number").sum(axis=1) % 10 + 1)
    df["bwd_pkts_tot"] = df.get("bwd_pkts_tot", (df.select_dtypes("number").sum(axis=1) // 10) % 10 + 1)

    df["flow_pkts_per_sec"] = (df["fwd_pkts_tot"] + df["bwd_pkts_tot"]) / df["flow_duration"]
    df["fwd_pkts_per_sec"] = df["fwd_pkts_tot"] / df["flow_duration"]
    df["bwd_pkts_per_sec"] = df["bwd_pkts_tot"] / df["flow_duration"]

    # Header sizes synthetic fallback
    df["fwd_header_size_tot"] = df.get("fwd_header_size_tot", df["fwd_pkts_tot"] * 20)
    df["bwd_header_size_tot"] = df.get("bwd_header_size_tot", df["bwd_pkts_tot"] * 20)

    df["fwd_header_size_min"] = df["fwd_header_size_tot"] // df["fwd_pkts_tot"]
    df["fwd_header_size_max"] = df["fwd_header_size_tot"]
    df["bwd_header_size_min"] = df["bwd_header_size_tot"] // df["bwd_pkts_tot"]
    df["bwd_header_size_max"] = df["bwd_header_size_tot"]

    df["payload_bytes_per_second"] = df.get("payload_bytes", df["fwd_pkts_tot"] * 8) / df["flow_duration"]

    # Flags (fallback = 0)
    flag_cols = [
        'flow_ack_flag_count', 'fwd_last_window_size', 'fwd_data_pkts_tot',
        'bwd_data_pkts_tot', 'bwd_psh_flag_count', 'flow_rst_flag_count',
        'bwd_last_window_size', 'fwd_psh_flag_count',
        'fwd_init_window_size', 'bwd_init_window_size', 'flow_syn_flag_count',
        'flow_fin_flag_count', 'flow_cwr_flag_count', 'bwd_urg_flag_count',
        'flow_ece_flag_count', 'fwd_urg_flag_count', 'down_up_ratio'
    ]

    for col in flag_cols:
        if col not in df.columns:
            df[col] = 0

    # Ensure all model features exist
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df_final = df[MODEL_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_final


if __name__ == "__main__":
    df = pd.read_csv("binary_dataset.csv")
    converted = convert_to_webids23(df)
    converted.to_csv("converted_dataset.csv", index=False)
    print("🎉 Converted dataset saved as converted_dataset.csv")
