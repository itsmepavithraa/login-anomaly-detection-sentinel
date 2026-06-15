"""
STEP 1 - Dataset Generation
============================
Since we're building a LOGIN ANOMALY DETECTION system, we generate a
realistic synthetic dataset that mimics real-world login behavior.

Why synthetic? Because:
- Real login datasets are private/confidential
- We can control exactly what "normal" and "attack" looks like
- This is the standard approach in academic cybersecurity research

Features we generate:
- hour_of_day       : When the login happened (0-23)
- login_duration    : How long the session lasted (seconds)
- failed_attempts   : Number of failed attempts before success
- country_code      : 0=home country, 1=foreign country
- device_known      : 1=known device, 0=new/unknown device
- typing_speed_wpm  : Words per minute (behavioral biometric)
- ip_reputation     : IP threat score (0=clean, 1=blacklisted)
- logins_last_24h   : How many logins in last 24 hours
- time_since_last   : Hours since last login
- label             : 0=normal, 1=attack/suspicious
"""

import pandas as pd
import numpy as np

np.random.seed(42)

def generate_dataset(n_normal=4000, n_attack=500):

    # ── NORMAL LOGIN BEHAVIOR ──────────────────────────────────────────────
    # Normal users log in during business hours, from known devices,
    # from their home country, with low failed attempts

    normal = {
        "hour_of_day"      : np.random.choice(
                                range(7, 22),             # 7am - 10pm
                                size=n_normal,
                                p=[0.02,0.05,0.10,0.12,0.12,0.12,
                                   0.10,0.09,0.08,0.07,0.05,0.04,
                                   0.02,0.01,0.01]
                             ),
        "login_duration"   : np.random.normal(1800, 400, n_normal).clip(60, 7200),
        "failed_attempts"  : np.random.choice([0,1,2], size=n_normal, p=[0.80,0.15,0.05]),
        "country_code"     : np.random.choice([0,1], size=n_normal, p=[0.97, 0.03]),
        "device_known"     : np.random.choice([1,0], size=n_normal, p=[0.95, 0.05]),
        "typing_speed_wpm" : np.random.normal(55, 10, n_normal).clip(20, 100),
        "ip_reputation"    : np.random.choice([0,1], size=n_normal, p=[0.99, 0.01]),
        "logins_last_24h"  : np.random.randint(1, 6, n_normal),
        "time_since_last"  : np.random.normal(12, 6, n_normal).clip(0.5, 48),
        "label"            : np.zeros(n_normal, dtype=int)
    }

    # ── ATTACK / SUSPICIOUS BEHAVIOR ──────────────────────────────────────
    # Attackers log in at odd hours, from unknown devices/countries,
    # with many failed attempts, unusual typing speed

    attack = {
        "hour_of_day"      : np.random.choice(
                                list(range(0, 6)) + list(range(22, 24)),
                                size=n_attack
                             ),
        "login_duration"   : np.random.normal(180, 100, n_attack).clip(10, 600),
        "failed_attempts"  : np.random.choice([3,4,5,6,7,8,9,10],
                                size=n_attack,
                                p=[0.25,0.20,0.15,0.12,0.10,0.08,0.06,0.04]
                             ),
        "country_code"     : np.random.choice([0,1], size=n_attack, p=[0.15, 0.85]),
        "device_known"     : np.random.choice([1,0], size=n_attack, p=[0.10, 0.90]),
        "typing_speed_wpm" : np.concatenate([
                                np.random.normal(15, 5, n_attack//2),   # bot: too slow
                                np.random.normal(120, 15, n_attack - n_attack//2)  # bot: too fast
                             ])[:n_attack].clip(1, 200),
        "ip_reputation"    : np.random.choice([0,1], size=n_attack, p=[0.30, 0.70]),
        "logins_last_24h"  : np.random.randint(10, 50, n_attack),
        "time_since_last"  : np.random.normal(0.5, 0.3, n_attack).clip(0.01, 3),
        "label"            : np.ones(n_attack, dtype=int)
    }

    df_normal = pd.DataFrame(normal)
    df_attack = pd.DataFrame(attack)

    df = pd.concat([df_normal, df_attack], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Round floats to 2 decimal places for clean output
    df["login_duration"]   = df["login_duration"].round(2)
    df["typing_speed_wpm"] = df["typing_speed_wpm"].round(2)
    df["time_since_last"]  = df["time_since_last"].round(2)

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("login_dataset.csv", index=False)

    print("=" * 50)
    print("  DATASET GENERATED: login_dataset.csv")
    print("=" * 50)
    print(f"\n  Total records  : {len(df)}")
    print(f"  Normal logins  : {(df.label == 0).sum()} ({(df.label==0).mean()*100:.1f}%)")
    print(f"  Attack logins  : {(df.label == 1).sum()} ({(df.label==1).mean()*100:.1f}%)")
    print(f"\n  Features ({len(df.columns)-1}):")
    for col in df.columns[:-1]:
        print(f"    - {col}")
    print(f"\n  Sample rows:")
    print(df.head(5).to_string(index=False))
    print("\n  ✓ Ready for Step 2\n")