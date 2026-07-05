"""
daily_scan.py — weekly momentum model with signal-strength weights.

MODEL (backtested 2013-2026, after costs + 20% tax):
  Weekly rebalance, 15 holdings, score-weighted (3-12% per name),
  ATR x3 trailing stop, staged breadth exposure.
  ~25% CAGR backtested / high-teens realistic | max drawdown ~ -27%.

BEHAVIOUR:
  * WEEKLY rebalance: first run of a new ISO week -> BUY / SELL / KEEP lists.
  * Other days: only stop-loss exits ("act now") or "no action".
  * Names are ranked by signal strength; if you can't buy all 15, buy from the top.
  * WEIGHT column = relative signal strength (position size), NOT a probability.

Secrets needed: EMAIL_USER, EMAIL_PASS, EMAIL_TO
"""
import os, json, smtplib, ssl, time
from email.mime.text import MIMEText
import pandas as pd, numpy as np, yfinance as yf

SYMS = [l.strip() for l in open('symbols.txt') if l.strip()]
STATE_FILE = 'state.json'
N_FULL = 15
W_MIN, W_MAX = 0.03, 0.12          # per-name weight bounds
MIN_TURNOVER = 3e7                 # Rs.3cr/day liquidity floor (same in both universes)

def fetch():
    cl, hi, lo, vo = {}, {}, {}, {}
    for i in range(0, len(SYMS), 80):
        batch = [s + '.NS' for s in SYMS[i:i+80]]
        data = None
        for attempt in range(3):
            try:
                data = yf.download(batch, period='2y', interval='1d', auto_adjust=False,
                                   progress=False, group_by='ticker', threads=True)
                if data is not None and len(data) > 0:
                    break
            except Exception:
                pass
            time.sleep(20 * (attempt + 1))
        if data is None or len(data) == 0:
            continue
        for s in SYMS[i:i+80]:
            try:
                d = data[s + '.NS']
                if d['Close'].dropna().empty: continue
                cl[s], hi[s], lo[s], vo[s] = d['Close'], d['High'], d['Low'], d['Volume']
            except Exception:
                continue
    C = pd.DataFrame(cl).sort_index()
    return C, pd.DataFrame(hi).reindex_like(C), pd.DataFrame(lo).reindex_like(C), pd.DataFrame(vo).reindex_like(C)

def compute(C, H, L, V):
    sma50 = C.rolling(50).mean(); sma200 = C.rolling(200).mean()
    pc = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs()))
    atr22 = tr.rolling(22).mean(); hh22 = H.rolling(22).max()
    advol = (C * V).rolling(20).mean()
    mom = 0.5*(C/C.shift(252)-1) + 0.5*(C/C.shift(126)-1)
    vol126 = C.pct_change().rolling(126).std()
    RAM = mom / (vol126*np.sqrt(252) + 1e-9)
    comp = (1 + C.pct_change().mean(axis=1)).cumprod()
    regime = bool(comp.iloc[-1] > comp.rolling(200).mean().iloc[-1])
    breadth = float((C > sma200).iloc[-1].mean())
    if not regime or breadth < 0.30: exposure, nhold = 0.0, 0
    elif breadth < 0.40:             exposure, nhold = 0.5, N_FULL // 2
    else:                            exposure, nhold = 1.0, N_FULL
    price = C.iloc[-1]; stop_now = (hh22.iloc[-1] - 3*atr22.iloc[-1])
    elig = (advol.iloc[-1] >= MIN_TURNOVER) & (price > sma50.iloc[-1]) & (price > sma200.iloc[-1]) & RAM.iloc[-1].notna()
    scores = RAM.iloc[-1][elig].sort_values(ascending=False)
    asof = C.index[-1]
    return dict(asof=str(asof.date()), week=f"{asof.isocalendar().year}-W{asof.isocalendar().week:02d}",
                regime=regime, breadth=breadth, exposure=exposure, nhold=nhold,
                ranked=list(scores.index), scores=scores, price=price, stop_now=stop_now,
                nuniv=len(C.columns))

def weights_for(m, names):
    """Signal-strength weights: proportional to score, clipped to 3-12%, renormalised."""
    sc = np.array([max(float(m['scores'].get(s, 0.01)), 0.01) for s in names])
    w = sc / sc.sum()
    w = np.clip(w, W_MIN, W_MAX)
    return dict(zip(names, w / w.sum()))

def load_state():
    try: return json.load(open(STATE_FILE))
    except Exception: return None

def save_state(st):
    json.dump(st, open(STATE_FILE, 'w'), indent=2)

def main_logic(m, st):
    lines = [f"Momentum model — data as of {m['asof']}  ({m['nuniv']} stocks scanned)",
             f"Market: trend={'UP' if m['regime'] else 'DOWN'}, breadth={m['breadth']*100:.0f}%, "
             f"stance={'FULL' if m['exposure']==1 else 'HALF' if m['exposure']==0.5 else 'CASH'}", ""]
    target = m['ranked'][:m['nhold']]
    held = st['held'] if st else []
    stops = dict(st['stops']) if st else {}
    last_week = st.get('week') if st else None
    rebalance = (st is None) or (m['week'] != last_week) or (len(held) == 0 and m['nhold'] > 0)

    if rebalance:
        W = weights_for(m, target) if target else {}
        enter = [s for s in target if s not in held]
        exit_ = [s for s in held if s not in target]
        keep  = [s for s in held if s in target]
        new_stops = {}
        for s in target:
            cur = float(m['stop_now'].get(s, np.nan))
            prev = stops.get(s, cur)
            new_stops[s] = max(prev, cur) if not np.isnan(cur) else prev
        subject = f"WEEKLY REBALANCE {m['asof']} — buy {len(enter)}, sell {len(exit_)}"
        lines += [">>> WEEKLY REBALANCE — target list, STRONGEST FIRST.",
                  "    Can't buy all? Buy from the TOP DOWN. Weight = position size (signal strength, not a guarantee).", ""]
        lines.append(f"{'#':>2} {'STOCK':12}{'WEIGHT':>8}{'PRICE':>10}{'STOP':>10}{'ACTION':>8}")
        for i, s in enumerate(target, 1):
            act = "BUY" if s in enter else "KEEP"
            lines.append(f"{i:>2} {s:12}{W[s]*100:7.1f}%{m['price'][s]:10.1f}{new_stops[s]:10.1f}{act:>8}")
        lines.append("SELL (dropped out):" + ("  none" if not exit_ else ""))
        for s in exit_:
            lines.append(f"   - {s:12} at ~{float(m['price'].get(s, float('nan'))):.1f}")
        if m['exposure'] == 0.5:
            lines += ["", f"NOTE: HALF stance — hold only the top {m['nhold']}, keep the rest of your capital in cash."]
        elif m['exposure'] == 0.0:
            lines += ["", "NOTE: CASH stance — hold nothing this week."]
        new_state = dict(week=m['week'], held=target, stops=new_stops)
    else:
        triggered, surviving, new_stops = [], [], {}
        for s in held:
            cur = float(m['stop_now'].get(s, np.nan))
            trail = max(stops.get(s, cur), cur) if not np.isnan(cur) else stops.get(s, np.nan)
            px = float(m['price'].get(s, np.nan))
            new_stops[s] = trail
            if not np.isnan(px) and not np.isnan(trail) and px < trail:
                triggered.append((s, px, trail))
            else:
                surviving.append(s)
        if triggered:
            subject = f"STOP HIT {m['asof']} — SELL {len(triggered)} now"
            lines += [">>> ACT NOW — closed below stop, SELL today:", ""]
            for s, px, tr_ in triggered:
                lines.append(f"   ! {s:12} price {px:.1f}  <  stop {tr_:.1f}  -> SELL")
            lines += ["", "Still holding:"]
        else:
            subject = f"Scan {m['asof']} — no action"
            lines += ["No rebalance today, no stops hit. HOLD everything.", "", "Currently holding:"]
        for s in surviving:
            lines.append(f"   = {s:12}  stop {new_stops.get(s, float('nan')):.1f}")
        new_state = dict(week=last_week, held=surviving, stops={s: new_stops[s] for s in surviving})

    lines += ["", "Weekly rebalance; between times only sell on a stop.",
              "Weights are relative signal strength for position sizing — roughly half of picks",
              "still lose; the edge is the portfolio, not any single name.",
              "Not investment advice. Survivorship-biased backtest; realistic expectation is lower."]
    return subject, "\n".join(lines), new_state

def send(subject, body):
    u, p, to = os.environ['EMAIL_USER'], os.environ['EMAIL_PASS'], os.environ['EMAIL_TO']
    msg = MIMEText(body); msg['Subject'] = subject; msg['From'] = u; msg['To'] = to
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ssl.create_default_context()) as srv:
        srv.login(u, p); srv.sendmail(u, [to], msg.as_string())

if __name__ == '__main__':
    C, H, L, V = fetch()
    n_fetched = C.shape[1]
    n_with_history = int((C.notna().sum() >= 200).sum())
    if n_fetched < 0.65 * len(SYMS) or n_with_history < 0.7 * n_fetched:
        send("Stock scan — DATA PROBLEM (no signal today)",
             f"Fetched only {n_fetched}/{len(SYMS)} stocks; {n_with_history} with full history.\n"
             "Too incomplete to trust. NO ACTION today; state unchanged; retrying tomorrow.")
        raise SystemExit
    m = compute(C, H, L, V)
    subject, body, new_state = main_logic(m, load_state())
    if len(SYMS) > 1000:
        subject = "[FULL-NSE] " + subject
    save_state(new_state)
    send(subject, body)
    print(subject)
