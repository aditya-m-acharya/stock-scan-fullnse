"""
daily_scan_fullnse.py — full-NSE momentum scanner, WITH weak-market warnings.

Same model as the Nifty-750 scanner, but tuned for an experimental/observational
role on the wider ~2,000-stock universe. KEY DIFFERENCE: it always shows the
top-ranked candidates for visibility (so you can watch what the wider universe
is finding), but it makes the market-health status impossible to miss, and
explicitly tells you NOT to buy when the underlying market is weak.

Backtested result on full NSE (13y): ~14.6% CAGR / -43.5% max drawdown, i.e.
meaningfully WORSE and riskier than the Nifty-750 scanner (~21-25% / ~-25%).
Treat this scanner's picks as informational, not a trading instruction, unless
the market-health banner below says the coast is clear.

Secrets needed: EMAIL_USER, EMAIL_PASS, EMAIL_TO
"""
import os, json, smtplib, ssl, time
from email.mime.text import MIMEText
import pandas as pd, numpy as np, yfinance as yf

SYMS = [l.strip() for l in open('symbols.txt') if l.strip()]
STATE_FILE = 'state.json'
N_FULL = 15
W_MIN, W_MAX = 0.03, 0.12
MIN_TURNOVER = 3e7
BREADTH_CASH, BREADTH_HALF = 0.30, 0.40   # same tiers as the main model

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
    # market health verdict — independent of whether we show candidates
    if not regime or breadth < BREADTH_CASH:
        health, nhold, exposure = "WEAK", 0, 0.0
    elif breadth < BREADTH_HALF:
        health, nhold, exposure = "SHAKY", N_FULL // 2, 0.5
    else:
        health, nhold, exposure = "HEALTHY", N_FULL, 1.0
    price = C.iloc[-1]; stop_now = (hh22.iloc[-1] - 3*atr22.iloc[-1])
    elig = (advol.iloc[-1] >= MIN_TURNOVER) & (price > sma50.iloc[-1]) & (price > sma200.iloc[-1]) & RAM.iloc[-1].notna()
    scores = RAM.iloc[-1][elig].sort_values(ascending=False)
    asof = C.index[-1]
    return dict(asof=str(asof.date()), week=f"{asof.isocalendar().year}-W{asof.isocalendar().week:02d}",
                regime=regime, breadth=breadth, health=health, exposure=exposure, nhold=nhold,
                ranked=list(scores.index), scores=scores, price=price, stop_now=stop_now,
                nuniv=len(C.columns))

def weights_for(m, names):
    if not names: return {}
    sc = np.array([max(float(m['scores'].get(s, 0.01)), 0.01) for s in names])
    w = sc / sc.sum()
    w = np.clip(w, W_MIN, W_MAX)
    return dict(zip(names, w / w.sum()))

def load_state():
    try: return json.load(open(STATE_FILE))
    except Exception: return None

def save_state(st):
    json.dump(st, open(STATE_FILE, 'w'), indent=2)

def health_banner(m):
    """Loud, impossible-to-miss market-health block. Always at the very top."""
    L = []
    if m['health'] == "WEAK":
        L += ["=" * 60,
              "  ⚠️  WEAK MARKET — DO NOT BUY FROM THIS LIST  ⚠️",
              "=" * 60,
              f"  Breadth is only {m['breadth']*100:.0f}% (need 30%+ to consider buying, 40%+ to go full).",
              "  Fewer than a third of NSE stocks are in a healthy uptrend right now.",
              "  This is exactly the condition that historically preceded deep,",
              "  grinding drawdowns in the full-NSE backtest (-43% max drawdown).",
              "  The names below are shown FOR INFORMATION ONLY — they are what the",
              "  scanner is watching, NOT a buy list. Recommended action: hold cash.",
              "=" * 60, ""]
    elif m['health'] == "SHAKY":
        L += ["=" * 60,
              "  ⚠️  SHAKY MARKET — reduced size only  ⚠️",
              "=" * 60,
              f"  Breadth is {m['breadth']*100:.0f}% — moderate, not confirmed healthy.",
              f"  If you act at all, only buy the TOP {m['nhold']} below, rest stays cash.",
              "  Do not deploy full size in this condition.",
              "=" * 60, ""]
    else:
        L += ["=" * 60,
              "  ✅ HEALTHY MARKET — breadth confirms the trend",
              "=" * 60,
              f"  Breadth {m['breadth']*100:.0f}%, market trend up. Full-size candidates below.",
              "=" * 60, ""]
    return L

def main_logic(m, st):
    lines = [f"[FULL-NSE, EXPERIMENTAL] scan as of {m['asof']}  ({m['nuniv']} stocks scanned)", ""]
    lines += health_banner(m)

    # Always compute a display list (top 20 by rank) regardless of health,
    # so you can SEE what the wider universe is finding — even in a weak market.
    # Only the first N_FULL (or fewer, per breadth tier) are ever actionable.
    display_list = m['ranked'][:20]
    buy_count = m['nhold']                      # how many are ACTUALLY actionable
    W = weights_for(m, display_list[:buy_count]) if buy_count else {}

    held = st['held'] if st else []
    stops = dict(st['stops']) if st else {}
    last_week = st.get('week') if st else None
    rebalance = (st is None) or (m['week'] != last_week)

    if rebalance:
        target = display_list[:buy_count]       # only the actionable portion becomes "held"
        enter = [s for s in target if s not in held]
        exit_ = [s for s in held if s not in target]
        new_stops = {}
        for s in target:
            cur = float(m['stop_now'].get(s, np.nan))
            prev = stops.get(s, cur)
            new_stops[s] = max(prev, cur) if not np.isnan(cur) else prev
        subject = f"[FULL-NSE] {m['health']} {m['asof']} — {len(target)} actionable"
        lines.append(f"{'#':>2} {'STOCK':12}{'WEIGHT':>8}{'PRICE':>10}{'STOP':>10}{'STATUS':>10}")
        for i, s in enumerate(display_list, 1):
            actionable = s in target
            w_str = f"{W[s]*100:6.1f}%" if actionable else "   --  "
            status = "BUY" if (actionable and s in enter) else ("KEEP" if actionable else "watch-only")
            px = m['price'].get(s, float('nan'))
            stop_disp = new_stops.get(s, m['stop_now'].get(s, float('nan')))
            lines.append(f"{i:>2} {s:12}{w_str:>8}{px:10.1f}{stop_disp:10.1f} {status:>11}")
        if exit_:
            lines.append("\nSELL (dropped out):")
            for s in exit_:
                lines.append(f"   - {s:12} at ~{float(m['price'].get(s, float('nan'))):.1f}")
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
            subject = f"[FULL-NSE] STOP HIT {m['asof']} — SELL {len(triggered)} now"
            lines.append(">>> ACT NOW — closed below stop, SELL today:\n")
            for s, px, tr_ in triggered:
                lines.append(f"   ! {s:12} price {px:.1f}  <  stop {tr_:.1f}  -> SELL")
        else:
            subject = f"[FULL-NSE] {m['health']} {m['asof']} — no action"
            lines.append("No rebalance today, no stops hit.")
        if surviving:
            lines.append("\nCurrently holding:")
            for s in surviving:
                lines.append(f"   = {s:12}  stop {new_stops.get(s, float('nan')):.1f}")
        if not held:
            lines.append("\n(No live positions from this scanner right now.)")
            lines.append("Top 20 of the watchlist, for reference (NOT a buy list):")
            for i, s in enumerate(display_list[:20], 1):
                lines.append(f"   {i:>2}. {s:12} price {m['price'].get(s, float('nan')):.1f}")
        new_state = dict(week=last_week, held=surviving, stops={s: new_stops[s] for s in surviving})

    lines += ["", "-"*60,
              "This is the EXPERIMENTAL full-NSE scanner (wider, riskier universe).",
              "Backtest: ~14.6% CAGR / -43.5% max drawdown over 11y — meaningfully",
              "worse and more volatile than your main Nifty-750 model (~21-25% / ~-25%).",
              "Only act on 'BUY'/'KEEP' rows when the banner above says HEALTHY or SHAKY.",
              "'watch-only' rows are NOT recommendations. Not investment advice."]
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
        send("[FULL-NSE] Stock scan — DATA PROBLEM (no signal today)",
             f"Fetched only {n_fetched}/{len(SYMS)} stocks; {n_with_history} with full history.\n"
             "Too incomplete to trust. NO ACTION today; state unchanged; retrying tomorrow.")
        raise SystemExit
    m = compute(C, H, L, V)
    subject, body, new_state = main_logic(m, load_state())
    save_state(new_state)
    send(subject, body)
    print(subject)
