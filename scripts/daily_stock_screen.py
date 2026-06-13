#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, os, sys, traceback, urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


def now_tpe():
    return datetime.now(ZoneInfo("Asia/Taipei")) if ZoneInfo else datetime.utcnow() + timedelta(hours=8)


def ensure(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def pct(x):
    if x in ("", None):
        return ""
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


def num(x):
    if x in ("", None):
        return ""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def yahoo_symbol(code):
    code = str(code).strip()
    if "." in code:
        return code
    return f"{code}.SS" if code.startswith(("6", "9")) else f"{code}.SZ"


def fetch_quote(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=10d&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ai-hedge-fund-screen"})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            payload = json.loads(r.read().decode())
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not result:
            return {"price": None, "prev": None, "err": "empty result"}
        meta = result.get("meta", {}) or {}
        quote = ((result.get("indicators", {}) or {}).get("quote") or [{}])[0]
        closes = [x for x in quote.get("close", []) if isinstance(x, (int, float))]
        price = meta.get("regularMarketPrice")
        if not isinstance(price, (int, float)) and closes:
            price = closes[-1]
        prev = meta.get("previousClose")
        if not isinstance(prev, (int, float)) and len(closes) > 1:
            prev = closes[-2]
        return {"price": float(price) if isinstance(price, (int, float)) else None, "prev": float(prev) if isinstance(prev, (int, float)) else None, "err": None}
    except Exception as e:
        return {"price": None, "prev": None, "err": str(e)[:200]}


def run_aihf(cfg, candidates, dt):
    acfg = cfg.get("ai_hedge_fund", {}) or {}
    if str(acfg.get("enabled", True)).lower() in {"0", "false", "no"}:
        return {}, "aiHF disabled"
    if not any(os.getenv(k) for k in ("OPENAI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")):
        return {}, "aiHF skipped: missing LLM API key secret"
    if not os.getenv("FINANCIAL_DATASETS_API_KEY"):
        return {}, "aiHF skipped: missing FINANCIAL_DATASETS_API_KEY secret"

    tmap = {str(c["aihf_ticker"]): str(c["code"]) for c in candidates if c.get("aihf_ticker")}
    if not tmap:
        return {}, "aiHF skipped: no aihf_ticker mapping"
    try:
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from src.main import run_hedge_fund

        tickers = list(tmap)
        portfolio = {
            "cash": float(acfg.get("initial_cash", 1_000_000)),
            "margin_requirement": float(acfg.get("margin_requirement", 0.5)),
            "margin_used": 0.0,
            "positions": {t: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0} for t in tickers},
            "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers},
        }
        res = run_hedge_fund(
            tickers=tickers,
            start_date=(dt - timedelta(days=int(acfg.get("start_days_back", 120)))).date().isoformat(),
            end_date=dt.date().isoformat(),
            portfolio=portfolio,
            show_reasoning=False,
            selected_analysts=acfg.get("selected_analysts", ["technical_analyst", "fundamentals_analyst", "valuation_analyst", "sentiment_analyst"]),
            model_name=acfg.get("model_name", "gpt-4.1"),
            model_provider=acfg.get("model_provider", "OpenAI"),
        )
        out = {}
        for t, d in (res.get("decisions") or {}).items():
            if isinstance(d, dict):
                out[tmap.get(t, t)] = d
        return out, f"aiHF ran for {len(out)} tickers"
    except Exception as e:
        detail = "".join(traceback.format_exception_only(type(e), e)).strip()
        return {}, f"aiHF failed: {detail[:240]}"


def is_true(v):
    return bool(v) and str(v).lower() not in {"false", "0", "no", "none", "null"}


def evaluate(c, q, ai):
    price, prev = q.get("price"), q.get("prev")
    chg = None if price is None or not prev else price / prev - 1
    cur = float(c.get("current_position_pct") or 0)
    tgt = float(c.get("target_initial_pct") or 0)
    maxp = float(c.get("max_position_pct") or tgt)
    action, reason, target = "观察", "等待事件确认或价格触发", tgt

    rc = c.get("research_report_count_3m")
    if isinstance(rc, int) and rc > int(c.get("max_research_reports_3m", 15)):
        action, reason, target = ("熔断" if cur else "回避"), f"近3个月研报{rc}篇，超过阈值", 0
    elif price is not None and isinstance(c.get("stop_loss_price"), (int, float)) and price <= float(c["stop_loss_price"]):
        action, reason, target = ("熔断" if cur else "回避"), f"现价{price:.2f}低于硬止损{c['stop_loss_price']:.2f}", 0
    elif c.get("hard_avoid"):
        action, reason, target = ("熔断" if cur else "回避"), c.get("hard_avoid_reason", "配置为硬回避"), 0
    elif not is_true(c.get("event_confirmed")):
        action, reason, target = ("持有" if cur else "观察"), "核心事件尚未由公告/互动易/财报确认", cur
    else:
        in_zone = price is not None and isinstance(c.get("buy_zone_low"), (int, float)) and isinstance(c.get("buy_zone_high"), (int, float)) and c["buy_zone_low"] <= price <= c["buy_zone_high"]
        breakout = price is not None and isinstance(c.get("breakout_price"), (int, float)) and price >= c["breakout_price"]
        if cur <= 0 and (in_zone or breakout):
            action, reason, target = "建仓", "事件已确认且价格进入建仓触发区", tgt
        elif cur and price is not None and isinstance(c.get("take_profit_1_low"), (int, float)) and price >= c["take_profit_1_low"]:
            action, reason, target = "减仓", "触及第一止盈区", cur / 2
        elif cur and cur < maxp and breakout:
            action, reason, target = "加仓", "事件确认后突破关键价，允许向最高仓位靠拢", min(maxp, cur + tgt)
        elif cur:
            action, reason, target = "持有", "事件已确认但未触发加减仓", cur
        else:
            action, reason, target = "观察", "事件确认但价格未进入买点", cur

    if ai:
        a, conf = str(ai.get("action", "")).lower(), int(ai.get("confidence") or 0)
        if a in {"sell", "short"} and conf >= 60:
            action, reason, target = ("回避" if cur <= 0 else "减仓"), f"aiHF偏空/卖出，置信度{conf}", 0 if cur <= 0 else min(cur, target)
        elif a == "buy" and conf >= 70 and is_true(c.get("event_confirmed")) and action in {"观察", "持有"}:
            action, reason, target = ("建仓" if cur <= 0 else "加仓"), f"aiHF买入信号，置信度{conf}；仍需遵守事件和价格条件", tgt if cur <= 0 else min(maxp, cur + tgt)

    return {
        "code": c.get("code"), "name": c.get("name"), "price": price, "chg": chg,
        "ai_action": (ai or {}).get("action", ""), "ai_conf": (ai or {}).get("confidence", ""),
        "action": action, "target": target, "current": cur, "reason": reason,
        "bom": c.get("bom_link", ""), "buy": c.get("buy_trigger", ""), "stop": c.get("stop_rule", ""),
        "tp": c.get("take_profit_rule", ""), "short": c.get("short_thesis", ""),
        "src": "; ".join(c.get("source_urls", [])), "notes": c.get("notes", "")
    }


def col(n):
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def cell(r, c, v, st=0):
    if v is None:
        return ""
    ref, s = f"{col(c)}{r}", (f' s="{st}"' if st else "")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
        return f'<c r="{ref}"{s}><v>{v}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{s}><is><t>{escape(str(v))}</t></is></c>'


def sheet_xml(rows, widths=None):
    widths = widths or {}
    mx = max([len(r) for r in rows] or [1])
    cols = "<cols>" + "".join(f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>' for i, w in sorted(widths.items())) + "</cols>" if widths else ""
    body = []
    for ri, row in enumerate(rows, 1):
        body.append(f'<row r="{ri}">{"".join(cell(ri, ci, v, 1 if ri == 1 else 0) for ci, v in enumerate(row, 1))}</row>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><dimension ref="A1:{col(mx)}{len(rows)}"/>{cols}<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetData>{"".join(body)}</sheetData></worksheet>"""


def styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F2937"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>"""


def write_xlsx(path, sheets):
    ensure(path)
    wb_sheets, rels, overrides = [], [], ['<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>', '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    with ZipFile(path, "w", ZIP_DEFLATED) as z:
        for i, (name, rows, widths) in enumerate(sheets, 1):
            wb_sheets.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
            rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
            overrides.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
            z.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(rows, widths))
        rels.append(f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>' + "".join(overrides) + '</Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + "".join(wb_sheets) + '</sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + '</Relationships>')
        z.writestr("xl/styles.xml", styles_xml())


def read_logs(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def build(cfg, ev, meta, logs):
    dash = [["项目", "值", "说明"], ["运行时间", meta["run_time"], "Asia/Taipei"], ["运行时段", meta["session"], "09:00筛选；14:40复盘"], ["候选数", len(ev), ""], ["建仓/加仓建议", meta["buy_add_count"], ""], ["回避/熔断", meta["avoid_fuse_count"], ""], ["ai-hedge-fund状态", meta["aihf_status"], ""], ["交易约束", "不自动下单", "仅输出规划"]]
    watch = [["日期时间", "代码", "名称", "现价", "涨跌幅", "BOM/卡脖子环节", "aiHF动作", "aiHF置信度", "建议", "目标仓位", "当前仓位", "买入触发", "止损/熔断", "止盈", "空头证伪", "理由", "数据源"]]
    for e in ev:
        watch.append([meta["run_time"], e["code"], e["name"], num(e["price"]), pct(e["chg"]), e["bom"], e["ai_action"], e["ai_conf"], e["action"], pct(e["target"]), pct(e["current"]), e["buy"], e["stop"], e["tp"], e["short"], e["reason"], e["src"]])
    hold = [["代码", "名称", "当前仓位", "建议目标仓位", "动作", "执行区间/触发", "硬止损", "止盈/减仓", "下一步", "备注"]]
    for e in ev:
        hold.append([e["code"], e["name"], pct(e["current"]), pct(e["target"]), e["action"], e["buy"], e["stop"], e["tp"], e["reason"], e["notes"]])
    cats = [["代码", "名称", "事件", "最晚时限", "熔断阈值", "确认渠道", "状态"]]
    for c in cfg.get("candidates", []):
        for x in c.get("catalysts", []):
            cats.append([c.get("code"), c.get("name"), x.get("event"), x.get("deadline"), x.get("breaker"), x.get("channel"), x.get("status", "待确认")])
    params = [["参数", "阈值/规则", "说明"]] + [[k, json.dumps(v, ensure_ascii=False), "来自配置文件"] for k, v in (cfg.get("screen_filters", {}) or {}).items()]
    run = [["日期时间", "运行时段", "候选数", "建仓/加仓", "回避/熔断", "aiHF状态"]]
    for l in (logs[-49:] + [meta])[-50:]:
        run.append([l.get("run_time"), l.get("session"), l.get("candidate_count"), l.get("buy_add_count"), l.get("avoid_fuse_count"), l.get("aihf_status")])
    return [
        ("Dashboard", dash, {1: 20, 2: 45, 3: 55}),
        ("Watchlist", watch, {1: 20, 6: 32, 12: 40, 13: 36, 14: 36, 15: 45, 16: 40, 17: 50}),
        ("Holdings_Plan", hold, {1: 12, 2: 14, 6: 42, 7: 36, 8: 38, 9: 40, 10: 38}),
        ("Catalyst_Breakers", cats, {3: 60, 5: 60, 6: 30}),
        ("Run_Log", run, {1: 20, 6: 60}),
        ("Parameters", params, {1: 28, 2: 42, 3: 30}),
    ]


def summary(path, ev, meta):
    ensure(path)
    lines = ["# Daily ai-hedge-fund stock screen", "", f"- Run time: {meta['run_time']} Asia/Taipei", f"- Session: {meta['session']}", f"- ai-hedge-fund status: {meta['aihf_status']}", "", "|代码|名称|现价|aiHF|建议|目标仓位|理由|", "|---|---|---:|---|---|---:|---|"]
    for e in ev:
        ai = e["ai_action"] or "n/a"
        if e["ai_conf"] != "":
            ai += f"({e['ai_conf']})"
        lines.append(f"|{e['code']}|{e['name']}|{num(e['price'])}|{ai}|{e['action']}|{pct(e['target'])}|{str(e['reason']).replace('|','/')}|")
    lines += ["", "Hard rule: this report does not place orders; event confirmation, research-report count, and official disclosure checks remain hard gates.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--run-log", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    dt = now_tpe()
    sess = "09:00" if dt.hour < 12 else "14:40"
    sig, status = run_aihf(cfg, cfg.get("candidates", []), dt)
    ev = []
    for c in cfg.get("candidates", []):
        q = fetch_quote(c.get("market_symbol") or yahoo_symbol(c.get("code", "")))
        ev.append(evaluate(c, q, sig.get(str(c.get("code")))))
    meta = {"run_time": dt.strftime("%Y-%m-%d %H:%M:%S"), "session": sess, "candidate_count": len(ev), "buy_add_count": sum(e["action"] in {"建仓", "加仓"} for e in ev), "avoid_fuse_count": sum(e["action"] in {"回避", "熔断"} for e in ev), "aihf_status": status}
    logp = Path(a.run_log)
    logs = read_logs(logp)
    ensure(logp)
    with logp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    write_xlsx(Path(a.out), build(cfg, ev, meta, logs))
    summary(Path(a.summary), ev, meta)
    print(f"Wrote {a.out}")
    print(f"Wrote {a.summary}")
    print(status)


if __name__ == "__main__":
    main()
