#!/usr/bin/env python3
"""
S&P 500 Parquet 后处理脚本：Carry-Forward
==========================================
功能：
  1. 读取季度 Parquet，识别有效股票（排除幽灵股票）
  2. 为每只有效股票构建 JSON 表格（每天一行，所有指标为列）
  3. 通过查询 Parquet 填充 JSON，并处理周末/节假日的 carry-forward
  4. 输出处理后的 JSON + 修正后的 Parquet

用法：
  python sp500_carryforward.py Q2012Q3.parquet
  python sp500_carryforward.py Q2012Q3.parquet Q2013Q1.parquet
  python sp500_carryforward.py *.parquet --json-dir ./output_json
"""

import sys, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  列分类规则 —— 决定每列在非交易日如何处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ① 标识列：不参与 carry-forward
ID_COLS = {"Date", "Ticker", "Sector", "SubIndustry"}

# ② 事件列：非交易日保持默认值（不 ffill，因为事件是点状的）
EVENT_DEFAULTS = {
    "Div_Amount": 0.0,
    "Split_Ratio": 1.0,
    "Insider_Filings": 0.0,
}

# ③ 日内变化列：周末 ffill（carry-on 周五的值）
#    虽然周末没有交易，但为了 ML 训练时表格完整性，用 ffill 比 NaN 或随机填充更合理
#    如果想改回 NaN 模式，把这三列加回 DAILY_CHANGE_COLS 集合即可
DAILY_CHANGE_COLS = set()   # 空集 = 全部 ffill
# DAILY_CHANGE_COLS = {"Return", "Log_Return", "Amplitude"}  # ← 取消注释可恢复 NaN 模式

# ④ 市场广度列：非交易日的 B_Adv=0, B_Dec=0, B_Count=N 是 P_assemble
#    计算的伪值（因为所有Return=NaN），应该用最近交易日的值替换
BREADTH_COLS = {"B_Adv", "B_Dec", "B_Count", "B_AD_Ratio", "B_NewHi", "B_NewLo"}

# ⑤ 所有其他列：forward-fill（包括价格、技术指标、基本面、系统数据等）
#    这些都是"水平值"，在周末保持不变是合理的


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 1: 识别有效股票
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def identify_valid_tickers(df):
    """
    有效股票 = 在该季度至少有 1 天 Close 不为 NaN
    幽灵股票 = Close 全为 NaN（虽然在 active 列表中，但 yfinance 没有数据）
    幽灵股票的特征：除了系统级列和事件默认值外，所有列都是 NaN
    """
    close_by_ticker = df.groupby("Ticker")["Close"].apply(lambda x: x.notna().any())
    valid = sorted(close_by_ticker[close_by_ticker].index.tolist())
    ghost = sorted(close_by_ticker[~close_by_ticker].index.tolist())
    return valid, ghost


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 2: 构建单股票 JSON 模板并从 Parquet 查询填充
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_ticker_table(df, ticker, all_dates, data_cols):
    """
    为一只股票构建完整的日期×指标表格，通过查询 Parquet 填充内容。

    逻辑：
    1. 创建空模板：所有日期 × 所有数据列
    2. 查询 Parquet 中该 ticker 的数据
    3. 按日期逐行匹配填充（不是直接赋值，而是 query→fill）
    4. 标记每天是星期几、是否交易日
    """
    # 创建空模板
    template = pd.DataFrame({"Date": all_dates})
    template["Ticker"] = ticker
    template["day_of_week"] = template["Date"].dt.dayofweek  # 0=Mon, 6=Sun
    template["day_name"] = template["Date"].dt.day_name()

    # 从 Parquet 查询该股票的所有行
    ticker_data = df.loc[df["Ticker"] == ticker, ["Date"] + data_cols].copy()
    ticker_data["Date"] = pd.to_datetime(ticker_data["Date"])

    # 判断交易日：该股票在该日有 Close 数据
    trading_dates = set(ticker_data.loc[ticker_data["Close"].notna(), "Date"].tolist())
    template["is_trading_day"] = template["Date"].isin(trading_dates)

    # 按日期查询填充 —— 用 merge 实现 "对每一天去 Parquet 里搜索"
    result = template.merge(ticker_data, on="Date", how="left", suffixes=("", "_src"))

    # 清理：如果 merge 产生了 Ticker 重复列
    if "Ticker_src" in result.columns:
        result = result.drop(columns=["Ticker_src"])

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 3: Carry-Forward 处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_carry_forward(ticker_df, data_cols):
    """
    对单只股票的时间序列应用 carry-forward 规则。

    核心逻辑：
    ┌──────────────────┬──────────────────────────────────────────┐
    │ 列类型           │ 非交易日处理                               │
    ├──────────────────┼──────────────────────────────────────────┤
    │ 事件列           │ 保持默认值 (Div=0, Split=1, Insider=0)     │
    │ 日内变化列       │ 保持 NaN（周末没有交易，不应伪造变化）       │
    │ 市场广度列       │ ffill（替换掉周末计算出的伪值 0）           │
    │ 所有其他列       │ ffill（价格、技术、基本面、系统数据等）      │
    └──────────────────┴──────────────────────────────────────────┘

    "所有其他列" 包括：
    - 价格：Open/High/Low/Close/Adj_Close/Volume → 周末沿用周五值
    - 技术指标：MA_*, Vol_*d, Mom_*, RSI, Beta 等 → 周末不变
    - 基本面：Revenue, NetIncome 等 → 本来就是季度 carry-forward
    - 系统数据：利率、汇率、指数、商品、FF因子 → 周末沿用周五值
    - 估值/比率：PE, ROE 等 → 周末不变
    """
    df = ticker_df.sort_values("Date").copy()

    # 分类列
    event_cols_present = [c for c in EVENT_DEFAULTS if c in df.columns]
    change_cols_present = [c for c in DAILY_CHANGE_COLS if c in df.columns]
    breadth_cols_present = [c for c in BREADTH_COLS if c in df.columns]

    ffill_cols = [
        c for c in data_cols
        if c not in ID_COLS
        and c not in EVENT_DEFAULTS
        and c not in DAILY_CHANGE_COLS
        # 市场广度列也要 ffill，但需要先清理伪值
    ]

    # ---- 处理市场广度列：先把非交易日的伪值 (B_Adv=0 等) 设为 NaN ----
    non_trading_mask = ~df["is_trading_day"]
    for col in breadth_cols_present:
        df.loc[non_trading_mask, col] = np.nan

    # ---- Forward-fill 所有水平列（包括已清理的广度列）----
    if ffill_cols:
        df[ffill_cols] = df[ffill_cols].ffill()

    # ---- 事件列：确保非交易日是默认值，不被 ffill 污染 ----
    for col, default in EVENT_DEFAULTS.items():
        if col in df.columns:
            df.loc[non_trading_mask & df[col].isna(), col] = default

    # ---- 日内变化列：当前设为 ffill 模式（ML兼容）----
    # 如果 DAILY_CHANGE_COLS 非空，这些列会在非交易日保持 NaN

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 4: 导出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def export_ticker_json(ticker_df, ticker, quarter, sector, sub_industry, output_dir):
    """导出单只股票的 JSON 文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建 JSON 结构
    records = []
    for _, row in ticker_df.iterrows():
        record = {"Date": row["Date"].strftime("%Y-%m-%d")}
        record["day_of_week"] = row["day_name"]
        record["is_trading_day"] = bool(row["is_trading_day"])
        for col in ticker_df.columns:
            if col in ("Date", "Ticker", "day_of_week", "day_name", "is_trading_day"):
                continue
            val = row[col]
            if pd.isna(val):
                record[col] = None
            elif isinstance(val, (np.integer,)):
                record[col] = int(val)
            elif isinstance(val, (np.floating,)):
                record[col] = round(float(val), 6)
            else:
                record[col] = val
        records.append(record)

    result = {
        "ticker": ticker,
        "quarter": quarter,
        "sector": sector if pd.notna(sector) and sector else "Unknown",
        "sub_industry": sub_industry if pd.notna(sub_industry) and sub_industry else "Unknown",
        "total_days": len(records),
        "trading_days": sum(1 for r in records if r["is_trading_day"]),
        "weekend_days": sum(1 for r in records if r["day_of_week"] in ("Saturday", "Sunday")),
        "data": records,
    }

    fp = output_dir / f"{quarter}_{ticker}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return fp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_one_parquet(parquet_path, json_dir=None, export_json_tickers=None,
                        export_parquet=True, verbose=True):
    """
    处理单个季度 Parquet 文件。

    Args:
        parquet_path: 输入的 parquet 文件路径
        json_dir: JSON 输出目录（None = 不导出 JSON）
        export_json_tickers: 导出 JSON 的 ticker 列表（None = 全部导出）
        export_parquet: 是否输出修正后的 parquet
        verbose: 是否打印详细信息
    """
    path = Path(parquet_path)
    quarter = path.stem  # e.g. "Q2012Q3"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  处理: {path.name}")
        print(f"{'='*60}")

    # ---- 读取 Parquet ----
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    all_cols = [c for c in df.columns if c not in ID_COLS]
    data_cols = [c for c in df.columns if c != "Date" and c != "Ticker"]

    if verbose:
        print(f"  原始: {len(df):,} 行, {df['Ticker'].nunique()} tickers, "
              f"{df['Date'].nunique()} 天")

    # ---- STEP 1: 识别有效/幽灵股票 ----
    valid_tickers, ghost_tickers = identify_valid_tickers(df)
    if verbose:
        print(f"  有效股票: {len(valid_tickers)}, 幽灵股票: {len(ghost_tickers)}")

    # ---- 获取完整日期序列和 Sector 映射 ----
    all_dates = sorted(df["Date"].unique())
    sector_map = (
        df.drop_duplicates("Ticker")
        .set_index("Ticker")[["Sector", "SubIndustry"]]
        .to_dict("index")
    )

    # ---- STEP 2 & 3: 逐股票构建表格 + carry-forward ----
    processed_frames = []
    total = len(valid_tickers)

    for i, ticker in enumerate(valid_tickers):
        # Step 2: 构建表格，从 Parquet 查询填充
        ticker_table = build_ticker_table(df, ticker, all_dates, data_cols)

        # 填充 Sector/SubIndustry（标识信息）
        info = sector_map.get(ticker, {})
        ticker_table["Sector"] = info.get("Sector", "")
        ticker_table["SubIndustry"] = info.get("SubIndustry", "")

        # Step 3: apply carry-forward
        ticker_table = apply_carry_forward(ticker_table, data_cols)

        # Step 4: 可选导出 JSON
        if json_dir and (export_json_tickers is None or ticker in export_json_tickers):
            export_ticker_json(
                ticker_table, ticker, quarter,
                info.get("Sector", ""), info.get("SubIndustry", ""),
                json_dir
            )

        # 收集处理后的数据（用于输出 parquet）
        # 只保留原始列（去掉辅助列）
        keep_cols = [c for c in df.columns if c in ticker_table.columns]
        processed_frames.append(ticker_table[keep_cols])

        if verbose and (i + 1) % 100 == 0:
            print(f"    进度: {i+1}/{total} tickers")

    if verbose:
        print(f"    进度: {total}/{total} tickers - 完成")

    # ---- 合并幽灵股票行（保留但不处理）----
    ghost_rows = df[df["Ticker"].isin(ghost_tickers)].copy()

    # ---- 合并最终结果 ----
    final = pd.concat(processed_frames + [ghost_rows], ignore_index=True)
    final = final.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # ---- 输出修正后的 Parquet ----
    if export_parquet:
        out_path = Path(".") / f"{quarter}_fixed.parquet"
        final.to_parquet(out_path, index=False, engine="pyarrow")
        if verbose:
            print(f"\n  输出: {out_path.name}")

    # ---- 质量报告 ----
    if verbose:
        print_quality_report(df, final, valid_tickers)

    return final


def print_quality_report(original, fixed, valid_tickers):
    """对比修复前后的数据质量"""
    print(f"\n  {'─'*50}")
    print(f"  修复前后对比（仅有效股票，排除幽灵）")
    print(f"  {'─'*50}")

    orig_valid = original[original["Ticker"].isin(valid_tickers)]
    fix_valid = fixed[fixed["Ticker"].isin(valid_tickers)]

    # 整体填充率
    skip = {"Date", "Ticker", "Sector", "SubIndustry"}
    check_cols = [c for c in original.columns if c not in skip and c in fixed.columns]

    orig_fill = orig_valid[check_cols].notna().mean().mean() * 100
    fix_fill = fix_valid[check_cols].notna().mean().mean() * 100
    print(f"  整体填充率: {orig_fill:.1f}% → {fix_fill:.1f}% (+{fix_fill-orig_fill:.1f}%)")

    # 分类别对比
    categories = {
        "价格": ["Close", "Open", "High", "Low", "Adj_Close", "Volume"],
        "系统数据": ["Fed_Funds_Rate", "Treasury_10Y", "SP500", "VIX", "Gold",
                   "FF_MktRF", "CPI"],
        "基本面": ["Revenue", "NetIncome", "Total_Assets", "Mkt_Cap", "PE"],
        "技术指标": ["Return", "Vol_20d", "Mom_1m", "RSI_14d", "Beta"],
        "市场广度": ["B_Adv", "B_Dec", "B_Count"],
    }
    for cat, cols in categories.items():
        present = [c for c in cols if c in check_cols]
        if not present:
            continue
        o = orig_valid[present].notna().mean().mean() * 100
        f = fix_valid[present].notna().mean().mean() * 100
        delta = f - o
        arrow = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        print(f"  {cat:<12}: {o:>5.1f}% → {f:>5.1f}% ({arrow})")

    # 周末填充验证
    fix_weekend = fix_valid[fix_valid["Date"].dt.dayofweek >= 5]
    if len(fix_weekend) > 0:
        wk_close = fix_weekend["Close"].notna().mean() * 100
        wk_sys = fix_weekend["Fed_Funds_Rate"].notna().mean() * 100 if "Fed_Funds_Rate" in fix_weekend.columns else 0
        print(f"\n  周末行验证:")
        print(f"    Close 填充率:          {wk_close:.1f}%")
        print(f"    Fed_Funds_Rate 填充率: {wk_sys:.1f}%")
        wk_return = fix_weekend['Return'].notna().mean() * 100 if "Return" in fix_weekend.columns else 0
        print(f"    Return 填充率 (ffill):  {wk_return:.1f}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="S&P 500 Parquet 后处理：Carry-Forward",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件，只输出修正 parquet
  python sp500_carryforward.py Q2012Q3.parquet

  # 处理并导出所有股票的 JSON
  python sp500_carryforward.py Q2012Q3.parquet --json-dir ./output

  # 只导出指定股票的 JSON（用于检查）
  python sp500_carryforward.py Q2012Q3.parquet --json-dir ./output --tickers AAPL MSFT AMZN
        """,
    )
    parser.add_argument("files", nargs="+", help="季度 Parquet 文件")
    parser.add_argument("--json-dir", default=None, help="JSON 输出目录")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="只导出这些 ticker 的 JSON（默认全部）")
    parser.add_argument("--no-parquet", action="store_true",
                        help="不输出修正后的 parquet")

    args = parser.parse_args()

    for filepath in args.files:
        process_one_parquet(
            filepath,
            json_dir=args.json_dir,
            export_json_tickers=args.tickers,
            export_parquet=not args.no_parquet,
        )

    print(f"\n完成。")
