#!/usr/bin/env python3
"""
S&P 500 Quantitative Data Pipeline
====================================
python sp500_pipeline.py run
python sp500_pipeline.py load
python sp500_pipeline.py process
python sp500_pipeline.py eval
"""

import os, sys, time, json, logging, argparse, warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")
# 静默 yfinance 的 ERROR 输出
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

START_DATE = "2000-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
OUT  = Path("sp500_data")
CACHE = Path("sp500_data/_cache")
ENRICHED = CACHE / "enriched"
CONFIG_FILE = Path("sp500_config.json")

# 保守速率: 宁可慢一点也不丢数据
SEC_RATE   = 0.15   # SEC限10req/s, 我们用~6.6req/s
YF_BATCH   = 40     # 每批40只, 减少超时风险
YF_SLEEP   = 2.0    # 批间2秒
FRED_SLEEP = 0.2    # FRED每请求0.2秒

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

FRED_SERIES = {
    "DFF":"Fed_Funds_Rate","DGS1MO":"Treasury_1M","DGS3MO":"Treasury_3M","DGS6MO":"Treasury_6M",
    "DGS1":"Treasury_1Y","DGS2":"Treasury_2Y","DGS5":"Treasury_5Y","DGS7":"Treasury_7Y",
    "DGS10":"Treasury_10Y","DGS20":"Treasury_20Y","DGS30":"Treasury_30Y",
    "T10Y2Y":"Term_Spread_10Y2Y","T10Y3M":"Term_Spread_10Y3M",
    "DFII10":"TIPS_10Y_Real","T10YIE":"Breakeven_10Y","T5YIE":"Breakeven_5Y",
    "BAMLC0A0CM":"IG_Spread","BAMLH0A0HYM2":"HY_Spread","TEDRATE":"TED_Spread",
    "DTWEXBGS":"USD_Broad","DEXUSEU":"USDEUR","DEXJPUS":"JPYUSD","DEXUSUK":"USDGBP","DEXCHUS":"CNYUSD",
    "VIXCLS":"VIX_FRED",
    "CPIAUCSL":"CPI","CPILFESL":"Core_CPI","PPIACO":"PPI","UNRATE":"Unemployment",
    "PAYEMS":"Nonfarm_Payrolls","UMCSENT":"Consumer_Sentiment","RSXFS":"Retail_Sales_ExAuto",
    "INDPRO":"Industrial_Production","HOUST":"Housing_Starts","PERMIT":"Building_Permits",
    "M2SL":"M2","NAPM":"ISM_PMI","ICSA":"Initial_Claims","WALCL":"Fed_Assets","GDPC1":"Real_GDP",
}
IDX_MAP = {"^GSPC":"SP500","^IXIC":"Nasdaq","^DJI":"DowJones","^RUT":"Russell2000","^VIX":"VIX"}
CMD_MAP = {"GC=F":"Gold","CL=F":"WTI_Oil","SI=F":"Silver","HG=F":"Copper","NG=F":"NatGas"}
XBRL = {
    "Revenue":["Revenues","RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet","SalesRevenueGoodsNet"],
    "CostOfRevenue":["CostOfRevenue","CostOfGoodsAndServicesSold"],"GrossProfit":["GrossProfit"],
    "OperatingIncome":["OperatingIncomeLoss"],"NetIncome":["NetIncomeLoss"],
    "EPS_Basic":["EarningsPerShareBasic"],"EPS_Diluted":["EarningsPerShareDiluted"],
    "RnD_Expense":["ResearchAndDevelopmentExpense"],"SGA_Expense":["SellingGeneralAndAdministrativeExpense"],
    "Interest_Expense":["InterestExpense","InterestExpenseDebt"],"Income_Tax":["IncomeTaxExpenseBenefit"],
    "DepAmort":["DepreciationDepletionAndAmortization","DepreciationAndAmortization"],
    "Total_Assets":["Assets"],"Current_Assets":["AssetsCurrent"],
    "Cash":["CashAndCashEquivalentsAtCarryingValue","CashCashEquivalentsAndShortTermInvestments"],
    "Receivables":["AccountsReceivableNetCurrent","AccountsReceivableNet"],"Inventory":["InventoryNet"],
    "Total_Liabilities":["Liabilities"],"Current_Liab":["LiabilitiesCurrent"],
    "LongTerm_Debt":["LongTermDebtNoncurrent","LongTermDebt"],"ShortTerm_Debt":["ShortTermBorrowings","DebtCurrent"],
    "Equity":["StockholdersEquity","StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "Shares_Out":["CommonStockSharesOutstanding","EntityCommonStockSharesOutstanding"],
    "Op_CashFlow":["NetCashProvidedByUsedInOperatingActivities"],
    "CapEx":["PaymentsToAcquirePropertyPlantAndEquipment"],
    "Dividends_Paid":["PaymentsOfDividends","PaymentsOfDividendsCommonStock"],
    "Buyback":["PaymentsForRepurchaseOfCommonStock"],
}

SCHEMA = [
    ("Date","datetime64[ns]","标识","交易日期"),("Ticker","object","标识","股票代码"),
    ("Sector","object","标识","行业"),("SubIndustry","object","标识","子行业"),
    ("Open","float64","价量","开盘价"),("High","float64","价量","最高价"),("Low","float64","价量","最低价"),
    ("Close","float64","价量","收盘价"),("Adj_Close","float64","价量","复权收盘价"),("Volume","float64","价量","成交量"),
    ("Return","float64","衍生","日收益率"),("Log_Return","float64","衍生","对数收益率"),
    ("Amplitude","float64","衍生","振幅"),("Turnover","float64","衍生","成交额"),("VWAP","float64","衍生","均价"),
    ("Vol_5d","float64","衍生","5日波动率"),("Vol_20d","float64","衍生","20日波动率"),("Vol_60d","float64","衍生","60日波动率"),
    ("Mom_1w","float64","衍生","1周动量"),("Mom_1m","float64","衍生","1月动量"),("Mom_3m","float64","衍生","3月动量"),
    ("Mom_6m","float64","衍生","6月动量"),("Mom_12m","float64","衍生","12月动量"),
    ("Drawdown_60d","float64","衍生","60日回撤"),("Skew_20d","float64","衍生","偏度"),("Kurt_20d","float64","衍生","峰度"),
    ("MA_5d","float64","衍生","5日均线"),("MA_Dev_5d","float64","衍生","5日偏离"),
    ("MA_10d","float64","衍生","10日均线"),("MA_Dev_10d","float64","衍生","10日偏离"),
    ("MA_20d","float64","衍生","20日均线"),("MA_Dev_20d","float64","衍生","20日偏离"),
    ("MA_60d","float64","衍生","60日均线"),("MA_Dev_60d","float64","衍生","60日偏离"),
    ("MA_120d","float64","衍生","120日均线"),("MA_Dev_120d","float64","衍生","120日偏离"),
    ("MA_250d","float64","衍生","250日均线"),("MA_Dev_250d","float64","衍生","250日偏离"),
    ("RSI_14d","float64","衍生","RSI"),("Vol_Ratio","float64","衍生","量比"),
    ("Amihud","float64","衍生","非流动性"),("Parkinson","float64","衍生","HiLo波动率"),
    ("Dist_52wH","float64","衍生","距52周高"),("Beta","float64","衍生","Beta"),("IdioVol","float64","衍生","特质波动率"),
    ("Div_Amount","float64","事件","分红金额"),("Split_Ratio","float64","事件","拆股比例"),
    ("Insider_Filings","float64","事件","内部人申报"),
    ("Revenue","float64","基本面","营收"),("CostOfRevenue","float64","基本面","营业成本"),
    ("GrossProfit","float64","基本面","毛利"),("OperatingIncome","float64","基本面","营业利润"),
    ("NetIncome","float64","基本面","净利润"),("EPS_Basic","float64","基本面","基本EPS"),
    ("EPS_Diluted","float64","基本面","稀释EPS"),("RnD_Expense","float64","基本面","研发费用"),
    ("SGA_Expense","float64","基本面","销管费用"),("Interest_Expense","float64","基本面","利息支出"),
    ("Income_Tax","float64","基本面","所得税"),("DepAmort","float64","基本面","折旧摊销"),("EBITDA","float64","基本面","EBITDA"),
    ("Total_Assets","float64","基本面","总资产"),("Current_Assets","float64","基本面","流动资产"),
    ("Cash","float64","基本面","现金"),("Receivables","float64","基本面","应收"),("Inventory","float64","基本面","存货"),
    ("Total_Liabilities","float64","基本面","总负债"),("Current_Liab","float64","基本面","流动负债"),
    ("LongTerm_Debt","float64","基本面","长期债"),("ShortTerm_Debt","float64","基本面","短期债"),
    ("Equity","float64","基本面","股东权益"),("Shares_Out","float64","基本面","股数"),
    ("Op_CashFlow","float64","基本面","经营现金流"),("CapEx","float64","基本面","资本支出"),
    ("Dividends_Paid","float64","基本面","已付股息"),("Buyback","float64","基本面","回购"),("FCF","float64","基本面","自由现金流"),
    ("Mkt_Cap","float64","估值","市值"),("EV","float64","估值","企业价值"),
    ("PE","float64","估值","市盈率"),("PB","float64","估值","市净率"),("PS","float64","估值","市销率"),
    ("PCF","float64","估值","市现率"),("EV_EBITDA","float64","估值","EV/EBITDA"),("EV_Sales","float64","估值","EV/营收"),
    ("Earn_Yield","float64","估值","盈利收益率"),("FCF_Yield","float64","估值","FCF收益率"),
    ("Gross_Margin","float64","比率","毛利率"),("Net_Margin","float64","比率","净利率"),
    ("Op_Margin","float64","比率","营业利润率"),("RnD_Int","float64","比率","研发强度"),
    ("CapEx_Rev","float64","比率","CapEx/营收"),("ROE","float64","比率","ROE"),("ROA","float64","比率","ROA"),
    ("Asset_Turn","float64","比率","资产周转"),("AR_Turn","float64","比率","应收周转"),("Inv_Turn","float64","比率","存货周转"),
    ("Debt_Assets","float64","比率","负债率"),("Current_Ratio","float64","比率","流动比率"),
    ("Quick_Ratio","float64","比率","速动比率"),("Int_Coverage","float64","比率","利息保障"),("Accruals","float64","比率","应计"),
    ("Fed_Funds_Rate","float64","系统","联邦基金利率"),
    ("Treasury_1M","float64","系统","1月国债"),("Treasury_3M","float64","系统","3月国债"),
    ("Treasury_6M","float64","系统","6月国债"),("Treasury_1Y","float64","系统","1年国债"),
    ("Treasury_2Y","float64","系统","2年国债"),("Treasury_5Y","float64","系统","5年国债"),
    ("Treasury_7Y","float64","系统","7年国债"),("Treasury_10Y","float64","系统","10年国债"),
    ("Treasury_20Y","float64","系统","20年国债"),("Treasury_30Y","float64","系统","30年国债"),
    ("Term_Spread_10Y2Y","float64","系统","利差10-2"),("Term_Spread_10Y3M","float64","系统","利差10-3M"),
    ("TIPS_10Y_Real","float64","系统","实际利率"),("Breakeven_10Y","float64","系统","10Y通胀"),("Breakeven_5Y","float64","系统","5Y通胀"),
    ("IG_Spread","float64","系统","投资级利差"),("HY_Spread","float64","系统","高收益利差"),
    ("TED_Spread","float64","系统","TED利差"),("USD_Broad","float64","系统","美元指数"),
    ("USDEUR","float64","系统","美元/欧元"),("JPYUSD","float64","系统","日元/美元"),
    ("USDGBP","float64","系统","美元/英镑"),("CNYUSD","float64","系统","人民币/美元"),("VIX_FRED","float64","系统","VIX(FRED)"),
    ("CPI","float64","系统","CPI"),("Core_CPI","float64","系统","核心CPI"),("PPI","float64","系统","PPI"),
    ("Unemployment","float64","系统","失业率"),("Nonfarm_Payrolls","float64","系统","非农"),
    ("Consumer_Sentiment","float64","系统","消费者信心"),("Retail_Sales_ExAuto","float64","系统","零售销售"),
    ("Industrial_Production","float64","系统","工业产出"),("Housing_Starts","float64","系统","新屋开工"),
    ("Building_Permits","float64","系统","建筑许可"),("M2","float64","系统","M2"),("ISM_PMI","float64","系统","ISM PMI"),
    ("Initial_Claims","float64","系统","初请"),("Fed_Assets","float64","系统","美联储资产"),("Real_GDP","float64","系统","实际GDP"),
    ("SP500","float64","系统","标普500"),("Nasdaq","float64","系统","纳斯达克"),("DowJones","float64","系统","道琼斯"),
    ("Russell2000","float64","系统","罗素2000"),("VIX","float64","系统","VIX"),
    ("Gold","float64","系统","黄金"),("WTI_Oil","float64","系统","原油"),("Silver","float64","系统","白银"),
    ("Copper","float64","系统","铜"),("NatGas","float64","系统","天然气"),
    ("FF_MktRF","float64","系统","FF市场"),("FF_SMB","float64","系统","FF规模"),("FF_HML","float64","系统","FF价值"),
    ("FF_RMW","float64","系统","FF盈利"),("FF_CMA","float64","系统","FF投资"),("FF_Mom","float64","系统","FF动量"),("FF_RF","float64","系统","FF无风险"),
    ("B_Adv","float64","系统","上涨数"),("B_Dec","float64","系统","下跌数"),("B_Count","float64","系统","总数"),
    ("B_AD_Ratio","float64","系统","涨跌比"),("B_NewHi","float64","系统","新高数"),("B_NewLo","float64","系统","新低数"),
    ("YC_Slope","float64","系统","曲线斜率"),("YC_Curv","float64","系统","曲线曲率"),
    ("VIX_MA20","float64","系统","VIX均值"),("VIX_Dev","float64","系统","VIX偏离"),("Vol_Risk_Prem","float64","系统","波动率溢价"),
]
COLS = [s[0] for s in SCHEMA]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def out(msg=""): print(msg, flush=True)

def progress(current, total, prefix="", suffix="", width=30):
    """单行进度条, 原地覆盖."""
    pct = current / total if total > 0 else 1
    filled = int(width * pct)
    bar = "=" * filled + "-" * (width - filled)
    print(f"\r  {prefix} [{bar}] {current}/{total} {suffix}", end="", flush=True)
    if current >= total: print()

def sess(ua=None):
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1.5, status_forcelist=[429,500,502,503,504])))
    if ua: s.headers["User-Agent"] = ua
    return s

def cp(n):   return CACHE/f"{n}.parquet"
def ce(n):   return cp(n).exists()
def cr(n):   return pd.read_parquet(cp(n))
def cw(df,n): CACHE.mkdir(parents=True,exist_ok=True); df.to_parquet(cp(n),index=False,engine="pyarrow")
def ckp(n):  return CACHE/f"{n}.json"
def ckl(n):  return json.loads(ckp(n).read_text()) if ckp(n).exists() else {}
def cks(n,d): CACHE.mkdir(parents=True,exist_ok=True); ckp(n).write_text(json.dumps(d))
def ckd(n):  ckp(n).unlink(missing_ok=True)

def safe_col(df, col):
    return df[col] if col in df.columns else pd.Series(0.0, index=df.index)

def make_empty(dates, tickers):
    rows = [(d,t) for d in dates for t in tickers]
    df = pd.DataFrame(rows, columns=["Date","Ticker"])
    df["Date"] = pd.to_datetime(df["Date"])
    for nm,dt,_,_ in SCHEMA:
        if nm in ("Date","Ticker"): continue
        df[nm] = 0.0 if nm=="Div_Amount" else (1.0 if nm=="Split_Ratio" else (0.0 if nm=="Insider_Filings" else np.nan))
    return df[COLS]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INTERACTIVE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_config():
    """从配置文件读取, 或交互式引导用户输入."""
    if CONFIG_FILE.exists():
        cfg = json.loads(CONFIG_FILE.read_text())
        if cfg.get("fred_key") and cfg.get("sec_name") and cfg.get("sec_email"):
            return cfg

    out("\n" + "="*50)
    out("  S&P 500 Pipeline - 首次配置")
    out("="*50)

    out("\n[1/3] FRED API Key")
    out("  获取: https://fred.stlouisfed.org/docs/api/api_key.html")
    fred_key = input("  请输入你的 FRED API Key: ").strip()

    out("\n[2/3] SEC EDGAR 联系人姓名")
    sec_name = input("  请输入名字: ").strip() or "Researcher"

    out("\n[3/3] SEC EDGAR 联系邮箱")
    sec_email = input("  请输入邮箱: ").strip() or "user@example.com"

    cfg = {"fred_key": fred_key, "sec_name": sec_name, "sec_email": sec_email}
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    out(f"\n  配置已保存到 {CONFIG_FILE}")
    return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODULE 1: LOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def L_constituents():
    if ce("metadata"): out("  constituents    [cached]"); return
    out("  constituents    loading from Wikipedia...")
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": BROWSER_UA}, timeout=30)
    tables = pd.read_html(StringIO(r.text))
    raw = tables[0].copy(); cm = {}
    for c in raw.columns:
        cl = str(c).lower()
        if "symbol" in cl or "ticker" in cl: cm[c]="Ticker"
        elif "security" in cl: cm[c]="Company"
        elif "sub" in cl and "indust" in cl: cm[c]="SubIndustry"
        elif "sector" in cl: cm[c]="Sector"
        elif "cik" in cl: cm[c]="CIK"
        elif "date" in cl and "added" in cl: cm[c]="Date_Added"
    raw = raw.rename(columns=cm); raw["Ticker"]=raw["Ticker"].str.replace(".","-",regex=False)
    if "CIK" in raw.columns: raw["CIK"]=raw["CIK"].astype(str).str.strip().str.zfill(10)
    chg = tables[1].copy() if len(tables)>1 else pd.DataFrame()
    recs = []
    for _,r2 in raw.iterrows():
        di = pd.NaT
        try: di = pd.to_datetime(r2.get("Date_Added"))
        except: pass
        if pd.isna(di) or di < pd.Timestamp(START_DATE): di = pd.Timestamp(START_DATE)
        recs.append({"Ticker":r2["Ticker"],"Company":r2.get("Company",""),"CIK":r2.get("CIK",""),
                     "Sector":r2.get("Sector",""),"SubIndustry":r2.get("SubIndustry",""),
                     "Date_In":di,"Date_Out":pd.NaT,"Is_Current":True})
        if streaming == True:
            
            # what to add
            time.sleep(0.02)

    if len(chg)>0:
        if isinstance(chg.columns,pd.MultiIndex): chg.columns=["_".join(str(x) for x in c).strip("_") for c in chg.columns]
        dc=rc=None
        for c in chg.columns:
            cl=str(c).lower()
            if "date" in cl: dc=c
            if "removed" in cl and ("ticker" in cl or "symbol" in cl): rc=c
        if dc and rc:
            for _,r2 in chg.iterrows():
                t,d=r2.get(rc),r2.get(dc)
                if pd.isna(t) or pd.isna(d): continue
                t=str(t).strip().replace(".","-")
                try: d=pd.to_datetime(d)
                except: continue
                if d<pd.Timestamp(START_DATE): continue
                recs.append({"Ticker":t,"Company":"","CIK":"","Sector":"","SubIndustry":"",
                             "Date_In":pd.Timestamp(START_DATE),"Date_Out":d,"Is_Current":False})
    cw(pd.DataFrame(recs),"metadata"); cw(raw,"current")
    current_n = sum(1 for r in recs if r["Is_Current"])
    out(f"  constituents    {len(recs)} total ({current_n} current, {len(recs)-current_n} historical)")

def L_cik(sec_ua):
    if ce("cik_map"): out("  cik_map         [cached]"); return
    out("  cik_map         loading from SEC...")
    meta=cr("metadata"); tc={}
    for _,r in meta.iterrows():
        c=str(r.get("CIK","")).strip()
        if c and c!="nan" and len(c)>=4: tc[r["Ticker"]]=c.zfill(10)
    try:
        sm={str(v.get("ticker","")).replace(".","-"):str(v.get("cik_str","")).zfill(10)
            for v in sess(sec_ua).get("https://www.sec.gov/files/company_tickers.json",timeout=30).json().values()}
        for t in [t for t in meta["Ticker"].unique() if t not in tc]:
            if t in sm: tc[t]=sm[t]
    except: pass
    cw(pd.DataFrame([{"Ticker":t,"CIK":c} for t,c in tc.items()]),"cik_map")
    out(f"  cik_map         {len(tc)} mappings")

def L_prices():
    import yfinance as yf
    if ce("prices"): out("  prices          [cached]"); return
    out("  prices          loading from Yahoo Finance...")
    tks=sorted(cr("metadata")["Ticker"].unique()); total_tks=len(tks)
    ck=ckl("pr"); done=set(ck.get("d",[])); tmp=ck.get("f",[])
    tb=(total_tks+YF_BATCH-1)//YF_BATCH
    ok_count=0; fail_count=0
    for i in range(0,total_tks,YF_BATCH):
        bn=i//YF_BATCH
        if bn in done:
            ok_count += YF_BATCH  # 估算
            continue
        batch=tks[i:i+YF_BATCH]
        pcs=[]
        try:
            raw=yf.download(" ".join(batch),start=START_DATE,end=END_DATE,group_by="ticker",auto_adjust=False,threads=True,progress=False)
            if not raw.empty:
                if len(batch)==1:
                    d=raw.copy();d["Ticker"]=batch[0];d.index.name="Date";pcs.append(d.reset_index())
                else:
                    for t in batch:
                        try:
                            if t in raw.columns.get_level_values(0):
                                d=raw[t].dropna(how="all")
                                if len(d)>0: d=d.copy();d["Ticker"]=t;d.index.name="Date";pcs.append(d.reset_index())
                        except: pass
        except: pass
        batch_ok = len(pcs)
        batch_fail = len(batch) - batch_ok
        ok_count += batch_ok; fail_count += batch_fail
        if pcs: nm=f"_pr{bn}"; cw(pd.concat(pcs,ignore_index=True),nm); tmp.append(nm)
        done.add(bn); cks("pr",{"d":list(done),"f":tmp})
        progress(min(i+YF_BATCH, total_tks), total_tks, "prices", f"ok:{ok_count} skip:{fail_count}")
        if i+YF_BATCH<total_tks: time.sleep(YF_SLEEP)
    # 合并
    dfs=[cr(n) for n in tmp if ce(n)]
    if not dfs: out("  prices          FAILED - no data"); return
    p=pd.concat(dfs,ignore_index=True); rm={}
    for c in p.columns:
        cl=str(c).lower().replace(" ","_")
        if "adj" in cl and "close" in cl:rm[c]="Adj_Close"
        elif cl=="open":rm[c]="Open"
        elif cl=="high":rm[c]="High"
        elif cl=="low":rm[c]="Low"
        elif cl=="close":rm[c]="Close"
        elif cl=="volume":rm[c]="Volume"
    p=p.rename(columns=rm); p["Date"]=pd.to_datetime(p["Date"]).dt.tz_localize(None)
    p=p.dropna(subset=["Close"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
    cw(p,"prices")
    out(f"  prices          {len(p):,} rows, {p['Ticker'].nunique()} tickers ({fail_count} delisted skipped)")
    for n in tmp: cp(n).unlink(missing_ok=True)
    ckd("pr")

def L_divs():
    import yfinance as yf
    if ce("dividends") and ce("splits"): out("  divs/splits     [cached]"); return
    out("  divs/splits     loading from Yahoo Finance...")
    tks=sorted(cr("metadata")["Ticker"].unique()); total=len(tks)
    ck=ckl("ds"); done=set(ck.get("d",[])); dr=ck.get("dr",[]); sr=ck.get("sr",[]); st=pd.Timestamp(START_DATE)
    ok=0; skip=0
    for i,t in enumerate(tks):
        if t in done: ok+=1; continue
        got_data = False
        try:
            stk=yf.Ticker(t)
            hist=stk.history(period="max", auto_adjust=False)
            if "Dividends" in hist.columns:
                for dt,v in hist["Dividends"].items():
                    if v > 0:
                        dt2=pd.to_datetime(dt).tz_localize(None)
                        if dt2>=st: dr.append({"Date":str(dt2),"Ticker":t,"Div_Amount":float(v)}); got_data=True
            if "Stock Splits" in hist.columns:
                for dt,v in hist["Stock Splits"].items():
                    if v > 0 and v != 1.0:
                        dt2=pd.to_datetime(dt).tz_localize(None)
                        if dt2>=st: sr.append({"Date":str(dt2),"Ticker":t,"Split_Ratio":float(v)}); got_data=True
        except: pass
        if got_data: ok+=1
        else: skip+=1
        done.add(t)
        if (i+1)%100==0:
            cks("ds",{"d":list(done),"dr":dr,"sr":sr})
            progress(i+1, total, "divs", f"ok:{ok} skip:{skip}")
        if (i+1)%50==0: time.sleep(0.3)
    progress(total, total, "divs", f"ok:{ok} skip:{skip}")
    dv=pd.DataFrame(dr) if dr else pd.DataFrame(columns=["Date","Ticker","Div_Amount"])
    sp=pd.DataFrame(sr) if sr else pd.DataFrame(columns=["Date","Ticker","Split_Ratio"])
    if len(dv)>0: dv["Date"]=pd.to_datetime(dv["Date"])
    if len(sp)>0: sp["Date"]=pd.to_datetime(sp["Date"])
    cw(dv,"dividends"); cw(sp,"splits"); ckd("ds")
    out(f"  divs/splits     {len(dv)} dividends, {len(sp)} splits")

def _xbrl_extract(ns, tags):
    for tag in tags:
        u=ns.get(tag,{}).get("units",{})
        fl=u.get("USD") or u.get("USD/shares") or u.get("shares") or u.get("pure") or []
        r={}
        for f in fl:
            e,v,fr=f.get("end"),f.get("val"),f.get("frame")
            if e and v is not None and fr and e not in r: r[e]=v
        if r: return r
    return {}

def L_fund(sec_ua):
    if ce("fundamentals"): out("  fundamentals    [cached]"); return
    out("  fundamentals    loading from SEC EDGAR...")
    items=list(zip(cr("cik_map")["Ticker"],cr("cik_map")["CIK"])); total=len(items)
    ck=ckl("sf"); done=set(ck.get("d",[])); rows=ck.get("rows",[])
    s=sess(sec_ua); ok=0; fail=0
    for i,(tk,cik) in enumerate(items):
        if tk in done: ok+=1; continue
        got=False
        try:
            r=s.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",timeout=30)
            if r.status_code==200:
                ns={**r.json().get("facts",{}).get("us-gaap",{})}
                for k,v in r.json().get("facts",{}).get("dei",{}).items():
                    if k not in ns: ns[k]=v
                fd={on:_xbrl_extract(ns,tl) for on,tl in XBRL.items()}
                fd={k:v for k,v in fd.items() if v}
                if fd:
                    for p in sorted(set().union(*fd.values())):
                        row={"Period_End":p,"Ticker":tk}
                        for fn,pv in fd.items(): row[fn]=pv.get(p)
                        rows.append(row)
                    got=True
        except: pass
        if got: ok+=1
        else: fail+=1
        done.add(tk); time.sleep(SEC_RATE)
        if (i+1)%50==0:
            cks("sf",{"d":list(done),"rows":rows})
            progress(i+1, total, "fund", f"ok:{ok} fail:{fail}")
    progress(total, total, "fund", f"ok:{ok} fail:{fail}")
    if rows:
        f=pd.DataFrame(rows); f["Period_End"]=pd.to_datetime(f["Period_End"])
        f=f.sort_values(["Ticker","Period_End"]).reset_index(drop=True)
        if "OperatingIncome" in f.columns and "DepAmort" in f.columns: f["EBITDA"]=f["OperatingIncome"].fillna(0)+f["DepAmort"].fillna(0)
        if "Op_CashFlow" in f.columns and "CapEx" in f.columns: f["FCF"]=f["Op_CashFlow"]-f["CapEx"].abs()
        cw(f,"fundamentals")
        out(f"  fundamentals    {len(f):,} records from {f['Ticker'].nunique()} companies")
    else: cw(pd.DataFrame(),"fundamentals"); out("  fundamentals    no data")
    ckd("sf")

def L_insider(sec_ua):
    if ce("insider"): out("  insider         [cached]"); return
    out("  insider         loading from SEC EDGAR...")
    items=list(zip(cr("cik_map")["Ticker"],cr("cik_map")["CIK"])); total=len(items)
    ck=ckl("si"); done=set(ck.get("d",[])); recs=ck.get("r",[])
    s=sess(sec_ua); st=pd.Timestamp(START_DATE)
    for i,(tk,cik) in enumerate(items):
        if tk in done: continue
        try:
            r=s.get(f"https://data.sec.gov/submissions/CIK{cik}.json",timeout=30)
            if r.status_code==200:
                rc=r.json().get("filings",{}).get("recent",{})
                dy=defaultdict(int)
                for fm,fd in zip(rc.get("form",[]),rc.get("filingDate",[])):
                    if fm in ("4","4/A"): dy[fd]+=1
                for fd,cnt in dy.items():
                    try:
                        dt=pd.to_datetime(fd)
                        if dt>=st: recs.append({"Date":str(dt),"Ticker":tk,"Insider_Filings":cnt})
                    except: pass
        except: pass
        done.add(tk); time.sleep(SEC_RATE)
        if (i+1)%50==0:
            cks("si",{"d":list(done),"r":recs})
            progress(i+1, total, "insider")
    progress(total, total, "insider")
    ins=pd.DataFrame(recs) if recs else pd.DataFrame(columns=["Date","Ticker","Insider_Filings"])
    if len(ins)>0: ins["Date"]=pd.to_datetime(ins["Date"])
    cw(ins,"insider"); ckd("si")
    out(f"  insider         {len(ins):,} filings")

def L_fred(key):
    from fredapi import Fred
    if ce("fred"): out("  fred            [cached]"); return
    out("  fred            loading from FRED...")
    fred=Fred(api_key=key); ss={}; total=len(FRED_SERIES)
    for i,(code,name) in enumerate(FRED_SERIES.items()):
        try:
            s=fred.get_series(code,observation_start=START_DATE,observation_end=END_DATE)
            if s is not None and len(s)>0: ss[name]=s
        except: pass
        time.sleep(FRED_SLEEP)
        if (i+1)%10==0: progress(i+1, total, "fred")
    progress(total, total, "fred")
    if not ss: out("  fred            FAILED"); return
    df=pd.DataFrame(ss);df.index=pd.to_datetime(df.index);df.index.name="Date"
    all_days=df.index.union(pd.bdate_range(START_DATE,END_DATE))
    df=df.reindex(all_days).ffill().reindex(pd.bdate_range(START_DATE,END_DATE))
    df.index.name="Date";df=df.reset_index()
    cw(df,"fred")
    out(f"  fred            {len(df)} days, {len(ss)}/{total} series")

def L_ff():
    import pandas_datareader.data as web
    if ce("ff"): out("  ff_factors      [cached]"); return
    out("  ff_factors      loading from Ken French...")
    r=pd.DataFrame()
    try:
        d=web.DataReader("F-F_Research_Data_5_Factors_2x3_daily","famafrench",start=START_DATE,end=END_DATE)[0]/100
        d.index=pd.to_datetime(d.index,format="%Y%m%d")
        d=d.rename(columns={"Mkt-RF":"FF_MktRF","SMB":"FF_SMB","HML":"FF_HML","RMW":"FF_RMW","CMA":"FF_CMA","RF":"FF_RF"})
        r=d
    except: pass
    try:
        m=web.DataReader("F-F_Momentum_Factor_daily","famafrench",start=START_DATE,end=END_DATE)[0]/100
        m.index=pd.to_datetime(m.index,format="%Y%m%d");m=m.rename(columns={m.columns[0]:"FF_Mom"})
        r=r.join(m,how="outer") if len(r)>0 else m
    except: pass
    if len(r)>0: r.index.name="Date";r=r.reset_index();r["Date"]=pd.to_datetime(r["Date"])
    cw(r,"ff")
    out(f"  ff_factors      {len(r)} days")

def L_market():
    import yfinance as yf
    if ce("market"): out("  market          [cached]"); return
    out("  market          loading from Yahoo Finance...")
    at={**IDX_MAP,**CMD_MAP}
    try: raw=yf.download(" ".join(at.keys()),start=START_DATE,end=END_DATE,auto_adjust=False,threads=True,progress=False)
    except: out("  market          FAILED"); return
    r=pd.DataFrame();r.index=pd.to_datetime(raw.index).tz_localize(None);r.index.name="Date"
    for yt,nm in at.items():
        try:
            if yt in raw.columns.get_level_values(1): r[nm]=raw["Close"][yt].values
        except: pass
    r=r.reset_index();cw(r,"market")
    out(f"  market          {len(r)} days, {len(r.columns)-1} series")

def module_load(cfg, only=None):
    sec_ua = f"{cfg['sec_name']} {cfg['sec_email']}"
    out("\n" + "="*50)
    out("  MODULE 1: LOAD")
    out("="*50)
    tasks = [
        ("constituents", lambda: L_constituents()),
        ("cik_map",      lambda: L_cik(sec_ua)),
        ("prices",       lambda: L_prices()),
        ("divs_splits",  lambda: L_divs()),
        ("sec_edgar",    lambda: L_fund(sec_ua)),
        ("insider",      lambda: L_insider(sec_ua)),
        ("fred",         lambda: L_fred(cfg["fred_key"])),
        ("ff",           lambda: L_ff()),
        ("market",       lambda: L_market()),
    ]
    for name, fn in tasks:
        if only and name != only: continue
        try: fn()
        except Exception as e: out(f"  {name:<16} FAILED: {e}")
    out("\n  Cache: " + " ".join(f"{n}[{'ok' if ce(n) else 'X'}]" for n in
        ["metadata","cik_map","prices","dividends","splits","fundamentals","insider","fred","ff","market"]))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODULE 2: PROCESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def enrich(tk, rows, fund, mkt_ret):
    df = rows.sort_values("Date").copy()
    if len(df)<2: return df
    df["Return"]=df["Adj_Close"].pct_change(); df["Log_Return"]=np.log(df["Adj_Close"]/df["Adj_Close"].shift(1))
    df["Amplitude"]=(df["High"]-df["Low"])/df["Close"].shift(1)
    df["Turnover"]=df["Close"]*df["Volume"]; df["VWAP"]=(df["High"]+df["Low"]+df["Close"])/3
    for w in [5,20,60]: df[f"Vol_{w}d"]=df["Return"].rolling(w,min_periods=max(3,w//3)).std()*np.sqrt(252)
    for d,l in [(5,"1w"),(21,"1m"),(63,"3m"),(126,"6m"),(252,"12m")]: df[f"Mom_{l}"]=df["Adj_Close"].pct_change(d)
    df["Drawdown_60d"]=(df["Adj_Close"]-df["Adj_Close"].rolling(60,min_periods=1).max())/df["Adj_Close"].rolling(60,min_periods=1).max()
    df["Skew_20d"]=df["Return"].rolling(20,min_periods=10).skew()
    df["Kurt_20d"]=df["Return"].rolling(20,min_periods=10).kurt()
    for w in [5,10,20,60,120,250]:
        ma=df["Adj_Close"].rolling(w,min_periods=max(1,w//3)).mean()
        df[f"MA_{w}d"]=ma; df[f"MA_Dev_{w}d"]=(df["Adj_Close"]-ma)/ma
    delta=df["Adj_Close"].diff()
    df["RSI_14d"]=100-100/(1+delta.clip(lower=0).rolling(14,min_periods=7).mean()/(-delta).clip(lower=0).rolling(14,min_periods=7).mean().replace(0,np.nan))
    df["Vol_Ratio"]=df["Volume"]/df["Volume"].rolling(20,min_periods=5).mean()
    df["Amihud"]=df["Return"].abs()/df["Turnover"].replace(0,np.nan)
    df["Parkinson"]=np.sqrt((1/(4*np.log(2)))*np.log(df["High"]/df["Low"].replace(0,np.nan))**2)
    df["Dist_52wH"]=(df["Close"]-df["High"].rolling(252,min_periods=60).max())/df["High"].rolling(252,min_periods=60).max()
    ft=fund[fund["Ticker"]==tk] if not fund.empty else pd.DataFrame()
    if not ft.empty:
        fc=[c for c in ft.columns if c not in ("Ticker","Period_End")]
        df=pd.merge_asof(df,ft[["Period_End"]+fc].sort_values("Period_End"),left_on="Date",right_on="Period_End",direction="backward")
        df=df.drop(columns=["Period_End"],errors="ignore")
    if "Shares_Out" in df.columns: df["Mkt_Cap"]=df["Close"]*df["Shares_Out"]
    lt=safe_col(df,"LongTerm_Debt"); st2=safe_col(df,"ShortTerm_Debt")
    debt=pd.to_numeric(lt,errors="coerce").fillna(0)+pd.to_numeric(st2,errors="coerce").fillna(0)
    if "Mkt_Cap" in df.columns: df["EV"]=df["Mkt_Cap"]+debt-safe_col(df,"Cash").fillna(0)
    def R(a,b,n):
        if a in df.columns and b in df.columns: df[n]=df[a]/df[b].replace(0,np.nan)
    R("Mkt_Cap","NetIncome","PE");R("Mkt_Cap","Equity","PB");R("Mkt_Cap","Revenue","PS");R("Mkt_Cap","Op_CashFlow","PCF")
    R("EV","EBITDA","EV_EBITDA");R("EV","Revenue","EV_Sales")
    if "Mkt_Cap" in df.columns:
        mc=df["Mkt_Cap"].replace(0,np.nan)
        if "NetIncome" in df.columns: df["Earn_Yield"]=df["NetIncome"]/mc
        if "FCF" in df.columns: df["FCF_Yield"]=df["FCF"]/mc
    rv=safe_col(df,"Revenue").replace(0,np.nan)
    for a,n in [("GrossProfit","Gross_Margin"),("NetIncome","Net_Margin"),("OperatingIncome","Op_Margin"),("RnD_Expense","RnD_Int"),("CapEx","CapEx_Rev")]:
        if a in df.columns: df[n]=df[a].abs()/rv
    R("NetIncome","Equity","ROE");R("NetIncome","Total_Assets","ROA");R("Revenue","Total_Assets","Asset_Turn")
    R("Revenue","Receivables","AR_Turn");R("CostOfRevenue","Inventory","Inv_Turn")
    R("Total_Liabilities","Total_Assets","Debt_Assets");R("Current_Assets","Current_Liab","Current_Ratio")
    if all(c in df.columns for c in ["Current_Assets","Inventory","Current_Liab"]):
        df["Quick_Ratio"]=(df["Current_Assets"]-df["Inventory"].fillna(0))/df["Current_Liab"].replace(0,np.nan)
    R("OperatingIncome","Interest_Expense","Int_Coverage")
    if all(c in df.columns for c in ["NetIncome","Op_CashFlow","Total_Assets"]):
        df["Accruals"]=(df["NetIncome"]-df["Op_CashFlow"])/df["Total_Assets"].replace(0,np.nan)
    if mkt_ret is not None:
        mr=mkt_ret.reindex(df["Date"]).values; df["_mr"]=mr; w=250
        xy=df["Return"]*df["_mr"]
        rc2=xy.rolling(w,min_periods=60).mean()-df["Return"].rolling(w,min_periods=60).mean()*df["_mr"].rolling(w,min_periods=60).mean()
        rv2=df["_mr"].rolling(w,min_periods=60).var()
        df["Beta"]=rc2/rv2.replace(0,np.nan)
        df["IdioVol"]=(df["Return"]-df["Beta"]*df["_mr"]).rolling(w,min_periods=60).std()*np.sqrt(252)
        df=df.drop(columns=["_mr"])
    return df

def P_prepare():
    out("  Step 1/3: Enriching per-ticker...")
    ENRICHED.mkdir(parents=True,exist_ok=True)
    prices=cr("prices"); fund=cr("fundamentals") if ce("fundamentals") else pd.DataFrame()
    mkt_ret=None
    if ce("market"):
        m=cr("market")
        if "SP500" in m.columns: mkt_ret=m.set_index(pd.to_datetime(m["Date"]))["SP500"].pct_change()
    tks=sorted(prices["Ticker"].unique()); total=len(tks)
    ck=ckl("en"); done=set(ck.get("d",[]))
    for i,tk in enumerate(tks):
        if tk in done: continue
        e=enrich(tk,prices[prices["Ticker"]==tk],fund,mkt_ret)
        e.to_parquet(ENRICHED/f"{tk}.parquet",index=False,engine="pyarrow")
        done.add(tk)
        if (i+1)%50==0:
            cks("en",{"d":list(done)})
            progress(i+1, total, "enrich")
    cks("en",{"d":list(done)})
    progress(total, total, "enrich")
    out(f"           {len(done)} tickers enriched")

def P_assemble():
    out("  Step 2/3: Assembling quarterly tables...")
    meta=cr("metadata"); OUT.mkdir(parents=True,exist_ok=True)
    # 系统数据索引
    fred_ix = cr("fred").set_index(pd.to_datetime(cr("fred")["Date"])).drop(columns=["Date"]) if ce("fred") else pd.DataFrame()
    ff_ix   = cr("ff").set_index(pd.to_datetime(cr("ff")["Date"])).drop(columns=["Date"]) if ce("ff") else pd.DataFrame()
    mkt_ix  = cr("market").set_index(pd.to_datetime(cr("market")["Date"])).drop(columns=["Date"]) if ce("market") else pd.DataFrame()
    div_ix  = cr("dividends").assign(Date=lambda x:pd.to_datetime(x["Date"])).set_index(["Date","Ticker"])["Div_Amount"] if ce("dividends") and len(cr("dividends"))>0 else pd.Series(dtype=float)
    spl_ix  = cr("splits").assign(Date=lambda x:pd.to_datetime(x["Date"])).set_index(["Date","Ticker"])["Split_Ratio"] if ce("splits") and len(cr("splits"))>0 else pd.Series(dtype=float)
    ins_ix  = cr("insider").assign(Date=lambda x:pd.to_datetime(x["Date"])).set_index(["Date","Ticker"])["Insider_Filings"] if ce("insider") and len(cr("insider"))>0 else pd.Series(dtype=float)
    sec_map=meta.drop_duplicates("Ticker").set_index("Ticker")
    # VIX衍生
    vix_d=pd.DataFrame()
    if not mkt_ix.empty and "VIX" in mkt_ix.columns and "SP500" in mkt_ix.columns:
        vix=mkt_ix["VIX"]; sp_ret=mkt_ix["SP500"].pct_change()
        vix_d["VIX_MA20"]=vix.rolling(20,min_periods=5).mean()
        vix_d["VIX_Dev"]=(vix-vix_d["VIX_MA20"])/vix_d["VIX_MA20"]
        vix_d["Vol_Risk_Prem"]=vix-sp_ret.rolling(20,min_periods=5).std()*np.sqrt(252)*100

    etks=[p.stem for p in ENRICHED.glob("*.parquet")]
    if not etks: out("           No enriched data"); return
    sample_dates=pd.to_datetime(pd.read_parquet(ENRICHED/f"{etks[0]}.parquet",columns=["Date"])["Date"])
    quarters=pd.period_range(sample_dates.min(),sample_dates.max(),freq="Q")
    total_q=len(quarters)

    for qi,qp in enumerate(quarters):
        qs,qe=qp.start_time,qp.end_time; qn=f"Q{qp.year}Q{qp.quarter}"
        active=[]
        for _,r in meta.iterrows():
            di=r["Date_In"]; do=r["Date_Out"] if pd.notna(r["Date_Out"]) else pd.Timestamp("2099-01-01")
            if di<=qe and do>=qs: active.append(r["Ticker"])
        if not active: continue
        slices={}; all_dates=set()
        for tk in active:
            fp=ENRICHED/f"{tk}.parquet"
            if not fp.exists(): continue
            try:
                td=pd.read_parquet(fp); td["Date"]=pd.to_datetime(td["Date"])
                s=td[(td["Date"]>=qs)&(td["Date"]<=qe)]
                if len(s)>0: slices[tk]=s; all_dates.update(s["Date"].tolist())
            except: pass
        if not all_dates: continue
        q_dates=sorted(pd.date_range(min(all_dates), max(all_dates), freq="D").tolist())
        table=make_empty(q_dates, sorted(active))
        if slices:
            combined=pd.concat(slices.values(),ignore_index=True)
            combined["Date"]=pd.to_datetime(combined["Date"])
            common=[c for c in combined.columns if c in COLS]
            ti=table.set_index(["Date","Ticker"]); fi=combined[common].set_index(["Date","Ticker"])
            ti.update(fi); table=ti.reset_index()
        for idx,col,default in [(div_ix,"Div_Amount",0),(spl_ix,"Split_Ratio",1.0),(ins_ix,"Insider_Filings",0)]:
            if len(idx)==0: continue
            try:
                sub=idx.reset_index(); sub["Date"]=pd.to_datetime(sub["Date"])
                sub=sub[(sub["Date"]>=qs)&(sub["Date"]<=qe)]
                if len(sub)>0:
                    table=table.drop(columns=[col],errors="ignore")
                    table=table.merge(sub,on=["Date","Ticker"],how="left"); table[col]=table[col].fillna(default)
            except: pass
        for sys_ix in [fred_ix, ff_ix, mkt_ix, vix_d]:
            if sys_ix.empty: continue
            scols=[c for c in sys_ix.columns if c in COLS]
            if not scols: continue
            sub=sys_ix.loc[sys_ix.index.isin(q_dates),scols]
            if len(sub)>0:
                sub2=sub.reset_index().rename(columns={"index":"Date"})
                table=table.drop(columns=[c for c in scols if c in table.columns],errors="ignore")
                table=table.merge(sub2,on="Date",how="left")
        if "Return" in table.columns:
            brd=table.groupby("Date").agg(B_Adv=("Return",lambda x:(x>0).sum()),B_Dec=("Return",lambda x:(x<0).sum()),B_Count=("Ticker","count")).reset_index()
            brd["B_AD_Ratio"]=brd["B_Adv"]/brd["B_Dec"].replace(0,np.nan)
            if "Dist_52wH" in table.columns:
                h=table.groupby("Date")["Dist_52wH"].agg(B_NewHi=lambda x:(x>=-0.001).sum(),B_NewLo=lambda x:(x<=-0.25).sum()).reset_index()
                brd=brd.merge(h,on="Date",how="left")
            for c in brd.columns:
                if c!="Date" and c in table.columns: table=table.drop(columns=[c])
            table=table.merge(brd,on="Date",how="left")
        if "Treasury_10Y" in table.columns and "Treasury_2Y" in table.columns:
            table["YC_Slope"]=table["Treasury_10Y"]-table["Treasury_2Y"]
        if all(c in table.columns for c in ["Treasury_5Y","Treasury_2Y","Treasury_10Y"]):
            table["YC_Curv"]=2*table["Treasury_5Y"]-table["Treasury_2Y"]-table["Treasury_10Y"]
        for tk in active:
            if tk in sec_map.index:
                mask=table["Ticker"]==tk
                try: table.loc[mask,"Sector"]=sec_map.loc[tk,"Sector"]
                except: pass
                try: table.loc[mask,"SubIndustry"]=sec_map.loc[tk,"SubIndustry"]
                except: pass
        for c in COLS:
            if c not in table.columns: table[c]=np.nan
        table=table[COLS].sort_values(["Date","Ticker"]).reset_index(drop=True)
        table.to_parquet(OUT/f"{qn}.parquet",index=False,engine="pyarrow")
        progress(qi+1, total_q, "quarter", qn)

    n=len(list(OUT.glob("Q*.parquet")))
    out(f"           {n} quarterly files written")

def P_finalize():
    out("  Step 3/3: Merging and generating output files...")
    meta=cr("metadata"); meta.to_parquet(OUT/"metadata.parquet",index=False,engine="pyarrow")
    sj={"description":"S&P 500 Daily Quantitative Data","total_columns":len(SCHEMA),
        "columns":[{"name":s[0],"type":s[1],"category":s[2],"description":s[3]} for s in SCHEMA]}
    (OUT/"schema.json").write_text(json.dumps(sj,indent=2,ensure_ascii=False))
    qf=sorted(OUT.glob("Q*.parquet"))
    if not qf: out("           No quarter files"); return
    full=pd.concat([pd.read_parquet(f) for f in qf],ignore_index=True)
    fp=OUT/"sp500_full.parquet"; full.to_parquet(fp,index=False,engine="pyarrow")
    sd=full["Date"].dropna().iloc[len(full)//2]
    sample=full[full["Date"]==sd].head(5)
    (OUT/"sample.json").write_text(json.dumps(
        json.loads(sample.to_json(orient="records",date_format="iso",default_handler=str)),indent=2,ensure_ascii=False))
    out(f"           sp500_full.parquet: {fp.stat().st_size/1024/1024:.0f} MB")
    out(f"           schema.json + sample.json written")
    return full

def module_process():
    out("\n" + "="*50)
    out("  MODULE 2: PROCESS")
    out("="*50)
    if not ce("prices"): out("  ERROR: prices cache missing, run 'load' first"); return None
    P_prepare(); P_assemble(); full=P_finalize()
    return full


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def eval_quick(df):
    """输出季度文件时调用的简单检查."""
    rows = len(df); cols = len(df.columns); tks = df["Ticker"].nunique()
    fill = df.drop(columns=["Date","Ticker"],errors="ignore").notna().mean().mean()*100
    out(f"           Quick check: {rows:,} rows, {cols} cols, {tks} tickers, {fill:.0f}% filled")

def eval_full(df=None):
    """完整数据质量报告."""
    if df is None:
        fp = OUT / "sp500_full.parquet"
        if not fp.exists(): out("  No data to evaluate"); return
        df = pd.read_parquet(fp)

    out("\n" + "="*50)
    out("  DATA QUALITY REPORT")
    out("="*50)
    out(f"  Rows:     {len(df):>12,}")
    out(f"  Columns:  {len(df.columns):>12}")
    out(f"  Tickers:  {df['Ticker'].nunique():>12}")
    out(f"  Dates:    {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    out(f"  Size:     {(OUT/'sp500_full.parquet').stat().st_size/1024/1024:>10.0f} MB")

    # Ticker覆盖
    tk_fill = df.groupby("Ticker")["Close"].apply(lambda x: x.notna().mean())
    has_data = (tk_fill > 0).sum(); no_data = (tk_fill == 0).sum()
    out(f"\n  Ticker coverage:")
    out(f"    With price data:  {has_data}")
    out(f"    No data (empty):  {no_data}")

    # 列填充率
    out(f"\n  Column fill rates:")
    out(f"    {'Column':<28} {'Fill%':>6}  {'Category':<8}")
    out(f"    {'------':<28} {'-----':>6}  {'--------':<8}")
    cat_map = {s[0]:s[2] for s in SCHEMA}
    for col in COLS:
        if col in ("Date","Ticker"): continue
        if col not in df.columns: continue
        pct = df[col].notna().mean()*100
        if pct == 0:
            # ISM_PMI deleted from the FRED database
            # https://news.research.stlouisfed.org/2016/06/institute-for-supply-management-data-to-be-removed-from-fred/
            continue
        if pct < 1 or pct > 99.9 or col in ["Close","Return","Mkt_Cap","Revenue","PE","ROE",
                                              "Fed_Funds_Rate","VIX","FF_MktRF","Beta"]:
            flag = " !" if pct < 50 and cat_map.get(col) not in ("事件",) else "  "
            out(f"   {flag}{col:<28} {pct:>5.1f}%  {cat_map.get(col,''):<8}")

    # 样本数据
    out(f"\n  Sample data (latest date):")
    last = df["Date"].max()
    sample = df[df["Date"]==last].head(3)
    for _,r in sample.iterrows():
        out(f"    {r.get('Ticker','?'):>6} | Close:{r.get('Close',np.nan):>8.2f} "
            f"| PE:{r.get('PE',np.nan):>7.1f} | ROE:{r.get('ROE',np.nan):>7.3f} "
            f"| Sector:{str(r.get('Sector',''))[:15]}")

    # 综合评分
    overall = df.drop(columns=["Date","Ticker"],errors="ignore").notna().mean().mean()*100
    price_fill = df["Close"].notna().mean()*100
    out(f"\n  Overall fill rate:  {overall:.1f}%")
    out(f"  Price coverage:     {price_fill:.1f}%")
    out("="*50)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__=="__main__":
    p=argparse.ArgumentParser(description="S&P 500 Data Pipeline",
        epilog="First run: python sp500_pipeline.py run\nEvaluate:  python sp500_pipeline.py eval",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest="cmd")
    pl=sub.add_parser("load", help="Download raw data to cache")
    pl.add_argument("--only",choices=["constituents","cik_map","prices","divs_splits","sec_edgar","insider","fred","ff","market"])
    pl.add_argument("--clear-cache",action="store_true")
    sub.add_parser("process", help="Build output from cache (no network)")
    sub.add_parser("run", help="Load + Process (full pipeline)")
    sub.add_parser("eval", help="Evaluate output data quality")
    pr=sub.add_parser("reconfig", help="Re-enter API keys and settings")
    args=p.parse_args()
    if not args.cmd: p.print_help(); sys.exit(0)

    if args.cmd=="reconfig":
        CONFIG_FILE.unlink(missing_ok=True); load_config(); sys.exit(0)

    if args.cmd=="eval":
        eval_full(); sys.exit(0)

    if getattr(args,"clear_cache",False) and CACHE.exists():
        import shutil; shutil.rmtree(CACHE); out("Cache cleared")

    if args.cmd in ("load","run"):
        cfg=load_config()

    t0=time.time()
    if args.cmd=="load":
        module_load(cfg, only=args.only)
    elif args.cmd=="process":
        full=module_process()
        if full is not None: eval_full(full)
    elif args.cmd=="run":
        module_load(cfg)
        full=module_process()
        if full is not None: eval_full(full)

    out(f"\nDone in {(time.time()-t0)/60:.1f} minutes")
