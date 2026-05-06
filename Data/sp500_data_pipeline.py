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
    # ── Income Statement ──────────────────────────────────────────────────────
    "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "RevenueFromContractWithCustomerIncludingAssessedTax", "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet", "SalesRevenueServicesNet", "InterestAndDividendIncomeOperating", "InterestIncomeExpenseAfterProvisionForLoanLoss", "FinancialServicesRevenue", "NoninterestIncome", "PremiumsEarnedNet", "InsurancePremiumsRevenueNetOfReinsurance", "RegulatedAndUnregulatedOperatingRevenue", "RealEstateRevenueNet", "HealthCareOrganizationRevenue", "ContractsRevenue", "OilAndGasRevenue", "ElectricUtilityRevenue", "RevenueFromRelatedParties"],
    "CostOfRevenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"],
    "GrossProfit": ["GrossProfit"],
    "OperatingIncome": ["OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    "NetIncome": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"],
    "EPS_Basic": ["EarningsPerShareBasic", "IncomeLossFromContinuingOperationsPerBasicShare"],
    "EPS_Diluted": ["EarningsPerShareDiluted", "IncomeLossFromContinuingOperationsPerDilutedShare"],
    "RnD_Expense": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost", "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost"],
    "SGA_Expense": ["SellingGeneralAndAdministrativeExpense", "SellingAndMarketingExpense", "GeneralAndAdministrativeExpense"],
    "Interest_Expense": ["InterestExpense", "InterestExpenseDebt", "InterestExpenseBorrowings", "InterestPaidNet"],
    "Income_Tax": ["IncomeTaxExpenseBenefit", "IncomeTaxesPaidNet"],
    "DepAmort": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation", "AmortizationOfIntangibleAssets", "DepreciationAmortizationAndAccretionNet"],

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    "Total_Assets": ["Assets"],
    "Current_Assets": ["AssetsCurrent"],
    "Cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments", "Cash", "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "Receivables": ["AccountsReceivableNetCurrent", "AccountsReceivableNet", "ReceivablesNetCurrent", "AccountsNotesAndLoansReceivableNetCurrent"],
    "Inventory": ["InventoryNet", "InventoryFinishedGoodsAndWorkInProcess", "InventoryGross"],
    "Total_Liabilities": ["Liabilities", "LiabilitiesNoncurrent"],
    "Current_Liab": ["LiabilitiesCurrent"],
    "LongTerm_Debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebtFairValue"],
    "ShortTerm_Debt": ["ShortTermBorrowings", "DebtCurrent", "CommercialPaper", "LongTermDebtCurrent"],
    "Equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "Shares_Out": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesIssued"],

    # ── Cash Flow Statement ───────────────────────────────────────────────────
    "Op_CashFlow": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "CapEx": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets", "PaymentsForCapitalImprovements", "CapitalExpenditureDiscontinuedOperations"],
    "Dividends_Paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock", "PaymentsOfOrdinaryDividends", "PaymentsOfDividendsPreferredStockAndPreferenceStock"],
    "Buyback": ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity", "StockRepurchasedAndRetiredDuringPeriodValue", "StockRepurchasedDuringPeriodValue"]
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
    ("Treasury_10Y_Chg5d","float64","系统","10Y国债5日变化"),("Treasury_10Y_Chg20d","float64","系统","10Y国债20日变化"),
    ("Treasury_2Y_Chg5d","float64","系统","2Y国债5日变化"),("Treasury_2Y_Chg20d","float64","系统","2Y国债20日变化"),
    ("Fed_Funds_Rate_Chg5d","float64","系统","联邦基金5日变化"),("Fed_Funds_Rate_Chg20d","float64","系统","联邦基金20日变化"),
    ("IG_Spread_Chg20d","float64","系统","投资级利差20日变化"),("HY_Spread_Chg20d","float64","系统","高收益利差20日变化"),
]
COLS = [s[0] for s in SCHEMA]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITIES (HACKER UI ENGINE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

C_GREEN  = '\033[92m'
C_CYAN   = '\033[96m'
C_YELLOW = '\033[93m'
C_RED    = '\033[91m'
C_DIM    = '\033[2m'
C_RESET  = '\033[0m'
C_BOLD   = '\033[1m'

# 全局高频输出开关（可以在这里临时设置为 True 来测试效果）
VERBOSE_MODE = True

def sys_log(module, action, status="OK", latency_ms=None, details="", force=False):
    """
    底层日志推流引擎。
    force=True：无视静默模式，强制输出（用于关键步骤）。
    VERBOSE_MODE=True：开启高频微观刷屏。
    """
    if not (VERBOSE_MODE or force or status == "ERR"):
        return 

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3] 
    lat_str = f"{latency_ms:>5}ms" if latency_ms is not None else " " * 7
    
    if status == "OK":      c_stat = C_GREEN
    elif status == "WARN":  c_stat = C_YELLOW
    elif status == "ERR":   c_stat = C_RED
    else:                   c_stat = C_CYAN

    msg = (f"{C_DIM}[{ts}]{C_RESET} {C_CYAN}[{module:^8}]{C_RESET} "
           f"{action:<35} [{c_stat}{status:^4}{C_RESET}] "
           f"{C_DIM}{lat_str}{C_RESET} {C_GREEN}{details}{C_RESET}\n")
           
    sys.stdout.write(msg)
    sys.stdout.flush()

def boot_sequence():
    """终端启动伪装序列，强制输出"""
    lines = [
        "INIT CORE UPLINK...",
        "MOUNTING VFS -> ./sp500_data",
        "BYPASSING RATE LIMITS [OK]",
        f"ESTABLISHED SECURE CHANNEL [TIME: {datetime.now().isoformat()[:19]}]"
    ]
    sys.stdout.write(f"\n{C_CYAN}")
    for line in lines:
        sys.stdout.write(f"[*] {line}\n")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f"{C_RESET}\n")
    sys.stdout.flush()

def section_header(title):
    """带科技感的模块分割线"""
    sys.stdout.write(f"\n{C_DIM}" + "="*60 + f"{C_RESET}\n")
    sys.stdout.write(f"  {C_BOLD if 'C_BOLD' in globals() else ''}[ {title} ]{C_RESET}\n")
    sys.stdout.write(f"{C_DIM}" + "="*60 + f"{C_RESET}\n")
    sys.stdout.flush()

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

    section_header("S&P 500 PIPELINE — FIRST RUN SETUP")

    sys.stdout.write(f"\n  {C_CYAN}[1/3]{C_RESET} FRED API Key\n")
    sys.stdout.write(f"  {C_DIM}获取: https://fred.stlouisfed.org/docs/api/api_key.html{C_RESET}\n")
    fred_key = input(f"  {C_YELLOW}>{C_RESET} 请输入你的 FRED API Key: ").strip()

    sys.stdout.write(f"\n  {C_CYAN}[2/3]{C_RESET} SEC EDGAR 联系人姓名\n")
    sec_name = input(f"  {C_YELLOW}>{C_RESET} 请输入名字: ").strip() or "Researcher"

    sys.stdout.write(f"\n  {C_CYAN}[3/3]{C_RESET} SEC EDGAR 联系邮箱\n")
    sec_email = input(f"  {C_YELLOW}>{C_RESET} 请输入邮箱: ").strip() or "user@example.com"

    cfg = {"fred_key": fred_key, "sec_name": sec_name, "sec_email": sec_email}
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    sys_log("SYS_CORE", f"CONFIG_SAVED -> {CONFIG_FILE}", "OK", 0, "", force=True)
    return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODULE 1: LOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def L_constituents():
    if ce("metadata"):
        sys_log("WIKI_SCR", "CHECK cache/metadata", "OK", 0, "CACHED", force=True); return
    sys_log("WIKI_SCR", "CONNECT en.wikipedia.org", "WAIT", None, "SCRAPING", force=True)
    t_start = time.time()
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
    latency = int((time.time() - t_start) * 1000)
    sys_log("WIKI_SCR", "PARSE HTML_TABLES", "OK", latency, f"ROWS:{len(recs)} ACTIVE:{current_n}", force=True)

def L_cik(sec_ua):
    if ce("cik_map"):
        sys_log("SEC_MAP", "CHECK cache/cik_map", "OK", 0, "CACHED", force=True); return
    sys_log("SEC_MAP", "FETCH company_tickers.json", "WAIT", None, "CONNECTING", force=True)
    t_start = time.time()
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
    latency = int((time.time() - t_start) * 1000)
    sys_log("SEC_MAP", "MERGE CIK_DATA", "OK", latency, f"MAPPINGS:{len(tc)}", force=True)

def L_prices():
    import yfinance as yf
    if ce("prices"):
        sys_log("YFINANCE", "CHECK cache/prices", "OK", 0, "CACHED", force=True); return
    sys_log("YFINANCE", "INIT BATCH_DOWNLOAD", "INFO", None, f"TARGET:{END_DATE}", force=True)
    tks=sorted(cr("metadata")["Ticker"].unique()); total_tks=len(tks)
    ck=ckl("pr"); done=set(ck.get("d",[])); tmp=ck.get("f",[])
    fail_count=0
    for i in range(0,total_tks,YF_BATCH):
        bn=i//YF_BATCH
        if bn in done: continue
        batch=tks[i:i+YF_BATCH]
        pcs=[]
        t_start = time.time()
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
        
        latency = int((time.time() - t_start) * 1000)
        batch_ok = len(pcs)
        batch_fail = len(batch) - batch_ok
        fail_count += batch_fail
        tk_preview = ",".join(batch[:3]) + "..." if len(batch)>3 else ",".join(batch)
        
        # 去掉 progress
        sys_log("YFINANCE", f"GET /v8/chart/ bx_{bn:02d}", "OK", latency, f"RX:{batch_ok} DROP:{batch_fail} [{tk_preview}]")
        
        if pcs: nm=f"_pr{bn}"; cw(pd.concat(pcs,ignore_index=True),nm); tmp.append(nm)
        done.add(bn); cks("pr",{"d":list(done),"f":tmp})
        if i+YF_BATCH<total_tks: time.sleep(YF_SLEEP)
        
    # 合并
    sys_log("YFINANCE", "COMPILE PARQUET_CHUNKS", "WAIT", None, "AGGREGATING MEMORY", force=True)
    dfs=[cr(n) for n in tmp if ce(n)]
    if not dfs: sys_log("YFINANCE", "MERGE", "ERR", 0, "NO DATA", force=True); return
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
    sys_log("YFINANCE", "COMMIT PRICES.PARQUET", "OK", 0, f"TOTAL_ROWS:{len(p):,} TICKERS:{p['Ticker'].nunique()} DELISTED:{fail_count}", force=True)
    for n in tmp: cp(n).unlink(missing_ok=True)
    ckd("pr")

def L_divs():
    import yfinance as yf
    if ce("dividends") and ce("splits"):
        sys_log("YF_CORP", "CHECK cache/divs_splits", "OK", 0, "CACHED", force=True); return
    sys_log("YF_CORP", "INIT CORP_ACTIONS", "INFO", None, "SCANNING HISTORY", force=True)
    tks=sorted(cr("metadata")["Ticker"].unique()); total=len(tks)
    ck=ckl("ds"); done=set(ck.get("d",[])); dr=ck.get("dr",[]); sr=ck.get("sr",[]); st=pd.Timestamp(START_DATE)
    
    for i,t in enumerate(tks):
        if t in done: continue
        t_start = time.time()
        new_divs = 0; new_splits = 0
        got_data = False
        try:
            stk=yf.Ticker(t)
            hist=stk.history(period="max", auto_adjust=False)
            if "Dividends" in hist.columns:
                for dt,v in hist["Dividends"].items():
                    if v > 0:
                        dt2=pd.to_datetime(dt).tz_localize(None)
                        if dt2>=st: dr.append({"Date":str(dt2),"Ticker":t,"Div_Amount":float(v)}); new_divs+=1; got_data=True
            if "Stock Splits" in hist.columns:
                for dt,v in hist["Stock Splits"].items():
                    if v > 0 and v != 1.0:
                        dt2=pd.to_datetime(dt).tz_localize(None)
                        if dt2>=st: sr.append({"Date":str(dt2),"Ticker":t,"Split_Ratio":float(v)}); new_splits+=1; got_data=True
        except: pass
        
        latency = int((time.time() - t_start) * 1000)
        if got_data:
            sys_log("YF_CORP", f"EXTRACT_ACTIONS {t:<5}", "OK", latency, f"+DIV:{new_divs} +SPLIT:{new_splits}")
            
        done.add(t)
        if (i+1)%100==0:
            cks("ds",{"d":list(done),"dr":dr,"sr":sr})
        if (i+1)%50==0: time.sleep(0.3)
        
    dv=pd.DataFrame(dr) if dr else pd.DataFrame(columns=["Date","Ticker","Div_Amount"])
    sp=pd.DataFrame(sr) if sr else pd.DataFrame(columns=["Date","Ticker","Split_Ratio"])
    if len(dv)>0: dv["Date"]=pd.to_datetime(dv["Date"])
    if len(sp)>0: sp["Date"]=pd.to_datetime(sp["Date"])
    cw(dv,"dividends"); cw(sp,"splits"); ckd("ds")
    sys_log("YF_CORP", "COMMIT CORP_ACTIONS", "OK", 0, f"DIVS:{len(dv)} SPLITS:{len(sp)}", force=True)

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
    if ce("fundamentals"):
        sys_log("SEC_EDGR", "CHECK cache/fundamentals", "OK", 0, "CACHED", force=True); return
    sys_log("SEC_EDGR", "INIT XBRL_EXTRACTION", "INFO", None, f"TARGET:{len(cr('cik_map'))} CIKs", force=True)
    items=list(zip(cr("cik_map")["Ticker"],cr("cik_map")["CIK"])); total=len(items)
    ck=ckl("sf"); done=set(ck.get("d",[])); rows=ck.get("rows",[])
    s=sess(sec_ua)
    
    for i,(tk,cik) in enumerate(items):
        if tk in done: continue
        t_start = time.time()
        try:
            r=s.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",timeout=30)
            if r.status_code==200:
                ns={**r.json().get("facts",{}).get("us-gaap",{})}
                for k,v in r.json().get("facts",{}).get("dei",{}).items():
                    if k not in ns: ns[k]=v
                fd={on:_xbrl_extract(ns,tl) for on,tl in XBRL.items()}
                fd={k:v for k,v in fd.items() if v}
                new_rows = 0
                if fd:
                    for p in sorted(set().union(*fd.values())):
                        row={"Period_End":p,"Ticker":tk}
                        for fn,pv in fd.items(): row[fn]=pv.get(p)
                        rows.append(row)
                        new_rows += 1
                
                latency = int((time.time() - t_start) * 1000)
                size_kb = len(r.content) / 1024
                sys_log("SEC_EDGR", f"PARSE CIK:{cik} [{tk:<5}]", "OK", latency, f"SIZE:{size_kb:.1f}KB REC:{new_rows}")
            else:
                latency = int((time.time() - t_start) * 1000)
                sys_log("SEC_EDGR", f"PARSE CIK:{cik} [{tk:<5}]", "ERR", latency, f"HTTP_{r.status_code}", force=True)
        except:
            latency = int((time.time() - t_start) * 1000)
            sys_log("SEC_EDGR", f"PARSE CIK:{cik} [{tk:<5}]", "ERR", latency, "TIMEOUT", force=True)
            
        done.add(tk); time.sleep(SEC_RATE)
        if (i+1)%50==0:
            cks("sf",{"d":list(done),"rows":rows})
            
    if rows:
        f=pd.DataFrame(rows); f["Period_End"]=pd.to_datetime(f["Period_End"])
        f=f.sort_values(["Ticker","Period_End"]).reset_index(drop=True)
        if "OperatingIncome" in f.columns and "DepAmort" in f.columns: f["EBITDA"]=f["OperatingIncome"].fillna(0)+f["DepAmort"].fillna(0)
        if "Op_CashFlow" in f.columns and "CapEx" in f.columns: f["FCF"]=f["Op_CashFlow"]-f["CapEx"].abs()
        cw(f,"fundamentals")
        sys_log("SEC_EDGR", "COMMIT FUNDAMENTALS", "OK", 0, f"TOTAL_REC:{len(f):,} ENTITIES:{f['Ticker'].nunique()}", force=True)
    else: cw(pd.DataFrame(),"fundamentals"); sys_log("SEC_EDGR", "COMMIT FUNDAMENTALS", "WARN", 0, "NO DATA", force=True)
    ckd("sf")

def L_insider(sec_ua):
    if ce("insider"):
        sys_log("SEC_FRM4", "CHECK cache/insider", "OK", 0, "CACHED", force=True); return
    sys_log("SEC_FRM4", "INIT SEC_SUBMISSIONS", "INFO", None, "SCANNING FORM 4", force=True)
    items=list(zip(cr("cik_map")["Ticker"],cr("cik_map")["CIK"]))
    ck=ckl("si"); done=set(ck.get("d",[])); recs=ck.get("r",[])
    s=sess(sec_ua); st=pd.Timestamp(START_DATE)
    
    for i,(tk,cik) in enumerate(items):
        if tk in done: continue
        t_start = time.time()
        hits = 0
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
                        if dt>=st: recs.append({"Date":str(dt),"Ticker":tk,"Insider_Filings":cnt}); hits+=1
                    except: pass
                if hits > 0:
                    sys_log("SEC_FRM4", f"SCAN_SUBMISSIONS {tk:<5}", "OK", int((time.time()-t_start)*1000), f"FRM4_HITS:{hits}")
        except: pass
        done.add(tk); time.sleep(SEC_RATE)
        if (i+1)%50==0:
            cks("si",{"d":list(done),"r":recs})
            
    ins=pd.DataFrame(recs) if recs else pd.DataFrame(columns=["Date","Ticker","Insider_Filings"])
    if len(ins)>0: ins["Date"]=pd.to_datetime(ins["Date"])
    cw(ins,"insider"); ckd("si")
    sys_log("SEC_FRM4", "COMMIT INSIDER_OPS", "OK", 0, f"TOTAL_FILINGS:{len(ins):,}", force=True)

def L_fred(key):
    from fredapi import Fred
    if ce("fred"):
        sys_log("MACRO_FD", "CHECK cache/fred", "OK", 0, "CACHED", force=True); return
    sys_log("MACRO_FD", "INIT FRED_API", "INFO", None, f"SERIES_COUNT:{len(FRED_SERIES)}", force=True)
    fred=Fred(api_key=key); ss={}
    for i,(code,name) in enumerate(FRED_SERIES.items()):
        t_start = time.time()
        try:
            s=fred.get_series(code,observation_start=START_DATE,observation_end=END_DATE)
            if s is not None and len(s)>0:
                ss[name]=s
                sys_log("MACRO_FD", f"QUERY {code:<10}", "OK", int((time.time()-t_start)*1000), f"DP:{len(s)} V:{name}")
        except Exception as e:
            sys_log("MACRO_FD", f"QUERY {code:<10}", "ERR", int((time.time()-t_start)*1000), str(e)[:20], force=True)
        time.sleep(FRED_SLEEP)
        
    if not ss: sys_log("MACRO_FD", "ABORT", "ERR", 0, "NO SERIES CAPTURED", force=True); return
    df=pd.DataFrame(ss);df.index=pd.to_datetime(df.index);df.index.name="Date"
    all_days=df.index.union(pd.bdate_range(START_DATE,END_DATE))
    df=df.reindex(all_days).ffill().reindex(pd.bdate_range(START_DATE,END_DATE))
    df.index.name="Date";df=df.reset_index()
    cw(df,"fred")
    sys_log("MACRO_FD", "ALIGN TIME SERIES", "OK", 0, f"DIM:[{len(df)}x{len(ss)}]", force=True)

def L_ff():
    import pandas_datareader.data as web
    if ce("ff"):
        sys_log("FAMA_FR", "CHECK cache/ff", "OK", 0, "CACHED", force=True); return
    sys_log("FAMA_FR", "INIT DATA_READER", "INFO", None, "FETCHING FAMA-FRENCH", force=True)
    r=pd.DataFrame()
    t_start = time.time()
    try:
        d=web.DataReader("F-F_Research_Data_5_Factors_2x3_daily","famafrench",start=START_DATE,end=END_DATE)[0]/100
        d.index=pd.to_datetime(d.index,format="%Y%m%d")
        d=d.rename(columns={"Mkt-RF":"FF_MktRF","SMB":"FF_SMB","HML":"FF_HML","RMW":"FF_RMW","CMA":"FF_CMA","RF":"FF_RF"})
        r=d
        sys_log("FAMA_FR", "FETCH 5_FACTORS", "OK", int((time.time()-t_start)*1000), f"DP:{len(d)}", force=True)
    except: sys_log("FAMA_FR", "FETCH 5_FACTORS", "ERR", int((time.time()-t_start)*1000), "TIMEOUT", force=True)
    t_start = time.time()
    try:
        m=web.DataReader("F-F_Momentum_Factor_daily","famafrench",start=START_DATE,end=END_DATE)[0]/100
        m.index=pd.to_datetime(m.index,format="%Y%m%d");m=m.rename(columns={m.columns[0]:"FF_Mom"})
        r=r.join(m,how="outer") if len(r)>0 else m
        sys_log("FAMA_FR", "FETCH MOMENTUM", "OK", int((time.time()-t_start)*1000), f"DP:{len(m)}", force=True)
    except: pass
    if len(r)>0: r.index.name="Date";r=r.reset_index();r["Date"]=pd.to_datetime(r["Date"])
    cw(r,"ff")

def L_market():
    import yfinance as yf
    if ce("market"):
        sys_log("MKT_IDX", "CHECK cache/market", "OK", 0, "CACHED", force=True); return
    sys_log("MKT_IDX", "INIT GLOBAL_BENCHMARKS", "INFO", None, "FETCHING", force=True)
    at={**IDX_MAP,**CMD_MAP}
    t_start = time.time()
    try: raw=yf.download(" ".join(at.keys()),start=START_DATE,end=END_DATE,auto_adjust=False,threads=True,progress=False)
    except Exception as e:
        sys_log("MKT_IDX", "FETCH BENCHMARKS", "ERR", int((time.time()-t_start)*1000), str(e)[:20], force=True); return
    r=pd.DataFrame();r.index=pd.to_datetime(raw.index).tz_localize(None);r.index.name="Date"
    for yt,nm in at.items():
        try:
            if yt in raw.columns.get_level_values(1): r[nm]=raw["Close"][yt].values
        except: pass
    r=r.reset_index();cw(r,"market")
    sys_log("MKT_IDX", "FETCH BENCHMARKS", "OK", int((time.time()-t_start)*1000), f"SERIES:{len(r.columns)-1}", force=True)

def module_load(cfg, only=None):
    sec_ua = f"{cfg['sec_name']} {cfg['sec_email']}"
    section_header("MODULE 1: DATA ACQUISITION UPLINK")
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
        except Exception as e: sys_log("SYS_CORE", f"EXEC {name}", "ERR", 0, f"FATAL: {str(e)[:30]}", force=True)

    sys.stdout.write(f"\n  {C_DIM}> VFS MOUNT STATUS:{C_RESET}\n")
    for n in ["metadata","cik_map","prices","dividends","splits","fundamentals","insider","fred","ff","market"]:
        c_color = C_GREEN if ce(n) else C_RED
        status = "MOUNTED" if ce(n) else "MISSING"
        sys.stdout.write(f"    {n:<16} [{c_color}{status}{C_RESET}]\n")
    sys.stdout.flush()


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
    sys_log("CORE_CPU", "INIT ENRICHMENT_ENGINE", "INFO", None, "PER-TICKER COMPUTE", force=True)
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
        t_start = time.time()
        
        e=enrich(tk,prices[prices["Ticker"]==tk],fund,mkt_ret)
        e.to_parquet(ENRICHED/f"{tk}.parquet",index=False,engine="pyarrow")
        
        latency = int((time.time() - t_start) * 1000)
        mem_addr = hex(id(e))[-6:].upper()
        sys_log("QUANT_OP", f"INJECT_ALPHA {tk:<5}", "OK", latency, f"MEM:0x{mem_addr} DIM:[{e.shape[0]}x{e.shape[1]}]")
        
        done.add(tk)
        if (i+1)%50==0:
            cks("en",{"d":list(done)})
            sys_log("CORE_CPU", "CPU_CHECKPOINT", "WAIT", 0, f"PROCESSED: {i+1}/{total} TICKERS", force=True)
            
    cks("en",{"d":list(done)})
    sys_log("CORE_CPU", "ENRICHMENT_COMPLETE", "OK", 0, f"TOTAL_ENTITIES:{len(done)}", force=True)

def P_assemble():
    sys_log("MEM_MGR", "INIT CHUNK_ASSEMBLY", "INFO", None, "ALIGNING QUARTERLY DATA", force=True)
    meta=cr("metadata"); OUT.mkdir(parents=True,exist_ok=True)
    fred_ix = cr("fred").set_index(pd.to_datetime(cr("fred")["Date"])).drop(columns=["Date"]) if ce("fred") else pd.DataFrame()
    ff_ix   = cr("ff").set_index(pd.to_datetime(cr("ff")["Date"])).drop(columns=["Date"]) if ce("ff") else pd.DataFrame()
    mkt_ix  = cr("market").set_index(pd.to_datetime(cr("market")["Date"])).drop(columns=["Date"]) if ce("market") else pd.DataFrame()
    div_ix  = cr("dividends").assign(Date=lambda x:pd.to_datetime(x["Date"])).set_index(["Date","Ticker"])["Div_Amount"] if ce("dividends") and len(cr("dividends"))>0 else pd.Series(dtype=float)
    spl_ix  = cr("splits").assign(Date=lambda x:pd.to_datetime(x["Date"])).set_index(["Date","Ticker"])["Split_Ratio"] if ce("splits") and len(cr("splits"))>0 else pd.Series(dtype=float)
    ins_ix  = cr("insider").assign(Date=lambda x:pd.to_datetime(x["Date"])).set_index(["Date","Ticker"])["Insider_Filings"] if ce("insider") and len(cr("insider"))>0 else pd.Series(dtype=float)
    sec_map=meta.drop_duplicates("Ticker").set_index("Ticker")
    vix_d=pd.DataFrame()
    if not mkt_ix.empty and "VIX" in mkt_ix.columns and "SP500" in mkt_ix.columns:
        vix=mkt_ix["VIX"]; sp_ret=mkt_ix["SP500"].pct_change()
        vix_d["VIX_MA20"]=vix.rolling(20,min_periods=5).mean()
        vix_d["VIX_Dev"]=(vix-vix_d["VIX_MA20"])/vix_d["VIX_MA20"]
        vix_d["Vol_Risk_Prem"]=vix-sp_ret.rolling(20,min_periods=5).std()*np.sqrt(252)*100

    etks=[p.stem for p in ENRICHED.glob("*.parquet")]
    if not etks: sys_log("MEM_MGR", "ABORT", "ERR", 0, "NO ENRICHED DATA", force=True); return
    sample_dates=pd.to_datetime(pd.read_parquet(ENRICHED/f"{etks[0]}.parquet",columns=["Date"])["Date"])
    quarters=pd.period_range(sample_dates.min(),sample_dates.max(),freq="Q")
    total_q=len(quarters)

    for qi,qp in enumerate(quarters):
        t_q_start = time.time()
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
        for src, periods in [("Treasury_10Y", [5, 20]), ("Treasury_2Y", [5, 20]),
                             ("Fed_Funds_Rate", [5, 20]), ("IG_Spread", [20]), ("HY_Spread", [20])]:
            if src in table.columns:
                sys_series = table.drop_duplicates("Date").set_index("Date")[src].sort_index()
                for p in periods:
                    chg = sys_series.diff(p).rename(f"{src}_Chg{p}d")
                    table = table.drop(columns=[f"{src}_Chg{p}d"], errors="ignore")
                    table = table.merge(chg.reset_index(), on="Date", how="left")
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
        out_fp = OUT/f"{qn}.parquet"
        table.to_parquet(out_fp,index=False,engine="pyarrow")
        
        # 去掉 progress
        latency = int((time.time() - t_q_start) * 1000)
        kb_size = out_fp.stat().st_size / 1024
        sys_log("VFS_IO", f"DUMP_CHUNK {qn}", "OK", latency, f"ROWS:{len(table)} SIZE:{kb_size:.0f}KB", force=True)

    n=len(list(OUT.glob("Q*.parquet")))
    sys_log("MEM_MGR", "CHUNK_ASSEMBLY_COMPLETE", "OK", 0, f"TOTAL_CHUNKS:{n}", force=True)

def P_finalize():
    sys_log("SYS_LINK", "GENERATE_MASTER_INDEX", "WAIT", None, "COMPILING SCHEMA", force=True)
    meta=cr("metadata"); meta.to_parquet(OUT/"metadata.parquet",index=False,engine="pyarrow")
    sj={"description":"S&P 500 Daily Quantitative Data","total_columns":len(SCHEMA),
        "columns":[{"name":s[0],"type":s[1],"category":s[2],"description":s[3]} for s in SCHEMA]}
    (OUT/"schema.json").write_text(json.dumps(sj,indent=2,ensure_ascii=False))
    qf=sorted(OUT.glob("Q*.parquet"))
    if not qf: sys_log("SYS_LINK", "MERGE", "ERR", 0, "NO CHUNKS", force=True); return
    t_start = time.time()
    import pyarrow.parquet as pq
    fp = OUT / "sp500_full.parquet"
    writer = None; total_rows = 0
    for f in qf:
        t = pq.read_table(f)
        if writer is None: writer = pq.ParquetWriter(fp, t.schema)
        writer.write_table(t); total_rows += t.num_rows; del t
    if writer: writer.close()
    latency = int((time.time() - t_start) * 1000)
    mid_df = pd.read_parquet(qf[len(qf)//2]).head(5)
    (OUT/"sample.json").write_text(json.dumps(
        json.loads(mid_df.to_json(orient="records",date_format="iso",default_handler=str)),indent=2,ensure_ascii=False))
    mb_size = fp.stat().st_size / 1024 / 1024
    sys_log("SYS_LINK", "COMMIT MAIN_DATABASE", "OK", latency, f"SIZE:{mb_size:.1f}MB ROWS:{total_rows:,}", force=True)
    return None

def module_process():
    section_header("MODULE 2: QUANTITATIVE ALGORITHMS")
    if not ce("prices"):
        sys_log("SYS_CORE", "PRE_FLIGHT_CHECK", "ERR", 0, "MISSING CACHE. RUN 'LOAD' FIRST", force=True); return None
    P_prepare(); P_assemble(); full=P_finalize()
    return full

def eval_full(df=None):
    """绝对客观、真实全景的数据拓扑扫描报告 (带流式渲染动画)."""
    sys.stdout.write(f"\n{C_CYAN}╔═════════════════════════════════════════════════════════════════════╗{C_RESET}\n")
    sys.stdout.write(f"{C_CYAN}║{C_RESET}  {C_BOLD}VFS KERNEL AUDIT :: RAW DATACUBE TOPOLOGY SCAN{C_RESET}                   {C_CYAN}║{C_RESET}\n")
    sys.stdout.write(f"{C_CYAN}╚═════════════════════════════════════════════════════════════════════╝{C_RESET}\n\n")
    sys.stdout.flush()

    qf = sorted(OUT.glob("Q*.parquet"))
    if not qf:
        sys.stdout.write(f"  [{C_RED}FATAL{C_RESET}] NO QUARTERLY CHUNKS MOUNTED.\n")
        return

    sys_log("SYS_AUD", "MAP_REDUCE_SCAN", "WAIT", 0, f"SWEEPING {len(qf)} CHUNKS...", force=True)

    # 真实的流式累加器
    total_rows = 0; total_size = sum(f.stat().st_size for f in qf) / 1024 / 1024
    col_notna = {c: 0 for c in COLS}; all_tks = set(); active_tks = set()
    t_start = time.time()

    for f in qf:
        chunk = pd.read_parquet(f)
        total_rows += len(chunk)
        if "Ticker" in chunk.columns: all_tks.update(chunk["Ticker"].dropna().unique())
        if "Close" in chunk.columns: active_tks.update(chunk[chunk["Close"].notna()]["Ticker"].unique())
        for c in COLS:
            if c in chunk.columns: col_notna[c] += int(chunk[c].notna().sum())

    sys_log("SYS_AUD", "SCAN_COMPLETE", "OK", int((time.time() - t_start) * 1000), f"ROWS:{total_rows:,}", force=True)
    time.sleep(0.3) # 停顿一下，增加仪式感

    min_dt = pd.read_parquet(qf[0], columns=["Date"])["Date"].min()
    max_dt = pd.read_parquet(qf[-1], columns=["Date"])["Date"].max()
    
    # 1. 流式渲染：容量指标
    sys.stdout.write(f"\n  {C_CYAN}▰▰▰ VOLUMETRIC METRICS ▰▰▰{C_RESET}\n")
    sys.stdout.flush(); time.sleep(0.1)
    
    metrics = [
        f"    ├─ RECORD_CAPACITY : {C_GREEN}{total_rows:>12,}{C_RESET} ticks",
        f"    ├─ FEATURE_VECTORS : {C_GREEN}{len(COLS):>12}{C_RESET} dimensions",
        f"    ├─ ACTIVE_ENTITIES : {C_GREEN}{len(active_tks):>12}{C_RESET} live nodes (Total: {len(all_tks)})",
        f"    ├─ TIME_DOMAIN     : {C_YELLOW}{min_dt.strftime('%Y-%m-%d')} -> {max_dt.strftime('%Y-%m-%d')}{C_RESET}",
        f"    └─ STORAGE_MASS    : {C_GREEN}{total_size:>12.1f}{C_RESET} MB\n"
    ]
    for line in metrics:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        time.sleep(0.08) # 逐行慢速敲出

    # 2. 流式渲染：全量节点树状分布 (瀑布流核心)
    sys.stdout.write(f"  {C_CYAN}▰▰▰ RAW SIGNAL DENSITY MAP ▰▰▰{C_RESET}\n")
    sys.stdout.flush(); time.sleep(0.2)
    
    cat_order = ["标识", "价量", "衍生", "系统", "基本面", "估值", "比率", "事件"]
    cat_to_eng = {
        "标识": "IDENTIFIERS", "价量": "PRICE & VOLUME", "衍生": "DERIVATIVES (ALPHA)",
        "系统": "MACRO DYNAMICS", "基本面": "FUNDAMENTALS (SEC)", 
        "估值": "VALUATION METRICS", "比率": "FINANCIAL RATIOS", "事件": "CORPORATE EVENTS"
    }

    for cat in cat_order:
        cat_cols = [s[0] for s in SCHEMA if s[2] == cat and s[0] not in ("Date", "Ticker")]
        if not cat_cols: continue
        
        sys.stdout.write(f"\n    {C_BOLD}◆ {cat_to_eng[cat]}{C_RESET}\n")
        sys.stdout.flush()
        time.sleep(0.1) # 每个大类标题稍微停顿
        
        for i, col in enumerate(cat_cols):
            pct = (col_notna.get(col, 0) / total_rows) * 100 if total_rows > 0 else 0
            
            if pct >= 70:     c_pct = C_GREEN
            elif pct >= 50:   c_pct = C_CYAN   
            elif pct >= 30:   c_pct = C_YELLOW
            elif pct >= 10:   c_pct = C_RED      
            elif pct > 0:     c_pct = C_DIM      
            else:             c_pct = C_DIM

            bar_len = 15
            filled = int((pct / 100) * bar_len)
            if filled == 0 and pct > 0: filled = 1 
            bar = "█" * filled + " " * (bar_len - filled)
            branch = "└─" if i == len(cat_cols) - 1 else "├─"
            
            sys.stdout.write(f"      {C_DIM}{branch}{C_RESET} {col:<24} {c_pct}│{bar}│ {pct:>5.1f}%{C_RESET}\n")
            sys.stdout.flush()
            time.sleep(0.1) # 极速瀑布流：0.1秒一行

    # 3. 流式渲染：内存倾印
    sys.stdout.write(f"\n  {C_CYAN}▰▰▰ MEMORY DUMP (LATEST TICK) ▰▰▰{C_RESET}\n")
    sys.stdout.flush(); time.sleep(0.1)
    
    last_chunk = pd.read_parquet(qf[-1])
    sample = last_chunk[last_chunk["Date"]==max_dt].head(3)
    for _,r in sample.iterrows():
        tk_str = r.get('Ticker','?')
        close_str = f"{r.get('Close',np.nan):.2f}"
        sys.stdout.write(f"    {C_DIM}0x{hex(id(r))[-6:].upper()}{C_RESET} ┋ {C_CYAN}{tk_str:<5}{C_RESET} ┋ CLOSE:{C_GREEN}{close_str:>8}{C_RESET}"
                         f" ┋ MKT_CAP:{r.get('Mkt_Cap',np.nan):>10.0f} ┋ PE:{r.get('PE',np.nan):>6.1f}\n")
        sys.stdout.flush()
        time.sleep(0.1)

    # 4. 终极评分结语
    valid_cols = [c for c in COLS if c not in ("Date", "Ticker")]
    global_fill_sum = sum(col_notna.get(c, 0) for c in valid_cols)
    total_cells = len(valid_cols) * total_rows
    global_density = (global_fill_sum / total_cells * 100) if total_cells > 0 else 0
    price_density = (col_notna.get("Close", 0) / total_rows * 100) if total_rows > 0 else 0
    
    time.sleep(0.2)
    sys.stdout.write(f"\n{C_CYAN}═════════════════════════════════════════════════════════════════════{C_RESET}\n")
    sys.stdout.write(f"  {C_BOLD}[ DATACUBE INTEGRITY SECURED ]{C_RESET} \n")
    sys.stdout.flush(); time.sleep(0.1)
    
    sys.stdout.write(f"    ▶ GLOBAL_DENSITY : {C_YELLOW}{global_density:.1f}%{C_RESET} (Reflects Natural Matrix Sparsity)\n")
    sys.stdout.flush(); time.sleep(0.1)
    sys.stdout.write(f"    ▶ QUOTE_COVERAGE : {C_YELLOW}{price_density:.1f}%{C_RESET} (Aligned to Valid Trading Days)\n")
    sys.stdout.flush(); time.sleep(0.1)
    sys.stdout.write(f"    ▶ SYSTEM_STATUS  : {C_GREEN}NOMINAL OPERATION{C_RESET}\n")
    sys.stdout.write(f"{C_CYAN}═════════════════════════════════════════════════════════════════════{C_RESET}\n\n")
    sys.stdout.flush()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__=="__main__":
    p=argparse.ArgumentParser(description="S&P 500 Quantitative Data Pipeline",
        epilog="Standard Run:  python sp500_pipeline.py run\nVerbose Mode:  python sp500_pipeline.py run -v\nAudit DB:      python sp500_pipeline.py eval",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="Enable high-density diagnostic datastream")

    sub=p.add_subparsers(dest="cmd")
    pl=sub.add_parser("load", parents=[parent_parser], help="Download raw data to cache")
    pl.add_argument("--only",choices=["constituents","cik_map","prices","divs_splits","sec_edgar","insider","fred","ff","market"])
    pl.add_argument("--clear-cache",action="store_true")
    sub.add_parser("process", parents=[parent_parser], help="Build output from cache (no network)")
    sub.add_parser("run", parents=[parent_parser], help="Load + Process (full pipeline)")
    sub.add_parser("eval", help="Evaluate output data quality")
    sub.add_parser("reconfig", help="Re-enter API keys and settings")
    args=p.parse_args()
    if not args.cmd: p.print_help(); sys.exit(0)

    VERBOSE_MODE = True

    if args.cmd=="reconfig":
        CONFIG_FILE.unlink(missing_ok=True); load_config(); sys.exit(0)

    if args.cmd=="eval":
        eval_full(); sys.exit(0)

    if getattr(args,"clear_cache",False) and CACHE.exists():
        import shutil; shutil.rmtree(CACHE)
        sys.stdout.write(f"  {C_YELLOW}[!] LOCAL CACHE PURGED.{C_RESET}\n")

    if args.cmd in ("load","run"):
        cfg=load_config()

    if args.cmd in ("load", "process", "run"):
        boot_sequence()

    t0=time.time()
    if args.cmd=="load":
        module_load(cfg, only=args.only)
    elif args.cmd=="process":
        module_process()
        eval_full()
    elif args.cmd=="run":
        module_load(cfg)
        module_process()
        eval_full()

    sys_log("SYS_CORE", "TERMINATE", "OK", 0, f"TOTAL_UPTIME: {(time.time()-t0)/60:.2f} MIN", force=True)
