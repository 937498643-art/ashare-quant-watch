# A股量化选股与盯盘系统

> 当前版本：**V5.9 stable**

## 项目简介

这是一个基于 Python 的 A 股量化分析系统，用于全市场行情扫描、因子评分、候选池观察和策略跟踪。系统将基础评分、交易质量、风险提示及数据完整性分层展示，帮助用户复核当天值得重点关注的股票。

项目只提供分析、筛选、复盘和提醒能力，**不包含自动交易、下单、撤单、券商账户接入或资金划转功能**。

## 核心功能

1. **全市场股票扫描**：按当日行情构建 A 股股票池并完成批量分析。
2. **多因子评分**：结合资金攻击、趋势结构、换手参与、量能增量和涨停活跃度等维度生成 V5.9 基础评分。
3. **Top50 交易观察池**：展示全市场评分与交易质量筛选后的重点观察股票。
4. **今日交易候选池**：基于 V5.9 Top50 进行数据完整性、风险、停牌、T+1 和买点状态复核。
5. **换手率、量比、成交额分析**：统一展示实时或计算得到的换手率、量比及成交额信息，并保留数据来源提示。
6. **策略跟踪**：将候选股票加入本地策略跟踪池，保存入选快照、收益、最大回撤、状态和每日历史记录。
7. **自选股管理**：支持本地维护自选股与持仓观察信息，不读取券商真实持仓。
8. **数据源管理**：支持公开行情源及本地缓存，包含 AKShare、Tushare、JQData 等数据接口的接入与状态提示。

## 技术栈

- Python 3
- Streamlit
- Pandas / NumPy
- AKShare
- Tushare
- JQData
- Plotly / Altair

## 安装方法

### 1. 获取项目并进入目录

```bash
git clone https://github.com/937498643-art/ashare-quant-watch.git
cd ashare-quant-watch
```

### 2. 创建并激活虚拟环境

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 配置私密环境变量

复制示例文件并按实际账号填写本机 `.env`：

```bash
cp .env.example .env
```

`.env` 可能包含 Tushare Token、JQData 账号等私密配置，已被 Git 忽略，**不要提交或上传该文件**。

Windows PowerShell 可使用：

```powershell
Copy-Item .env.example .env
```

## 运行方法

### 单次扫描

在激活虚拟环境后执行：

```bash
python main.py --once
```

### 循环盯盘

```bash
python main.py --watch
```

### 启动 Dashboard

建议先完成一次扫描，再启动看板：

```bash
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

浏览器访问：<http://127.0.0.1:8501>

## 项目目录说明

```text
.
├── main.py                 # 主扫描入口
├── dashboard/              # Streamlit 看板与展示逻辑
├── core/                   # 评分、指标、筛选、风险与策略模块
├── data_sources/           # 行情、基础数据和本地缓存读取接口
├── storage/                # 本地 CSV / SQLite / 自选与策略跟踪存储
├── alerts/                 # 邮件提醒等通知模块
├── scripts/                # 缓存构建、诊断、回测与检查脚本
├── config.yaml             # 系统功能开关与运行配置
├── requirements.txt        # Python 依赖
├── .env.example            # 私密配置示例（可提交）
└── data/                   # 本地行情、缓存、输出和用户数据（不提交）
```

## 本地数据与安全说明

- `.env`、`data/`、虚拟环境、数据库、日志和个人自选/持仓数据均不应提交到 GitHub。
- 邮件授权码、证券账户、交易密码、验证码等敏感信息不得写入代码或配置示例。
- `config/watchlist.yaml` 用于个人自选与持仓观察，默认不提交。
- 系统不具备自动交易能力；所有结果仅用于研究、观察和风险复核，不构成投资建议。

## 后续规划

- 持续积累策略跟踪样本，验证不同评分区间、换手率、量比和入选原因的历史有效性。
- 完善历史缓存与回测报告，提升跨市场环境下的稳定性评估。
- 优化 Dashboard 的统计、筛选、详情和复盘体验。
- 增强多数据源容错、状态诊断和本地缓存管理能力。
- 在保持只读分析边界的前提下，持续完善风险提示与研究辅助能力。

## 版本标签

当前稳定版本标签：`v5.9-stable`。
