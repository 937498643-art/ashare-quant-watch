# quant_stock_watch

A 股个股量化盯盘系统 V1。

## 项目边界

本项目只做行情读取、个股筛选、条件复核、风险提示和提醒。

本项目严禁自动交易，不接入买入、卖出、撤单、委托、银证转账等交易功能，也不保存证券账号、交易密码、验证码。

## 快速开始

```powershell
cd "C:\Users\admin\Documents\A 股个股量化盯盘系统 V1\quant_stock_watch"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py --once
```

循环盯盘：

```powershell
python main.py --watch
```

## 如何本地启动看板

先进入项目目录并激活虚拟环境：

```powershell
cd "C:\Users\admin\Documents\A 股个股量化盯盘系统 V1\quant_stock_watch"
.\.venv\Scripts\activate
```

启动 Streamlit 服务：

```powershell
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

然后在浏览器打开：

```text
http://127.0.0.1:8501
```

如果 8501 被占用，请改用 8502：

```powershell
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8502
```

也可以使用项目提供的 PowerShell 启动脚本。脚本会检查 8501 是否被占用；如果被占用，会提示你改用 8502：

```powershell
.\start_dashboard.ps1
```

## 输出文件

- 最新候选股：`data/latest_candidates.csv`
- SQLite 历史记录：`data/quant_watch.db`
- 日志：`logs/app.log`

## 数据源

第一版使用 AKShare / 东方财富公开行情数据。

预留后续数据源：

- Tushare
- iFinD
- QMT

## 邮件提醒

邮件默认关闭。启用前请在 `config.yaml` 配置 SMTP 信息，并通过 `.env` 或环境变量保存邮箱授权码。

不要把邮箱授权码、证券账号、交易密码、验证码写入代码。
