# MacBook Pro Setup

This directory is a portable project copy. It does not contain a Python virtual
environment or a real `.env` file.

## Install

```bash
cd ~/Documents/quant_stock_watch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in only the credentials you need. Do not
commit `.env` to a public repository.

## Run One Scan

```bash
python main.py --once
```

## Start the Dashboard

```bash
python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501` in a browser.

## Windows Command Difference

Windows commonly uses:

```powershell
.\.venv\Scripts\python.exe
```

On macOS, use:

```bash
./.venv/bin/python
```

or activate the virtual environment and use `python` directly.

## Cached Data

The migration copy includes local user data and Tushare cache files. They can be
used immediately, but their dates are historical. Refresh data only through the
project's existing cache scripts when appropriate.
