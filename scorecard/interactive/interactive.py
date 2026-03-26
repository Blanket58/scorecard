import os
import platform
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import joblib


def _load_data_raw():
    return joblib.load('model.pkl')


@contextmanager
def _streamlit_app(script_path, port=8501):
    """上下文管理器：启动并自动清理 Streamlit 进程"""
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        script_path,
        "--server.port", str(port),
        "--server.headless", "true"
    ]
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid
    process = subprocess.Popen(cmd, **kwargs)
    print(f"🚀 Streamlit starts at http://localhost:{port}")

    try:
        yield process
    finally:
        if platform.system() == "Windows":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)


def interactive(woe_encoder, X, y):
    joblib.dump((woe_encoder, X, y), 'model.pkl')
    file = Path(Path(__file__).resolve().parent / 'app.py').absolute()
    with _streamlit_app(file, port=8502) as proc:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    woe_encoder, _, _ = joblib.load('model.pkl')
    return woe_encoder
