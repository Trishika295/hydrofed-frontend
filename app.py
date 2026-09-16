import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

target_script = os.path.join(REPO_ROOT, "cdss", "app.py")

def is_running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

if is_running_in_streamlit():
    with open(target_script, "r", encoding="utf-8") as f:
        code = compile(f.read(), target_script, "exec")
        exec(code, globals())
else:
    print(f"Starting HydroFed-ICAF Streamlit Portal from {target_script}...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        target_script,
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping HydroFed-ICAF...")
