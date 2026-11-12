"""Fixed paths and network settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"

# The server only ever listens on the loopback interface, so nothing else on the
# network (other computers, phones on the same Wi-Fi) can reach it. This is
# deliberately not configurable.
HOST = "127.0.0.1"
PORT = 7860
ALLOWED_HOSTS = ("127.0.0.1", "localhost")
URL = f"http://{HOST}:{PORT}"

# Header the web page sends with every state-changing request. Browsers won't let
# another website add a custom header to a cross-site request, so this blocks
# other sites from driving the app through your browser.
CLIENT_HEADER = "X-FaceSwap-Studio"
