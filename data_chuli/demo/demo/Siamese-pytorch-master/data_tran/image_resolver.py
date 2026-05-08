import os
import threading
from typing import Any, Optional, Tuple

from data_tran.client import request_image


class ImagePathResolver:
    def __init__(
        self,
        server_ip: Optional[str] = None,
        server_port: Optional[int] = None,
        local_root: Optional[str] = None,
    ) -> None:
        self._lock = threading.Lock()
        self.server_ip = server_ip or os.environ.get("IMAGE_SERVER_IP", "10.100.2.229")
        self.server_port = int(server_port or os.environ.get("IMAGE_SERVER_PORT", "5000"))
        self.local_root = local_root or os.environ.get("IMAGE_LOCAL_ROOT", r"D:\\")

    def _ensure_local_path(self, remote_path: str) -> str:
        drive, path_without_drive = os.path.splitdrive(remote_path)
        rel_path = path_without_drive.lstrip("\\/")
        base = self.local_root
        if base.endswith(":"):
            base = base + os.sep
        local_path = os.path.join(base, rel_path)
        return os.path.abspath(local_path)

    def _iter_remote_candidates(self, remote_path: str):
        seen = set()

        def _add(p: str):
            p2 = p.strip()
            if not p2:
                return
            if p2 in seen:
                return
            seen.add(p2)
            return p2

        first = _add(remote_path)
        if first:
            yield first

        if "\\" in remote_path:
            p = _add(remote_path.replace("\\", "/"))
            if p:
                yield p

        if "/" in remote_path:
            p = _add(remote_path.replace("/", "\\"))
            if p:
                yield p

        if len(remote_path) >= 2 and remote_path[1] == ":":
            rest = remote_path[2:].lstrip("\\/")
            p = _add(remote_path[:2] + "/" + rest.replace("\\", "/"))
            if p:
                yield p

    def fetch_to_local(self, remote_path: Any) -> Tuple[bool, Optional[str], Optional[str]]:
        if not isinstance(remote_path, str) or not remote_path.strip():
            return False, None, "remote_path must be a non-empty string"

        remote_path = remote_path.strip()

        with self._lock:
            try:
                last_err: Optional[str] = None
                for candidate in self._iter_remote_candidates(remote_path):
                    local_path = self._ensure_local_path(candidate)
                    save_dir = os.path.dirname(local_path)
                    if not save_dir:
                        last_err = "invalid local path"
                        continue

                    os.makedirs(save_dir, exist_ok=True)
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        return True, local_path, None

                    ok = request_image(self.server_ip, self.server_port, candidate, save_dir)
                    if not ok:
                        last_err = f"request_image failed for {candidate}"
                        continue

                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        return True, local_path, None

                    alt_path = os.path.join(save_dir, os.path.basename(candidate))
                    if os.path.exists(alt_path) and os.path.getsize(alt_path) > 0:
                        return True, os.path.abspath(alt_path), None

                    last_err = f"download reported success but file not found locally for {candidate}"

                return False, None, last_err or "remote fetch failed"
            except Exception as e:
                return False, None, f"remote fetch exception: {e}"
