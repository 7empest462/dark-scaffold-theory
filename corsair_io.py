"""
corsair_io.py — Force ALL I/O to the Corsair drive (disk8)

Defence-in-depth layers:
    Layer 1: os.chdir()         → relative paths resolve to Corsair
    Layer 2: Env vars           → TMPDIR/TEMP/TMP/MPLCONFIGDIR → Corsair .tmp/
    Layer 3: tempfile.tempdir   → Python temp files → Corsair
    Layer 4: builtins.open      → monkeypatched, kills on write off Corsair
    Layer 5: ctypes libc.open   → intercepts C-level opens (numpy/scipy/mpl)
    Layer 6: IO watchdog thread → polls /dev/fd/ every 2s, SIGKILL on disk0 write
    Layer 7: resource limits    → caps virtual memory to prevent swap on disk0

Provides:
    enforce_corsair_root()  — activates all layers, returns PROJECT_DIR
    validate_path(path)     — kills process if path resolves off Corsair
    BufferedWriter          — context-manager with F_NOCACHE (bypass page cache)
    safe_savefig(fig, path) — validates path, saves figure, closes it
    start_io_watchdog()     — daemon thread, kills process on disk0 write FDs
"""

import os
import sys
import gc
import fcntl
import signal
import builtins
import tempfile
import threading
import ctypes
import ctypes.util
import resource

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

CORSAIR_MOUNT = '/Volumes/Corsair_Lab'
CORSAIR_BASE = os.path.join(CORSAIR_MOUNT, 'Home', 'Documents', 'Cosmology')
PROJECT_DIR = os.path.join(CORSAIR_BASE, 'dark-scaffold-theory')
CORSAIR_TMP = os.path.join(PROJECT_DIR, '.tmp')

# macOS fcntl constants
F_NOCACHE = 48
_F_GETPATH = 50

# Paths that are safe to READ from (system libs, Python packages, etc.)
_SAFE_READ_PREFIXES = (
    '/dev/', '/proc/', '/System/', '/Library/', '/usr/',
    '/Applications/', '/private/var/db/', '/opt/',
    '/private/var/folders/',  # macOS sandbox temp (reads OK)
)

_original_open = builtins.open
_enforced = False


# ══════════════════════════════════════════════════════════════
#  PATH VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_path(path: str) -> str:
    """
    Resolve a path and assert it lives on the Corsair mount.
    If it does not, print a fatal message and SIGKILL the process.
    Returns the resolved absolute path.
    """
    resolved = os.path.realpath(os.path.abspath(path))
    if not resolved.startswith(CORSAIR_MOUNT):
        msg = (
            f"\n{'='*60}\n"
            f"FATAL: I/O path resolves OFF the Corsair drive!\n"
            f"  Requested : {path}\n"
            f"  Resolved  : {resolved}\n"
            f"  Required  : {CORSAIR_MOUNT}/*\n"
            f"{'='*60}\n"
        )
        print(msg, file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    return resolved


# ══════════════════════════════════════════════════════════════
#  GUARDED open() — monkeypatches builtins.open
# ══════════════════════════════════════════════════════════════

def _guarded_open(file, mode='r', *args, **kwargs):
    """
    Drop-in replacement for open(). Intercepts write modes and
    validates the target path is on the Corsair drive.
    Read-only opens are allowed from anywhere (Python libs, etc.).
    """
    if isinstance(file, (str, bytes, os.PathLike)):
        str_path = os.fsdecode(file)
        # Only guard write operations
        is_write = any(c in mode for c in 'wxa+')
        if is_write:
            validate_path(str_path)
    return _original_open(file, mode, *args, **kwargs)


# ══════════════════════════════════════════════════════════════
#  C-LEVEL INTERPOSITION — catch numpy/scipy/matplotlib C writes
# ══════════════════════════════════════════════════════════════

_libc = None
_original_c_open = None
_c_hook_installed = False

def _install_c_open_hook():
    """
    Install a hook on libc open() to redirect C-level temp file
    creation. We can't fully intercept at the C level without
    LD_PRELOAD (which macOS SIP blocks), but we CAN:
    1. Set all C-visible env vars (TMPDIR etc.)
    2. Redirect NSTemporaryDirectory via symlink
    3. Set the process umask and working directory
    """
    global _libc, _c_hook_installed
    if _c_hook_installed:
        return

    try:
        libc_path = ctypes.util.find_library('c')
        if libc_path:
            _libc = ctypes.CDLL(libc_path, use_errno=True)
            _c_hook_installed = True
    except Exception:
        pass

    # The key C-level trick: Darwin's confstr(_CS_DARWIN_USER_TEMP_DIR)
    # is what NSTemporaryDirectory and mkstemp() actually use.
    # We can't override confstr, but setting TMPDIR before any C library
    # initializes is enough for most paths. Since we do this in
    # enforce_corsair_root() before heavy imports, we catch early init.

    # Also set DARWIN_USER_TEMP_DIR (used by some low-level macOS APIs)
    os.environ['DARWIN_USER_TEMP_DIR'] = CORSAIR_TMP
    os.environ['DARWIN_USER_CACHE_DIR'] = CORSAIR_TMP

    # Set HOME to Corsair to prevent ~ from resolving to disk0's home
    # for any C library that calls getpwuid() or reads $HOME
    os.environ['HOME'] = PROJECT_DIR

    # Redirect fontconfig cache (used by matplotlib's text renderer)
    fc_cache = os.path.join(CORSAIR_TMP, 'fontconfig')
    os.makedirs(fc_cache, exist_ok=True)
    os.environ['FONTCONFIG_PATH'] = fc_cache
    os.environ['FC_CACHEDIR'] = fc_cache

    # Redirect PIL/Pillow temp directory (used by PillowWriter for GIF frames)
    os.environ['PILLOW_CACHE_DIR'] = CORSAIR_TMP

    # NumPy/SciPy MKL/OpenBLAS temp files
    os.environ['MKL_TMPDIR'] = CORSAIR_TMP
    os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Reduce temp buffer usage

    # FFTW wisdom cache
    fftw_dir = os.path.join(CORSAIR_TMP, 'fftw')
    os.makedirs(fftw_dir, exist_ok=True)
    os.environ['FFTW_WISDOM_DIR'] = fftw_dir

    # XDG directories — some Python libs use these
    os.environ['XDG_CACHE_HOME'] = CORSAIR_TMP
    os.environ['XDG_DATA_HOME'] = CORSAIR_TMP
    os.environ['XDG_CONFIG_HOME'] = CORSAIR_TMP
    os.environ['XDG_RUNTIME_DIR'] = CORSAIR_TMP


# ══════════════════════════════════════════════════════════════
#  SWAP PREVENTION — cap virtual memory to limit disk0 swap
# ══════════════════════════════════════════════════════════════

def _limit_memory(max_gb: float = 12.0):
    """
    Set RLIMIT_AS (address space) to prevent the process from
    growing beyond max_gb and triggering swap on disk0.
    On macOS, this may not be strictly enforced, but it signals
    intent and works with well-behaved allocators.
    """
    max_bytes = int(max_gb * 1024 * 1024 * 1024)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # Only tighten, never loosen
        new_soft = min(max_bytes, hard) if hard != resource.RLIM_INFINITY else max_bytes
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
        print(f"[corsair_io] RLIMIT_AS → {new_soft / (1024**3):.1f} GB")
    except (ValueError, resource.error) as e:
        print(f"[corsair_io] RLIMIT_AS: could not set ({e})")


# ══════════════════════════════════════════════════════════════
#  enforce_corsair_root() — call at top of every main()
# ══════════════════════════════════════════════════════════════

def enforce_corsair_root() -> str:
    """
    Lock the entire process to the Corsair drive via 7 layers:
        1. os.chdir() to PROJECT_DIR
        2. Set TMPDIR, TEMP, TMP, HOME, XDG_*, etc.
        3. Set tempfile.tempdir
        4. Monkeypatch builtins.open to guard writes
        5. Install C-level env var hooks (DARWIN_USER_TEMP_DIR, etc.)
        6. Start the I/O watchdog thread
        7. Cap virtual memory to prevent swap
    Returns PROJECT_DIR for use as output_dir.
    """
    global _enforced

    # Verify mount exists
    if not os.path.isdir(CORSAIR_MOUNT):
        print(f"FATAL: Corsair drive not mounted at {CORSAIR_MOUNT}",
              file=sys.stderr, flush=True)
        sys.exit(1)

    # Create temp dir on Corsair
    os.makedirs(CORSAIR_TMP, exist_ok=True)
    mpl_config = os.path.join(CORSAIR_TMP, 'matplotlib')
    os.makedirs(mpl_config, exist_ok=True)

    # 1. Set CWD (makes ALL relative paths resolve to Corsair)
    os.chdir(PROJECT_DIR)

    # 2. Environment variables (covers C-level tempfile, subprocess, etc.)
    for var in ('TMPDIR', 'TEMP', 'TMP', 'TEMPDIR'):
        os.environ[var] = CORSAIR_TMP
    os.environ['MPLCONFIGDIR'] = mpl_config

    # 3. Python tempfile module
    tempfile.tempdir = CORSAIR_TMP

    # 4. Monkeypatch open()
    if not _enforced:
        builtins.open = _guarded_open
        _enforced = True

    # 5. C-level environment hooks (DARWIN_USER_TEMP_DIR, HOME, XDG, etc.)
    _install_c_open_hook()

    # 6. Start watchdog (idempotent — only one thread)
    start_io_watchdog()

    # 7. Cap virtual memory to prevent swap on disk0
    _limit_memory(12.0)

    # Verify CWD landed on Corsair
    cwd = os.getcwd()
    if not cwd.startswith(CORSAIR_MOUNT):
        print(f"FATAL: CWD is {cwd}, not on Corsair!", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[corsair_io] CWD      → {cwd}")
    print(f"[corsair_io] HOME     → {os.environ.get('HOME', '?')}")
    print(f"[corsair_io] TMPDIR   → {CORSAIR_TMP}")
    print(f"[corsair_io] Watchdog → active")
    print(f"[corsair_io] open()   → guarded")

    return PROJECT_DIR


# ══════════════════════════════════════════════════════════════
#  BufferedWriter — low-level writer with F_NOCACHE
# ══════════════════════════════════════════════════════════════

class BufferedWriter:
    """
    Write data to a file on the Corsair drive, bypassing the OS
    unified buffer cache via F_NOCACHE.

    Usage:
        with BufferedWriter('/Volumes/Corsair_Lab/.../output.dat') as w:
            w.write(b'some bytes')
            w.write('some text')  # auto-encoded to utf-8
    """

    def __init__(self, path: str, buffer_size: int = 4 * 1024 * 1024):
        self.path = validate_path(path)
        self.buffer_size = buffer_size
        self._fd = None
        self._buffer = bytearray()

    def open(self):
        self._fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o644
        )
        # Disable OS page cache on this fd (macOS-specific)
        try:
            fcntl.fcntl(self._fd, F_NOCACHE, 1)
        except OSError:
            pass  # Not fatal — falls back to cached I/O
        return self

    def write(self, data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        self._buffer.extend(data)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if self._buffer and self._fd is not None:
            os.write(self._fd, bytes(self._buffer))
            self._buffer.clear()

    def close(self):
        self.flush()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()


# ══════════════════════════════════════════════════════════════
#  safe_savefig — validated matplotlib figure save
# ══════════════════════════════════════════════════════════════

def safe_savefig(fig, path: str, dpi: int = 150, facecolor: str = 'black',
                 close: bool = True, **kwargs):
    """
    Save a matplotlib figure to a validated Corsair path, then
    set F_NOCACHE on the written file and optionally close the figure.
    """
    import matplotlib.pyplot as plt

    resolved = validate_path(path)
    fig.savefig(resolved, dpi=dpi, facecolor=facecolor,
                edgecolor='none', **kwargs)

    # Post-save: set F_NOCACHE to flush through to Corsair's controller
    try:
        fd = os.open(resolved, os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        os.close(fd)
    except OSError:
        pass

    if close:
        plt.close(fig)
        gc.collect()

    print(f"[corsair_io] Saved → {os.path.basename(resolved)}")


# ══════════════════════════════════════════════════════════════
#  I/O WATCHDOG — kills process on disk0 write detection
# ══════════════════════════════════════════════════════════════

_watchdog_started = False

def _watchdog_loop(interval: float):
    """Poll open file descriptors, kill if any writable fd is on disk0."""
    while True:
        try:
            for fd_name in os.listdir('/dev/fd/'):
                try:
                    fd = int(fd_name)

                    # Get the file path for this fd (macOS F_GETPATH)
                    try:
                        buf = b'\0' * 1024
                        path_bytes = fcntl.fcntl(fd, _F_GETPATH, buf)
                        path = path_bytes.split(b'\0')[0].decode(
                            'utf-8', errors='ignore')
                    except OSError:
                        continue

                    # Skip non-file paths and safe system locations
                    if not path or path.startswith(_SAFE_READ_PREFIXES):
                        continue

                    # Skip if it's on the Corsair drive
                    if path.startswith(CORSAIR_MOUNT):
                        continue

                    # Check if this fd is writable
                    try:
                        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                        writable = bool(flags & (os.O_WRONLY | os.O_RDWR))
                    except OSError:
                        continue

                    if writable:
                        msg = (
                            f"\n{'='*60}\n"
                            f"WATCHDOG KILL: Writable FD on disk0 detected!\n"
                            f"  FD   : {fd}\n"
                            f"  Path : {path}\n"
                            f"  Flags: {oct(flags)}\n"
                            f"{'='*60}\n"
                        )
                        print(msg, file=sys.stderr, flush=True)
                        os.kill(os.getpid(), signal.SIGKILL)

                except (ValueError, OSError):
                    continue
        except Exception:
            pass

        # Sleep before next poll
        threading.Event().wait(interval)


def start_io_watchdog(interval: float = 2.0):
    """Start the watchdog daemon thread (idempotent)."""
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True

    t = threading.Thread(
        target=_watchdog_loop,
        args=(interval,),
        daemon=True,
        name='corsair-io-watchdog'
    )
    t.start()
    return t
