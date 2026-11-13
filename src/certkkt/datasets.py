"""Benchmark data: sources, checksums and checksummed downloads.

Network access happens only in :func:`fetch`, only to the hosts in :data:`SOURCES`, with
timeouts and SHA-256 verification against the checksums recorded here. Downloads land in
``data/raw/<suite>/`` (gitignored). Files whose checksum is not recorded (the Ken French
files are updated monthly; the two large ``beam`` instances were never downloaded while
writing this list) are accepted on first download and their checksum is written to
``data/raw/<suite>/SHA256SUMS`` so a later run can tell whether they changed.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Download sources (only these hosts are ever contacted, redirects included).
SOURCES = {
    "mm": "https://www.doc.ic.ac.uk/~im/",
    "cblib": "https://cblib.zib.de/download/all/",
    "french": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/",
}
_HOSTS = frozenset(urllib.parse.urlsplit(u).hostname for u in SOURCES.values())


class _SourceHostsOnly(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to https on one of the SOURCES hosts."""

    def redirect_request(
        self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> urllib.request.Request | None:
        parts = urllib.parse.urlsplit(newurl)
        if parts.scheme != "https" or parts.hostname not in _HOSTS:
            raise urllib.error.URLError(f"refusing redirect to {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SourceHostsOnly)

#: Maros-Meszaros QP test set (138 problems in three archives).
MM_ARCHIVES = {
    "QPDATA1.ZIP": "1a851ba04d002c1e623367dd78a4c7d71730fc58f1296e7f83afa41e412b2323",
    "QPDATA2.ZIP": "8e96a76e3fcdac1999626926f3fa629fa7476b01a51b8d6cd312cc539e79994f",
    "QPDATA3.ZIP": "bc60bb823783ba10301ad48e4e8cca4d4af2a743ea739551ecb1c0ce40e21ed5",
}

#: CBLIB subset: the continuous SOCPs of the DIMACS challenge set (nb, nql, qssp, sched).
CBLIB_DIMACS = {
    "nb": "6e137968db4bb094a331efe0186f895b49566a4b5177200b7b3fcaf60da753e7",
    "nb_L1": "43781acdc55fd81a0791e835a3141b5900229e03838885e0673f60fdf9a74310",
    "nb_L2": "73b3d4b7511e90d7d3a0fa85db10b2e77028685201523212e0d913babccaaa1f",
    "nb_L2_bessel": "fc05306e2a2774b2b58eb3cd89b2606b43a1cd9c249985d5606f1e451f0056dd",
    "nql30": "f926413ff08c1c296254f60c54cb7a4154f501ddb9d2f6948918d72d14f93739",
    "nql60": "a6c680874314566d5808ea8174b4e3c8705d8c9d450d8e8b0f21a3734e15a87b",
    "nql180": "e34bc5e68a749c20e5a95d8672d9e6f12a946c5d59ca51ff3a653b76be54acaf",
    "qssp30": "b6d3c20ee39d89bd48db11343a8c95ca11f0c8e1c5fe2c33633f59a12c7a0406",
    "qssp60": "d47c420511287c4b9acd06edc7d99ef268ff0c041113fa11dd7033561ac6c15d",
    "qssp180": "0584405b2da10ea7849e938bb345fb79e1c3e9a20dca0696f6c19451f1bfcbc6",
    "sched_50_50_orig": "852150ed6696116e8db12dd595d5ebfea6c00deb48a31c02cef12bdef149f7f3",
    "sched_50_50_scaled": "16b8c16f5a9d06ab4f0d311cfee9412b0eb241ea65e95fc7f81b838a5e25cbfb",
    "sched_100_50_orig": "da1ad978c9ff3581d8f4275911fd280a60709c16212543c4c6a4bf9027bf8f79",
    "sched_100_50_scaled": "fc650754f04308716e830480353540a718caf1963e18e0413608d3882451b74d",
    "sched_100_100_orig": "5a4062bfd8b5fad62dc3998f138c6c25e7ef22e77c68e65ef0056740519726b2",
    "sched_100_100_scaled": "fd1440908522a4407ede1bd9eccb217230b2ff35ece187bb2e50f5c9132aba83",
    "sched_200_100_orig": "e25d9fc497ef96b604b9e8341d5cd585c6a5c7a777ac94bdf500e15af0e3d9cb",
    "sched_200_100_scaled": "759b117c8918b76226d338ef9345ab3eb8aed848f19bbe1a4147981804fa6daf",
}

#: The 18 instances of Mittelmann's "Large Second Order Cone Benchmark"
#: (https://plato.asu.edu/ftp/socp.html). All 18 are on CBLIB's download list; CBLIB names
#: the fir*, dsNRL and wbNRL instances with a "2013_" prefix (CBLIB_MITTELMANN).
MITTELMANN_SOCP = (
    "beam7",
    "beam30",
    "chainsing-50000-1",
    "chainsing-50000-2",
    "chainsing-50000-3",
    "db-joint-soerensen",
    "db-plate-yield-line",
    "dsNRL",
    "firL1",
    "firL1Linfalph",
    "firL1Linfeps",
    "firL2L1alph",
    "firL2L1eps",
    "firL2Linfalph",
    "firL2Linfeps",
    "firL2a",
    "firLinf",
    "wbNRL",
)
CBLIB_MITTELMANN: dict[str, str | None] = {
    "beam7": None,  # 55 MB compressed
    "beam30": None,  # 229 MB compressed
    "chainsing-50000-1": "3f9d28be8ba13c53181f0620a2e20b1730b6ca713305990a982564ce275f0fa9",
    "chainsing-50000-2": "1412bb97d9b7edaf58999aad53a2bd995722eee70dca2a300f36e067004cdac1",
    "chainsing-50000-3": "706edee4c87a57bac2f51f2c9905a0f6b474f6b369d7a1928461cf845235b7ac",
    "db-joint-soerensen": "e226d2a743425e4a54d4846765c9c107277b84dda1e3008be667dc525a8e2446",
    "db-plate-yield-line": "12ed886514a1b38e0e6f49030330ee657a30f774eaa5658098e32f21ced44d27",
    # 74 MB to 914 MB compressed (4.2 GB together), recorded on first download
    **{
        f"2013_{n}": None for n in MITTELMANN_SOCP if n.startswith("fir") or n in ("dsNRL", "wbNRL")
    },
}

#: Ken French industry portfolios (monthly, value weighted); updated monthly upstream.
FRENCH_FILES = {k: f"{k}_Industry_Portfolios_CSV.zip" for k in (10, 30, 49)}

#: Total download cap per fetch call.
MAX_TOTAL_BYTES = 2 * 1024**3


class ChecksumError(RuntimeError):
    """A downloaded file does not match its recorded SHA-256."""


@dataclass(frozen=True)
class RemoteFile:
    suite: str
    name: str
    url: str
    sha256: str | None


def remote_files(suite: str) -> list[RemoteFile]:
    """Every file of a suite ("mm", "cblib", "mittelmann" or "french")."""
    if suite == "mm":
        return [RemoteFile("mm", n, SOURCES["mm"] + n, h) for n, h in MM_ARCHIVES.items()]
    if suite == "cblib":
        return [
            RemoteFile("cblib", f"{n}.cbf.gz", f"{SOURCES['cblib']}{n}.cbf.gz", h)
            for n, h in CBLIB_DIMACS.items()
        ]
    if suite == "mittelmann":
        return [
            RemoteFile("cblib", f"{n}.cbf.gz", f"{SOURCES['cblib']}{n}.cbf.gz", h)
            for n, h in CBLIB_MITTELMANN.items()
        ]
    if suite == "french":
        return [RemoteFile("french", f, SOURCES["french"] + f, None) for f in FRENCH_FILES.values()]
    raise ValueError(f"unknown suite {suite!r}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    suite: str,
    root: Path,
    *,
    names: set[str] | None = None,
    log: Callable[[str], None] = print,
    timeout: float = 120.0,
    accept_new_data: bool = False,
) -> list[Path]:
    """Download (or verify already present) files of a suite into ``root/raw/<suite>``.

    ``names`` restricts the download to those file names (without ``.cbf.gz``). Existing
    files are verified, not re-downloaded. Raises ChecksumError on any mismatch, before
    the file is moved into place. Files without a pinned checksum (the French zips, which
    change upstream) are checked against the hash recorded at their first download;
    ``accept_new_data`` records a changed file instead of refusing it.
    """
    files = remote_files(suite)
    if names is not None:
        stems = {f.name.split(".")[0] if f.suite == "cblib" else f.name: f for f in files}
        unknown = names - set(stems)
        if unknown:
            raise ValueError(f"not in suite {suite!r}: {sorted(unknown)}")
        files = [stems[n] for n in sorted(names)]
    total = 0
    out = []
    for rf in files:
        dest = root / "raw" / rf.suite / rf.name
        if dest.exists():
            _verify(rf, dest, log, accept_new_data)
            out.append(dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"download {rf.url}")
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
            tmp_path = Path(tmp.name)
            try:
                with _OPENER.open(rf.url, timeout=timeout) as resp:
                    for chunk in iter(lambda: resp.read(1 << 20), b""):
                        total += len(chunk)
                        if total > MAX_TOTAL_BYTES:
                            raise RuntimeError("download cap exceeded")
                        tmp.write(chunk)
            except BaseException:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise
        try:
            _verify(rf, tmp_path, log, accept_new_data)
        except ChecksumError:
            tmp_path.unlink(missing_ok=True)
            raise
        tmp_path.replace(dest)
        log(f"  {dest.stat().st_size} bytes in {time.perf_counter() - t0:.1f}s")
        out.append(dest)
    return out


def _verify(rf: RemoteFile, path: Path, log: Callable[[str], None], accept_new: bool) -> None:
    digest = sha256_file(path)
    if rf.sha256 is not None:
        if digest != rf.sha256:
            raise ChecksumError(f"{rf.name}: sha256 {digest} != recorded {rf.sha256}")
        return
    sums = path.parent / "SHA256SUMS"
    known = {}
    if sums.exists():
        for line in sums.read_text().splitlines():
            h, _, n = line.partition("  ")
            known[n] = h
    if rf.name in known and known[rf.name] != digest:
        msg = f"{rf.name} changed since first download ({known[rf.name][:12]}...)"
        if not accept_new:
            raise ChecksumError(f"{msg}; pass --accept-new-data to use the new snapshot")
        log(f"  note: {msg}")
    known[rf.name] = digest
    sums.write_text("".join(f"{h}  {n}\n" for n, h in sorted(known.items())))


def mm_members(root: Path) -> Iterator[tuple[str, Path, str, int]]:
    """``(problem name, archive, member, uncompressed size)`` for the Maros-Meszaros set."""
    for arch in MM_ARCHIVES:
        path = root / "raw" / "mm" / arch
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                stem = info.filename.rsplit("/", 1)[-1]
                if stem.upper().endswith(".QPS"):
                    yield stem[:-4].upper(), path, info.filename, info.file_size


__all__ = [
    "CBLIB_DIMACS",
    "CBLIB_MITTELMANN",
    "ChecksumError",
    "FRENCH_FILES",
    "MITTELMANN_SOCP",
    "MM_ARCHIVES",
    "RemoteFile",
    "SOURCES",
    "fetch",
    "mm_members",
    "remote_files",
    "sha256_file",
]
