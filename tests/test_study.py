import dataclasses
import json
import random
import zipfile

import pytest

from certkkt import certify, study
from certkkt.portfolio import FAMILY, min_variance_socp, parse_french_csv
from certkkt.rational import Q
from certkkt.report import asis_ok, markdown, summarise
from certkkt.stats import fmt_share, holm, mcnemar_exact, quantile, wilson


def french_csv(k=10, start=2001, end=2025, seed=0):
    rng = random.Random(seed)
    names = [f"I{i}" for i in range(k)]
    lines = [
        "header text",
        "",
        "  Average Value Weighted Returns -- Monthly",
        "," + ",".join(names),
    ]
    for y in range(start, end + 1):
        for mth in range(1, 13):
            vals = [f"{rng.gauss(1, 5):.2f}" for _ in range(k)]
            lines.append(f"{y}{mth:02d}," + ",".join(vals))
    lines += ["", "  Average Equal Weighted Returns -- Monthly", "," + ",".join(names), "200101,1"]
    return "\n".join(lines) + "\n"


def test_parse_french_csv_reads_first_table_exactly():
    r = parse_french_csv(french_csv())
    assert len(r.industries) == 10 and r.months[0] == 200101 and r.months[-1] == 202512
    assert all(v.denominator in (1, 2, 4, 5, 10, 20, 25, 50, 100) for v in r.values[0])
    with pytest.raises(ValueError):
        r.window(200112, 60)


def test_portfolio_socp_is_feasible_at_equal_weights():
    r = parse_french_csv(french_csv())
    p = min_variance_socp(r.window(202512, 60), "1.3", "0.25")
    k = 10
    w0 = [Q(1, k)] * k
    # t large enough: any t >= ||R w0|| works; use a generous integer bound.
    x = w0 + [Q(1, k) * Q(11, 10)] * k + [Q(10**6)]
    assert certify(p, x, None).primal_member
    assert len(FAMILY) == 54 and len({s.name for s in FAMILY}) == 54


def test_wilson_interval():
    lo, hi = wilson(14, 18)
    assert round(lo, 3) == 0.548 and round(hi, 3) == 0.910  # computed by hand
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson(3, 2)
    assert quantile([3.0, 1.0, 2.0], 0.5) == 2.0


def test_paired_tests_and_share_rounding():
    assert mcnemar_exact(7, 0) == 2 / 2**7 and mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(4, 4) == 1.0
    assert holm([0.031, 0.002, 0.5]) == [0.062, 0.006, 0.5]
    # Ties round half up whatever the float representation of 100 * k / n.
    assert fmt_share(37, 40).startswith("37/40 = 93%")
    assert fmt_share(3, 40).startswith("3/40 = 8%")


def test_report_compares_exactly_and_counts_comparable_units():
    # float("1e-8") is above the decimal 1e-8, so it does not meet eps = "1e-8".
    rec = {"key": "a", "suite": "s", "solver": "x", "instance": "i", "tol": "default"}
    over = {**rec, "as_is": {"max_rel": float("1e-8")}}
    assert not asis_ok(over, "1e-8") and asis_ok(over, "1e-6")
    full = {**over, "float": {"rel_primal": 0.0, "rel_dual": 1e-9, "rel_gap": 0.0}}
    no_float = {**over, "key": "b", "instance": "j"}
    assert "disagree at eps = 1e-8 on 1 of 1 default" in markdown([full, no_float])


@pytest.fixture
def data_root(tmp_path):
    raw = tmp_path / "data" / "raw" / "french"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "10_Industry_Portfolios_CSV.zip", "w") as zf:
        zf.writestr("10_Industry_Portfolios.csv", french_csv())
    return tmp_path / "data"


@pytest.mark.slow
def test_study_checkpoints_and_resumes(tmp_path, data_root):
    run = tmp_path / "run"
    n1 = study.run_study(
        "quick",
        ("portfolio",),
        run_dir=run,
        data_root=data_root,
        workers=1,
        max_units=3,
        echo=lambda s: None,
    )
    assert n1 == 3
    recs = study.read_records(run / "units.jsonl")
    assert [r["outcome"] for r in recs] == ["certified"] * 3
    # A torn final line (interrupted write) is dropped, and that unit is redone.
    with (run / "units.jsonl").open("a") as fh:
        fh.write('{"key": "torn')
    n2 = study.run_study(
        "quick",
        ("portfolio",),
        run_dir=run,
        data_root=data_root,
        workers=1,
        max_units=2,
        echo=lambda s: None,
    )
    assert n2 == 2
    keys = [r["key"] for r in study.read_records(run / "units.jsonl")]
    assert len(keys) == len(set(keys)) == 5
    calls = json.loads((run / "manifest.json").read_text())["calls"]
    assert [c["profile"] for c in calls] == ["quick", "quick"]
    text = markdown(study.read_records(run / "units.jsonl"))
    assert "| portfolio | clarabel |" in text
    assert summarise(study.read_records(run / "units.jsonl"))


@pytest.mark.slow
def test_study_records_timeouts_and_replaces_the_worker(tmp_path, data_root, monkeypatch):
    quick = study.PROFILES["quick"]
    tiny = dataclasses.replace(
        quick, limits={**quick.limits, "portfolio": study.Limits(60.0, 30.0, 0.01)}
    )
    monkeypatch.setitem(study.PROFILES, "quick", tiny)
    run = tmp_path / "run"
    study.run_study(
        "quick",
        ("portfolio",),
        run_dir=run,
        data_root=data_root,
        workers=1,
        max_units=2,
        echo=lambda s: None,
    )
    recs = study.read_records(run / "units.jsonl")
    assert [r["outcome"] for r in recs] == ["timeout", "timeout"]


def test_sink_records_non_finite_values_as_null(tmp_path):
    sink = study._Sink(tmp_path, 1, 0, lambda s: None)
    sink.record({"key": "k", "result": {"objective": float("nan")}, "float": [float("inf")]})
    (rec,) = study.read_records(tmp_path / "units.jsonl")
    assert rec["result"]["objective"] is None and rec["float"] == [None]


def test_study_fails_loudly_when_a_driver_thread_dies(tmp_path, data_root, monkeypatch):
    def boom(*args):
        raise ValueError("driver failed")

    monkeypatch.setattr(study, "_drive", boom)
    with pytest.raises(RuntimeError):
        study.run_study(
            "quick",
            ("portfolio",),
            run_dir=tmp_path / "run",
            data_root=data_root,
            echo=lambda s: None,
        )


def test_fetch_refuses_a_changed_unpinned_file(data_root):
    from certkkt.datasets import ChecksumError, fetch

    name = "10_Industry_Portfolios_CSV.zip"
    assert len(fetch("french", data_root, names={name}, log=print)) == 1  # records its hash
    with zipfile.ZipFile(data_root / "raw" / "french" / name, "a") as zf:
        zf.writestr("extra.txt", "upstream update")
    with pytest.raises(ChecksumError):
        fetch("french", data_root, names={name}, log=print)
    fetch("french", data_root, names={name}, log=print, accept_new_data=True)
    fetch("french", data_root, names={name}, log=print)  # the new snapshot is now recorded


def test_downloads_follow_redirects_only_to_source_hosts():
    import urllib.error
    import urllib.request

    from certkkt.datasets import SOURCES, _SourceHostsOnly

    req = urllib.request.Request(SOURCES["mm"] + "QPDATA1.ZIP")
    redirect = _SourceHostsOnly().redirect_request
    assert redirect(req, None, 302, "Found", {}, SOURCES["cblib"] + "x").full_url.endswith("x")
    for bad in ("https://example.com/QPDATA1.ZIP", "http://www.doc.ic.ac.uk/~im/QPDATA1.ZIP"):
        with pytest.raises(urllib.error.URLError):
            redirect(req, None, 302, "Found", {}, bad)


def test_manifest_records_input_file_hashes(tmp_path, data_root):
    from certkkt.datasets import sha256_file

    run = tmp_path / "run"
    study.run_study(
        "quick", ("portfolio",), run_dir=run, data_root=data_root, max_units=0, echo=print
    )
    (call,) = json.loads((run / "manifest.json").read_text())["calls"]
    rel = "raw/french/10_Industry_Portfolios_CSV.zip"
    zf = data_root / rel
    assert call["inputs"] == {rel: {"sha256": sha256_file(zf), "bytes": zf.stat().st_size}}


def test_study_refuses_to_start_without_a_solver(tmp_path, data_root, monkeypatch):
    from certkkt.solvers import ADAPTERS

    monkeypatch.setattr(ADAPTERS["ecos"], "available", lambda: False)
    with pytest.raises(RuntimeError):
        study.run_study(
            "quick", ("portfolio",), run_dir=tmp_path / "run", data_root=data_root, echo=print
        )
    assert not (tmp_path / "run").exists()


def test_plan_rejects_unknown_suite(tmp_path):
    with pytest.raises(ValueError):
        study.plan(study.PROFILES["quick"], ("nope",), tmp_path)
