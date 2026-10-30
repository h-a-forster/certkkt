import json
import sys

import pytest

from certkkt import Q, certify, make_problem
from certkkt.cli import main
from certkkt.repair import RepairOutcome

LP = """\
NAME LP1
ROWS
 N obj
 G c1
COLUMNS
 x1 obj 1 c1 1
 x2 obj 1 c1 1
RHS
 rhs c1 1
ENDATA
"""


@pytest.fixture
def lp_file(tmp_path):
    f = tmp_path / "lp1.mps"
    f.write_text(LP)
    return f


def test_certify_canonical_and_boxed_solutions(tmp_path, lp_file, capsys):
    # Conic rows of LP1: [-x1 - x2 + s = -1, -x1 + s = 0, -x2 + s = 0], duals (1, 0, 0).
    sol = tmp_path / "sol.json"
    sol.write_text(json.dumps({"x": [0.5, 0.5], "z": [1, 0, 0]}))
    assert main(["certify", str(lp_file), str(sol), "--json", "--require-level", "optimal"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["as_is"]["level"] == "optimal"
    # Same point with boxed multipliers (OSQP signs: lower bound active => negative).
    sol.write_text(json.dumps({"x": [0.5, 0.5], "y_row": [-1], "y_col": [0, 0]}))
    assert main(["certify", str(lp_file), str(sol), "--no-repair", "--require-eps", "0"]) == 0


def test_require_eps_fails_with_exit_code_1(tmp_path, lp_file, capsys):
    sol = tmp_path / "sol.json"
    sol.write_text(json.dumps({"x": [0.5, 0.49], "z": [1, 0, 0]}))
    assert main(["certify", str(lp_file), str(sol), "--no-repair", "--require-eps", "1e-6"]) == 1
    assert "FAIL" in capsys.readouterr().err


def test_require_level_checks_every_repair_certificate(tmp_path, lp_file, monkeypatch):
    # The order is partial: the best-ranked certificate (small residuals, no exact side) can
    # miss a level that a lower-ranked one-sided certificate meets.
    p = make_problem(A=[[-1, -1], [-1, 0], [0, -1]], b=[-1, 0, 0], cones=[("nonneg", 3)], q=[1, 1])
    near = certify(p, [Q(1, 2), Q(1, 2) - Q(1, 10**9)], [1, -Q(1, 10**12), 0])
    one_sided = certify(p, [0, 0], [1, 0, 0])
    assert (near.level, one_sided.level) == ("residuals", "dual_feasible")
    stub = RepairOutcome([near, one_sided])
    monkeypatch.setattr(sys.modules["certkkt.repair"], "repair", lambda *a, **k: stub)
    sol = tmp_path / "sol.json"
    sol.write_text(json.dumps({"x": [0.5, 0.4], "z": [1, -1e-3, 0]}))
    args = ["certify", str(lp_file), str(sol), "--require-level"]
    assert main([*args, "dual_feasible"]) == 0
    assert main([*args, "primal_feasible"]) == 1


def test_bad_solution_file_is_rejected(tmp_path, lp_file):
    sol = tmp_path / "sol.json"
    sol.write_text(json.dumps({"x": [0.5, 0.5], "z": [1, 0, 0], "y_row": [1]}))
    with pytest.raises(ValueError, match="either"):
        main(["certify", str(lp_file), str(sol)])


@pytest.mark.solver
def test_solve_command(lp_file, tmp_path, capsys):
    pytest.importorskip("clarabel")
    saved = tmp_path / "out.json"
    code = main(
        [
            "solve",
            str(lp_file),
            "--solver",
            "clarabel",
            "--tol",
            "1e-9",
            "--save-solution",
            str(saved),
            "--require-level",
            "optimal",
        ]
    )
    assert code == 0
    assert set(json.loads(saved.read_text())) == {"x", "z"}
    assert "best after repair" in capsys.readouterr().out


@pytest.mark.solver
def test_solve_with_a_solver_that_cannot_take_the_problem(tmp_path, capsys):
    # OSQP has no second-order cones: an "unsupported" result has no timing and must be
    # reported, not crash the CLI.
    pytest.importorskip("osqp")
    f = tmp_path / "disc.cbf"
    f.write_text(
        "VER\n3\nOBJSENSE\nMIN\nVAR\n3 1\nQ 3\nOBJACOORD\n1\n1 -1\n"
        "CON\n1 1\nL= 1\nACOORD\n1\n0 0 1\nBCOORD\n1\n0 -1\n"
    )
    assert main(["solve", str(f), "--solver", "osqp"]) == 1
    assert "unsupported" in capsys.readouterr().err


def test_solvers_and_version(capsys):
    assert main(["solvers"]) == 0
    assert "clarabel" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--version"])
