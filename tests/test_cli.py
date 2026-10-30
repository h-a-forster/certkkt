import json

import pytest

from certkkt.cli import main

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


def test_solvers_and_version(capsys):
    assert main(["solvers"]) == 0
    assert "clarabel" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--version"])
