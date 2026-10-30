"""certkkt: exact rational-arithmetic optimality certificates for LP, QP and SOCP solutions.

Typical use::

    from certkkt import load_instance, certify, repair
    from certkkt.solvers import get_adapter

    inst = load_instance("QAFIRO.QPS")
    res = get_adapter("clarabel").solve(inst)
    cert = certify(inst.problem, res.x, res.z)  # the floats, checked exactly
    best = repair(inst.problem, res.x, res.z).best  # strongest exact certificate found
"""

from certkkt.certify import DEFAULT_EPS, LEVELS, Certificate, certify
from certkkt.instance import Instance, load_instance
from certkkt.problem import BoxedQP, Cone, ConicProblem, make_problem
from certkkt.rational import BACKEND, Q, to_q
from certkkt.repair import METHODS, RepairOutcome, repair

__version__ = "0.1.0"

__all__ = [
    "BACKEND",
    "DEFAULT_EPS",
    "LEVELS",
    "METHODS",
    "BoxedQP",
    "Certificate",
    "Cone",
    "ConicProblem",
    "Instance",
    "Q",
    "RepairOutcome",
    "__version__",
    "certify",
    "load_instance",
    "make_problem",
    "repair",
    "to_q",
]
