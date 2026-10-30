# Security policy

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Report privately through the repository's **Security** tab ("Report a vulnerability").
Do not open a public issue.

Include the certkkt version, your OS, what goes wrong, and how to reproduce it.
Expect an acknowledgement within a week.

## Threat model

certkkt is a local verification tool. It needs no credentials.

- **Problem files are untrusted input.** The MPS/QPS and CBF readers parse text into
  exact rationals; they never evaluate code. Exact arithmetic on adversarial input can
  be slow or memory hungry (huge exponents are rejected, elimination has an operation
  budget), so set time limits when certifying files from unknown sources.
- **Solvers run in-process.** Adapters call Clarabel, SCS, OSQP, ECOS and HiGHS from
  your environment. Install them from sources you trust.
- **Network access** happens only in `certkkt fetch`, only to the three hosts listed in
  `src/certkkt/datasets.py` (redirects elsewhere are refused), with timeouts, a download
  cap and SHA-256 verification.
- **Study workers** are local subprocesses started with `multiprocessing` (spawn).
