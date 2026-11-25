# CI checks

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests and manual
dispatch. After uploading, open the repository's Actions tab and select `CI`.
The first hosted run has not been observed during preparation.

| Job | Checks |
| --- | --- |
| Python | Bash syntax, Python compilation/import, all 99 tests and the synthetic HTTP smoke check on Python 3.10 and 3.12. |
| Manifests | Kustomize renders for baseline, batched, AWQ and Prometheus using kubectl 1.35.0. |
| Prometheus | Config validation and seven rule-test scenarios with Promtool from Prometheus 3.15.1. |

The workflow uses Ubuntu 24.04 runners. Actions are pinned to commit SHAs;
kubectl's download is checked against its official SHA-256 file. Promtool runs
with a read-only repository mount and container networking disabled. Downloads
and the image pull require runner network access. The project needs no third-party
Python packages, model weights, GPU or cluster credentials.

Jobs have ten-minute timeouts. A newer run cancels an older run for the same
workflow and ref. The token has read access to repository contents, and checkout
does not retain credentials. No deployment or external notification is performed.

## Run locally

With Python 3.10+ and Make:

```bash
make ci
```

This checks Bash syntax, runs `make check`, then `make smoke`. It opens temporary
loopback HTTP ports and saves `results/smoke.json`. Use `make unit` where socket
binding is restricted; that runs 91 offline tests and does not cover the eight
HTTP integration tests or the smoke check.

With kubectl installed, render the manifests without contacting a cluster:

```bash
make k8s-render PROFILE=baseline
make k8s-render PROFILE=batched
make k8s-render PROFILE=awq
make monitoring-render
```

With Promtool 3.15.1 installed:

```bash
make monitoring-check
```

The same Promtool commands run in the workflow's Docker container. Logs show
which assertion or validation failed; rerun the failing command after fixing it.
HTTP checks and Promtool tests remain unrun in the restricted preparation
environment. A green CI run checks protocol behaviour, rendering and rule
fixtures; it does not establish GPU performance, model quality, live metrics or
cluster readiness. Rendering does not validate Kubernetes API schemas.

## Upload the workflow

The ZIP contains a hidden `.github` folder. In Finder, press **Command+Shift+.**
to show it, then upload that folder with the other changed files. Confirm GitHub
lists `.github/workflows/ci.yml` before committing.

If GitHub refuses the hidden folder, upload the other files first. Use **Add file
→ Create new file**, name it `.github/workflows/ci.yml`, and paste the contents
of the ZIP's `ci.yml` file into the editor. Commit with `ci: add automated
project checks`. Keep the workflow at that exact path so GitHub can discover it.
