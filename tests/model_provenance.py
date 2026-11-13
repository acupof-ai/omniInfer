#!/usr/bin/env python3
"""Repeatable model provenance collection for the deployed SGLang backends.

The script never starts or stops a backend. Every remote operation is read-only:
SGLang GET /get_server_info, `find`, and `sha256sum` inside the running
container. Weight hashes are computed on the H20B host; no weight byte crosses
the network.

Optional Hugging Face verification pins the upstream repository and main
commit, then proves byte identity: LFS OIDs are content SHA256 hashes, and the
small metadata files are downloaded once and hashed locally.

Opt-in: without --evidence-dir the script prints usage and exits 0.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

UNKNOWN = "UNKNOWN"


# ---- pure helpers (unit-tested) -------------------------------------------

def parse_inventory(text):
    """Parse `find -printf '%P\\t%s\\n'` output into {path: size_bytes}."""
    out = {}
    for line in text.splitlines():
        if not line:
            continue
        path, size = line.rsplit("\t", 1)
        out[path] = int(size)
    return out


def parse_sha256_lines(text):
    """Parse sha256sum output (`<digest>  ./path`) into {path: digest}."""
    out = {}
    for line in text.splitlines():
        if not line:
            continue
        digest, path = line.split("  ", 1)
        if path.startswith("./"):
            path = path[2:]
        out[path] = digest
    return out


def server_provenance(server_info):
    """Extract identity fields; SGLang reports None when no revision exists."""
    return {
        "served_model_name": server_info.get("served_model_name"),
        "model_path": server_info.get("model_path"),
        "server_revision": server_info.get("revision") or UNKNOWN,
        "server_model_checksum": server_info.get("model_checksum") or UNKNOWN,
        "sglang_version": server_info.get("version"),
        "status": server_info.get("status"),
    }


def verify_against_hf(remote_hashes, hf_tree, hf_main_sha, raw_fetcher):
    """Compare one deployed directory with an HF repo at a pinned main commit.

    `raw_fetcher(repo, path)` returns the upstream bytes for non-LFS files.
    LFS entries need no download: their OID is the SHA256 of the file content.
    Returns matched and mismatched file lists plus the per-file verdict.
    """
    files = []
    for entry in hf_tree:
        path = entry["path"]
        deployed = remote_hashes.get(path)
        if deployed is None:
            files.append({"path": path, "match": False, "reason": "missing"})
            continue
        if entry.get("lfs"):
            upstream = entry["lfs"]["oid"]
            kind = "lfs_oid"
        else:
            upstream = hashlib.sha256(
                raw_fetcher(entry["repo"], path)).hexdigest()
            kind = "raw_content"
        files.append({
            "path": path, "kind": kind, "deployed_sha256": deployed,
            "upstream_sha256": upstream, "match": deployed == upstream,
        })
    matched = sorted(f["path"] for f in files if f["match"])
    mismatched = sorted(f["path"] for f in files if not f["match"])
    return {
        "hf_main_commit_sha": hf_main_sha,
        "file_count": len(files),
        "all_files_byte_identical": not mismatched,
        "matched": matched,
        "mismatched": mismatched,
        "files": files,
    }


# ---- remote read-only collection -------------------------------------------

class Remote:
    def __init__(self, target, control_sock=None):
        self.target = target
        self.control_sock = control_sock

    def run(self, command):
        args = ["ssh"]
        if self.control_sock:
            args += ["-S", self.control_sock]
        args += [self.target, command]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(
                f"remote command failed ({proc.returncode}): {proc.stderr[:500]}")
        return proc.stdout

    def docker_curl(self, container, port, endpoint):
        cmd = (f"docker exec {shlex.quote(container)} curl -s "
               f"http://127.0.0.1:{port}{endpoint}")
        return json.loads(self.run(cmd))

    def docker_inventory(self, container, model_dir):
        inner = (
            f"cd {shlex.quote(model_dir)} && "
            "find . -type f -printf '%P\\t%s\\n' | sort && "
            "echo __SHA256__ && "
            "find . -type f -print0 | sort -z | xargs -0 sha256sum")
        out = self.run(
            f"docker exec {shlex.quote(container)} sh -c {shlex.quote(inner)}")
        inventory_text, sha_text = out.split("__SHA256__\n", 1)
        return parse_inventory(inventory_text), parse_sha256_lines(sha_text)


def hf_get_json(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def hf_fetch_raw(repo, path, timeout=120):
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def collect_model(remote, container, model_dir, port, hf_repo):
    server_info = remote.docker_curl(container, port, "/get_server_info")
    inventory, hashes = remote.docker_inventory(container, model_dir)
    if set(inventory) != set(hashes):
        raise RuntimeError(f"{model_dir}: inventory/hash file sets differ")
    files = [{"path": p, "bytes": inventory[p], "sha256": hashes[p]}
             for p in sorted(inventory)]
    weights = [f for f in files if f["path"].endswith(".safetensors")]
    snapshot = {
        **server_provenance(server_info),
        "model_dir": model_dir,
        "config_sha256": hashes.get("config.json", UNKNOWN),
        "file_count": len(files),
        "weight_file_count": len(weights),
        "weight_total_bytes": sum(f["bytes"] for f in weights),
        "weights": weights,
        "files": files,
    }
    if hf_repo:
        repo = hf_get_json(f"https://huggingface.co/api/models/{hf_repo}")
        tree = hf_get_json(
            f"https://huggingface.co/api/models/{hf_repo}/tree/main"
            "?recursive=true")
        for entry in tree:
            entry["repo"] = hf_repo
        snapshot["upstream"] = {
            "hf_repo": hf_repo,
            "hf_main_commit_sha": repo["sha"],
            "hf_last_modified": repo["lastModified"],
        }
        snapshot["hf_verification"] = verify_against_hf(
            hashes, tree, repo["sha"], hf_fetch_raw)
    return snapshot, server_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", default="h20b")
    parser.add_argument("--control-sock", default=None,
                        help="existing ssh ControlMaster socket (-S)")
    parser.add_argument("--container", default="linyu-dev")
    parser.add_argument("--asr-dir",
                        default="/root/omniinfer-model-validation-20261111-k3/"
                                "models/asr")
    parser.add_argument("--omni-dir",
                        default="/root/omniinfer-model-validation-20261111-k3/"
                                "models/omni")
    parser.add_argument("--asr-port", type=int, default=31000)
    parser.add_argument("--omni-port", type=int, default=31001)
    parser.add_argument("--asr-hf-repo", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--omni-hf-repo",
                        default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--no-hf", action="store_true",
                        help="skip Hugging Face cross-verification")
    parser.add_argument("--evidence-dir", required=False)
    args = parser.parse_args()
    if not args.evidence_dir:
        parser.print_help()
        print("\nThis script needs a running H20B backend and an evidence "
              "directory.")
        return 0

    root = Path(args.evidence_dir)
    root.mkdir(parents=True, exist_ok=True)
    remote = Remote(args.ssh_target, args.control_sock)
    hostname = remote.run("hostname").strip()

    models = {}
    raw_info = {}
    for served, directory, port, repo in (
            ("qwen3-asr", args.asr_dir, args.asr_port,
             None if args.no_hf else args.asr_hf_repo),
            ("qwen3-omni", args.omni_dir, args.omni_port,
             None if args.no_hf else args.omni_hf_repo)):
        snapshot, server_info = collect_model(
            remote, args.container, directory, port, repo)
        models[served] = snapshot
        raw_info[served] = server_info

    payload = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"ssh_target": args.ssh_target, "hostname": hostname,
                 "container": args.container},
        "read_only": True,
        "models": models,
    }
    (root / "provenance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    for served, info in raw_info.items():
        (root / f"server_info_{served}.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    for served, snapshot in models.items():
        verdict = snapshot.get("hf_verification", {})
        print(f"{served}: {snapshot['weight_file_count']} weight file(s), "
              f"config {snapshot['config_sha256'][:12]}…, "
              f"server_revision={snapshot['server_revision']}, "
              f"hf_identical={verdict.get('all_files_byte_identical')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
