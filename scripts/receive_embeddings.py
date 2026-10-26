"""Poll a cloud run and resume checksummed downloads using password SSH.

The password is requested once and kept only in process memory. No login keys,
password files, shell command interpolation, or server account changes are used.
"""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import pty
import select
import shutil
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticated(command, password):
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(command[0], command)
    chunks, prompt_tail, prompts = [], b"", 0
    try:
        while True:
            readable, _, _ = select.select([fd], [], [], 30)
            if not readable:
                continue
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            prompt_tail = (prompt_tail+data)[-512:]
            if b"password:" in prompt_tail.lower():
                prompts += 1
                if prompts > 1:
                    raise RuntimeError("SSH authentication failed; refusing repeated login attempts")
                os.write(fd, password.encode()+b"\n")
                prompt_tail = b""
            chunks.append(data)
            # Bound captured transfer diagnostics, while keeping JSON poll output.
            if sum(map(len,chunks)) > 4*1024*1024:
                chunks = chunks[-16:]
    finally:
        os.close(fd)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), b"".join(chunks).decode(errors="replace")


def verify(bundle):
    meta = json.loads((bundle/"metadata.json").read_text())
    for key in ("ids", "features"):
        name = "features.npy" if key == "features" else "ids.json"
        if sha256(bundle/name) != meta[key+"_sha256"]:
            raise ValueError(f"Downloaded checksum differs: {bundle/name}")
    # Shape/finite validation is done by the producer, and the exact bytes are
    # checked here without requiring NumPy/PyTorch on the receiving computer.
    return dict(shape=meta["shape"], dtype=meta["dtype"], verified_unix=time.time())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host",required=True)
    p.add_argument("--user",default="ubuntu")
    p.add_argument("--remote",required=True)
    p.add_argument("--known-hosts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--poll-seconds",type=int,default=120)
    args=p.parse_args()
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-." for c in args.remote):
        raise ValueError("Remote path must be a simple absolute path")
    if not args.remote.startswith("/"):
        raise ValueError("Expected an absolute remote path")
    args.output.mkdir(parents=True,exist_ok=True)
    password=getpass.getpass("Cloud SSH password (memory only): ")
    ssh=["ssh","-o","StrictHostKeyChecking=yes","-o",f"UserKnownHostsFile={args.known_hosts}",
         "-o","ConnectTimeout=20","-o","ServerAliveInterval=30","-o","ServerAliveCountMax=4",
         "-o","PreferredAuthentications=password","-o","NumberOfPasswordPrompts=1"]
    host=f"{args.user}@{args.host}"
    verified={}
    for model in ("esm2","grover"):
        marker=args.output/model/"download-verified.json"
        if marker.exists():
            verified[model]=verify(marker.parent)
    while len(verified)<2:
        command=f"for m in esm2 grover; do test ! -f {args.remote}/$m/metadata.json || echo READY:$m; done; test ! -f {args.remote}/status.json || cat {args.remote}/status.json"
        code,output=authenticated(ssh+[host,command],password)
        summary = {"ready": [m for m in ("esm2","grover") if f"READY:{m}" in output]}
        status_start = output.find('{"time":')
        if status_start >= 0:
            status = json.loads(output[status_start:])
            summary.update({k:v for k,v in status.items() if k != "cpu"})
        elif code:
            summary["error"] = output[-1000:]
        print(time.strftime("%Y-%m-%d %H:%M:%S"),"poll",code,summary,flush=True)
        if code:
            time.sleep(args.poll_seconds)
            continue
        for model in ("esm2","grover"):
            if model in verified or f"READY:{model}" not in output:
                continue
            bundle=args.output/model
            bundle.mkdir(exist_ok=True)
            # 202 GB combined FP32 results plus ample metadata/transfer margin.
            required=(125 if model=="esm2" else 79)*10**9
            existing=(bundle/"features.npy").stat().st_size if (bundle/"features.npy").exists() else 0
            if shutil.disk_usage(args.output).free < required-existing+5*10**9:
                raise RuntimeError("Insufficient local free space for this embedding bundle")
            import shlex
            transfer=["rsync","-rtz","--partial","--inplace","--timeout=300","-e",shlex.join(ssh),
                      f"{host}:{args.remote}/{model}/",str(bundle)+"/"]
            print(time.strftime("%Y-%m-%d %H:%M:%S"),"download_start",model,flush=True)
            code,log=authenticated(transfer,password)
            if code:
                print("Download interrupted; will resume:",code,log[-2000:],flush=True)
                continue
            try:
                verified[model]=verify(bundle)
            except ValueError:
                # Recheck whole-file rsync checksums if a completed-size file was damaged.
                code,log=authenticated(transfer[:1]+["--checksum"]+transfer[1:],password)
                if code:
                    print("Checksum repair interrupted",code,log[-2000:],flush=True)
                    continue
                verified[model]=verify(bundle)
            (bundle/"download-verified.json").write_text(json.dumps(verified[model],indent=2)+"\n")
            print(time.strftime("%Y-%m-%d %H:%M:%S"),"download_verified",model,verified[model],flush=True)
        if len(verified)<2:
            time.sleep(args.poll_seconds)
    (args.output/"complete.json").write_text(json.dumps(verified,indent=2)+"\n")
    print("Both embedding bundles downloaded and SHA256 verified.",flush=True)


if __name__=="__main__":
    main()
