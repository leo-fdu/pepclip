"""Stage committed row ranges, then receive/assemble their exact bytes online.

The cloud model runner is unchanged. Final local NPY files, including headers,
must match the cloud bundle hashes. Passwords stay in receiver process memory.
"""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from receive_embeddings import authenticated, sha256, verify


MODELS = ("esm2", "grover")


def atomic_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def stage(args):
    import numpy as np
    root, output = args.root, args.staging
    output.mkdir(parents=True, exist_ok=True)
    while not (root/"run.json").exists():
        time.sleep(5)
    run = json.loads((root/"run.json").read_text())
    identity = hashlib.sha256(json.dumps(run["identity"], sort_keys=True).encode()).hexdigest()
    info, matrices = {}, {}
    if not (output/"ids.json").exists():
        shutil.copyfile(root/"ids.json", output/"ids.json.tmp")
        os.replace(output/"ids.json.tmp",output/"ids.json")
    if sha256(output/"ids.json") != sha256(root/"ids.json"):
        raise ValueError("Staging IDs do not belong to this run")
    for model in MODELS:
        (output/model/"chunks").mkdir(parents=True,exist_ok=True)
        matrix = np.load(root/model/"features.npy",mmap_mode="r")
        matrices[model] = matrix
        with (root/model/"features.npy").open("rb") as handle:
            header = handle.read(matrix.offset)
        info[model] = dict(shape=list(matrix.shape),dtype=str(matrix.dtype),offset=matrix.offset,
            file_bytes=(root/model/"features.npy").stat().st_size,header_base64=base64.b64encode(header).decode())
    manifest = dict(run_fingerprint=identity,count=run["count"],models=info,
                    ids_sha256=sha256(root/"ids.json"),total_shards=len(run["shards"]))
    if (output/"manifest.json").exists() and json.loads((output/"manifest.json").read_text()) != manifest:
        raise ValueError("Staging directory belongs to a different run")
    atomic_json(output/"manifest.json",manifest)
    while True:
        available, metadata = {}, {}
        for model in MODELS:
            records=[]
            for marker in sorted((root/"progress"/model).glob("*.json")):
                dest = output/model/"chunks"/(marker.stem+".f32")
                staged_marker = dest.with_suffix(".json")
                record=json.loads(marker.read_text())
                if not staged_marker.exists():
                    array=np.ascontiguousarray(matrices[model][record["start"]:record["start"]+record["count"]])
                    blob=array.tobytes()
                    if hashlib.sha256(blob).hexdigest()!=record["sha256"]:
                        raise ValueError(f"Committed source changed: {model}/{marker.stem}")
                    temp=dest.with_suffix(".tmp")
                    with temp.open("wb") as handle:
                        handle.write(blob)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temp,dest)
                    atomic_json(staged_marker,record)
                elif json.loads(staged_marker.read_text()) != record:
                    raise ValueError("A committed shard changed during staging")
                records.append(record)
            available[model]=records
            metadata[model]=(root/model/"metadata.json").exists()
            if metadata[model] and not (output/model/"metadata.json").exists():
                shutil.copyfile(root/model/"metadata.json",output/model/"metadata.json.tmp")
                os.replace(output/model/"metadata.json.tmp",output/model/"metadata.json")
        hardware={}
        if (root/"status.json").exists():
            hardware={k:v for k,v in json.loads((root/"status.json").read_text()).items() if k!="cpu"}
        status=dict(time=now(),manifest=manifest,available=available,metadata=metadata,hardware=hardware)
        atomic_json(output/"status.json",status)
        print(now(),"staged",{m:len(available[m]) for m in MODELS},flush=True)
        if all(metadata.values()) and all(len(available[m])==manifest["total_shards"] for m in MODELS):
            return
        time.sleep(args.poll_seconds)


def write_range(path, offset, source):
    fd=os.open(path,os.O_RDWR)
    try:
        with source.open("rb") as handle:
            while True:
                data=handle.read(4*1024*1024)
                if not data:
                    break
                while data:
                    written=os.pwrite(fd,data,offset)
                    if written<=0:
                        raise OSError("Short local embedding write")
                    offset+=written
                    data=data[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def initialize_local(root,manifest):
    state=root/"stream-manifest.json"
    if state.exists() and json.loads(state.read_text()) != manifest:
        raise ValueError("Local output belongs to a different source run")
    atomic_json(state,manifest)
    for model in MODELS:
        folder=root/model
        folder.mkdir(exist_ok=True)
        (root/"received-shards"/model).mkdir(parents=True,exist_ok=True)
        path=folder/"features.npy"
        header=base64.b64decode(manifest["models"][model]["header_base64"])
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(header)
                handle.truncate(manifest["models"][model]["file_bytes"])
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with path.open("rb") as handle:
                if handle.read(len(header))!=header or path.stat().st_size!=manifest["models"][model]["file_bytes"]:
                    raise ValueError("Existing matrix header/size differs from source")


def receive(args):
    import getpass
    import shlex
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    cache=root/"transfer-cache"
    cache.mkdir(exist_ok=True)
    if not args.remote_staging.startswith("/") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-." for c in args.remote_staging):
        raise ValueError("Expected simple absolute remote staging path")
    password=getpass.getpass("Cloud SSH password (memory only): ")
    ssh=["ssh","-o","StrictHostKeyChecking=yes","-o",f"UserKnownHostsFile={args.known_hosts}",
         "-o","ConnectTimeout=20","-o","ServerAliveInterval=30","-o","ServerAliveCountMax=4",
         "-o","PreferredAuthentications=password","-o","NumberOfPasswordPrompts=1"]
    host=f"{args.user}@{args.host}"
    verified=set()
    while len(verified)<2:
        code,output=authenticated(ssh+[host,f"test ! -f {args.remote_staging}/status.json || cat {args.remote_staging}/status.json"],password)
        start=output.find('{"time":')
        if code or start<0:
            print(now(),"waiting_for_staging",code,flush=True)
            time.sleep(args.poll_seconds)
            continue
        status=json.loads(output[start:])
        manifest=status["manifest"]
        initialize_local(root,manifest)
        selected={}
        for model in MODELS:
            selected[model]=[r for r in status["available"][model] if
                not (root/"received-shards"/model/(r["name"]+".json")).exists()][:args.chunks_per_model]
        files=[]
        if not (root/"ids.json").exists():
            files.append("ids.json")
        for model in MODELS:
            files += [f"{model}/chunks/{r['name']}.f32" for r in selected[model]]
            if status["metadata"][model] and not (cache/model/"metadata.json").exists():
                files.append(f"{model}/metadata.json")
        if files:
            needed=sum(r["count"]*manifest["models"][m]["shape"][1]*4 for m in MODELS for r in selected[m])
            if shutil.disk_usage(root).free<2*needed+5*10**9:
                raise RuntimeError("Insufficient disk space for transfer plus assembled rows")
            with tempfile.NamedTemporaryFile(mode="w",prefix="transfer-list-",dir=cache,delete=False) as handle:
                handle.write("\n".join(files)+"\n")
                listing=Path(handle.name)
            try:
                command=["rsync","-rtz","--partial","--inplace","--timeout=300","--files-from",str(listing),
                         "-e",shlex.join(ssh),f"{host}:{args.remote_staging}/",str(cache)+"/"]
                code,log=authenticated(command,password)
            finally:
                listing.unlink(missing_ok=True)
            if code:
                print(now(),"transfer_retry",code,log[-1000:],flush=True)
                time.sleep(args.poll_seconds)
                continue
        if not (root/"ids.json").exists() and (cache/"ids.json").exists():
            if sha256(cache/"ids.json")!=manifest["ids_sha256"]:
                (cache/"ids.json").unlink()
                raise ValueError("Downloaded IDs failed checksum")
            os.replace(cache/"ids.json",root/"ids.json")
        for model in MODELS:
            info=manifest["models"][model]
            for record in selected[model]:
                piece=cache/model/"chunks"/(record["name"]+".f32")
                if piece.stat().st_size!=record["count"]*info["shape"][1]*4 or sha256(piece)!=record["sha256"]:
                    piece.unlink(missing_ok=True)
                    raise ValueError("Received row shard failed checksum")
                write_range(root/model/"features.npy",info["offset"]+record["start"]*info["shape"][1]*4,piece)
                atomic_json(root/"received-shards"/model/(record["name"]+".json"),record)
                piece.unlink()
            received=list((root/"received-shards"/model).glob("*.json"))
            if model not in verified and len(received)==manifest["total_shards"] and (cache/model/"metadata.json").exists():
                shutil.copyfile(root/"ids.json",root/model/"ids.json")
                shutil.copyfile(cache/model/"metadata.json",root/model/"metadata.json")
                result=verify(root/model)
                atomic_json(root/model/"download-verified.json",result)
                verified.add(model)
                print(now(),"bundle_verified",model,result,flush=True)
        progress={m:len(list((root/"received-shards"/m).glob("*.json"))) for m in MODELS}
        atomic_json(root/"receive-status.json",dict(time=now(),received_shards=progress,
            total_shards=manifest["total_shards"],cloud_hardware=status["hardware"],verified=sorted(verified)))
        print(now(),"received",progress,"hardware",status["hardware"],flush=True)
        if not any(selected.values()) and len(verified)<2:
            time.sleep(args.poll_seconds)
    atomic_json(root/"complete.json",dict(time=now(),models=sorted(verified),sha256_verified=True))
    print("Both complete NPY bundles match the cloud SHA256 values.",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    cloud=sub.add_parser("stage")
    cloud.add_argument("--root",type=Path,required=True)
    cloud.add_argument("--staging",type=Path,required=True)
    cloud.add_argument("--poll-seconds",type=int,default=30)
    local=sub.add_parser("receive")
    local.add_argument("--host",required=True)
    local.add_argument("--user",default="ubuntu")
    local.add_argument("--known-hosts",type=Path,required=True)
    local.add_argument("--remote-staging",required=True)
    local.add_argument("--output",type=Path,required=True)
    local.add_argument("--poll-seconds",type=int,default=120)
    local.add_argument("--chunks-per-model",type=int,default=8)
    args=p.parse_args()
    if args.poll_seconds<1 or (args.mode=="receive" and args.chunks_per_model<1):
        raise ValueError("Intervals/chunk limits must be positive")
    folder=args.staging if args.mode=="stage" else args.output
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/(args.mode+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (stage if args.mode=="stage" else receive)(args)


if __name__=="__main__":
    main()
