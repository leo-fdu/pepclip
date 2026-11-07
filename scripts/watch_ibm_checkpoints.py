"""Watch atomic training snapshots and dispatch existing IBM commands, serially."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ibm_ood_queue import sha, write_json


def weight_digest(state):
    """Serialization-independent identity, including learned temperature."""
    h = hashlib.sha256(json.dumps(state['model_args'], sort_keys=True).encode())
    for key, value in sorted(state['model'].items()):
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def best_epoch(paths):
    records = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    # Training updates best only on strictly lower validation loss.
    return min(records, key=lambda row: row['validation']['loss'])['epoch']


def load_config(path):
    config = json.loads(path.read_text())
    for key in ('training', 'previous_training', 'training_launch', 'run', 'python',
                'feature_source', 'ibm_root', 'grover_source', 'grover_checkpoint'):
        config[key] = str((ROOT / config[key]).absolute() if key == 'python' else (ROOT / config[key]).resolve())
    if config['seeds'] != [0, 1, 2, 3, 4] or config['poll_seconds'] < 1:
        raise ValueError('Expected seeds 0-4 and positive poll interval')
    return config


def commands(config, entry, stage):
    root = Path(config['run']); name = entry['name']
    cp = root / 'checkpoints' / (name + '.pt')
    features = root / ('features-' + name)
    if stage == 'prepare':
        return [config['python'], str(ROOT/'scripts/reproject_ibm_features.py'),
                '--source', config['feature_source'], '--output', str(features),
                '--checkpoint', str(cp), '--epoch', str(entry['epoch']),
                '--ibm-root', config['ibm_root'], '--threads', '1']
    run = root / name
    if (run/'config.json').exists():
        return [config['python'], str(ROOT/'scripts/ibm_ood_queue.py'), '--run', str(run), '--resume']
    return [config['python'], str(ROOT/'scripts/ibm_ood_queue.py'), '--run', str(run),
            '--feature-run', str(features), '--representation', 'pepclip', '--seeds', '5',
            '--trials', '100', '--workers', str(entry['workers']), '--threads', str(entry['threads']),
            '--ibm-root', config['ibm_root'], '--grover-source', config['grover_source'],
            '--grover-checkpoint', config['grover_checkpoint'], '--projection-checkpoint', str(cp)]


def discover(config, state):
    import torch
    training = Path(config['training']); root = Path(config['run'])
    # These filenames are atomically renamed AFTER last.pt and best.pt are saved.
    for epoch in (25, 30):
        name = f'epoch-{epoch}'
        source = training / f'epoch_{epoch:04d}.pt'
        if name in state['entries'] or not source.exists():
            continue
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        if checkpoint['epoch'] != epoch:
            raise ValueError('Snapshot epoch mismatch')
        target = root/'checkpoints'/(name+'.pt')
        temporary = target.with_suffix('.tmp')
        shutil.copyfile(source, temporary); temporary.replace(target)
        state['entries'][name] = dict(name=name, epoch=epoch, digest=weight_digest(checkpoint),
            source=str(source), sha256=sha(target), stage='prepare')
    if 'epoch-30' in state['entries'] and 'best' not in state['entries']:
        source = training/'best.pt'
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        digest = weight_digest(checkpoint)
        epoch = best_epoch([Path(config['previous_training'])/'metrics.jsonl', training/'metrics.jsonl'])
        duplicate = next((name for name, value in state['entries'].items() if value['digest'] == digest), None)
        entry = dict(name='best', epoch=epoch, digest=digest, source=str(source), source_sha256=sha(source))
        if duplicate:
            entry.update(stage='alias', alias_of=duplicate)
        else:
            # Existing projector expects an epoch field. Only the isolated copy
            # receives this metadata; source weights and best.pt are unchanged.
            checkpoint['epoch'] = epoch
            target = root/'checkpoints/best.pt'; temporary = target.with_suffix('.tmp')
            torch.save(checkpoint, temporary); temporary.replace(target)
            entry.update(stage='prepare', sha256=sha(target))
        state['entries']['best'] = entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(); config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(dict(config=config, evaluations='40-60', checkpoints=[25,30,'final best'],
                             overlap_resources='1 worker x 1 thread', later_resources='4 workers x 3 threads'), indent=2))
        return
    import torch
    torch.set_num_threads(1)
    root = Path(config['run']); root.mkdir(parents=True, exist_ok=True)
    lock = (root/'watch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root/'checkpoints').mkdir(exist_ok=True)
    manifest = root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Watcher configuration changed; choose a new output directory')
    write_json(manifest, config)
    path = root/'status.json'
    state = json.loads(path.read_text()) if path.exists() else dict(entries={})
    for entry in state['entries'].values():
        if 'sha256' in entry and sha(root/'checkpoints'/(entry['name']+'.pt')) != entry['sha256']:
            raise ValueError('Saved checkpoint changed')
    state.update(state='watching', pid=os.getpid())
    child = log = active = None
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        while True:
            discover(config, state)
            if child is not None and child.poll() is not None:
                code = child.returncode; log.close(); child = None
                if code:
                    raise RuntimeError(f'{active[0]} {active[1]} exited {code}')
                entry = state['entries'][active[0]]
                entry['stage'] = 'evaluate' if active[1] == 'prepare' else 'complete'
                active = None
            if child is None:
                pending = next((v for v in state['entries'].values() if v['stage'] in ('prepare','evaluate')), None)
                if pending:
                    if 'workers' not in pending:
                        # Keep a queue's resources immutable, including on resume.
                        finished = 'epoch-30' in state['entries']
                        pending.update(workers=4 if finished else 1, threads=3 if finished else 1)
                    cmd = commands(config, pending, pending['stage'])
                    caffeine = shutil.which('agent-caffeine')
                    if caffeine:
                        cmd = [caffeine,'hold','--reason',f"IBM {pending['name']} {pending['stage']}",'--',*cmd]
                    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), HF_HUB_OFFLINE='1',
                               TRANSFORMERS_OFFLINE='1', MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
                    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
                        env[key] = str(pending['threads'] if pending['stage']=='evaluate' else 1)
                    active = (pending['name'],pending['stage'])
                    write_json(path, dict(state, active=active, updated=time.time()))
                    log = (root/(pending['name']+'-'+pending['stage']+'.log')).open('a')
                    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True)
            complete = 'best' in state['entries'] and all(v['stage'] in ('complete','alias') for v in state['entries'].values())
            state.update(state='complete' if complete else 'watching', active=active,
                         child_pid=child.pid if child else None, updated=time.time())
            write_json(path, state)
            if complete:
                break
            if 'epoch-30' not in state['entries']:
                pid = json.loads(Path(config['training_launch']).read_text())['pid']
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    # Recheck atomically published snapshots before declaring failure.
                    if not (Path(config['training'])/'epoch_0030.pt').exists():
                        raise RuntimeError('Training exited before epoch 30; watcher stopped')
            time.sleep(config['poll_seconds'])
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM); child.wait()
        if log is not None:
            log.close()
        state.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                     error=str(error), active=None, child_pid=None, updated=time.time())
        write_json(path,state)
        raise


if __name__ == '__main__':
    main()
