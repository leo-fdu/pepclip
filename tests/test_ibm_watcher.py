import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import watch_ibm_checkpoints as watcher


class WatcherTests(unittest.TestCase):
    def test_venv_executable_is_not_resolved_to_base_python(self):
        c = watcher.load_config(ROOT/'configs/ibm_watch_25_30_best.json')
        self.assertEqual(c['python'], str(ROOT/'.cache/ibm-venv/bin/python'))
        cmd = watcher.commands(c, dict(name='epoch-25',epoch=25,workers=1,threads=1), 'evaluate')
        self.assertEqual(cmd[cmd.index('--seeds')+1], '5')
        self.assertEqual(cmd[cmd.index('--representation')+1], 'pepclip')

    def check_discovery(self, best_value, expected_alias):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training=root/'training';previous=root/'previous';run=root/'run'
            for p in (training,previous,run/'checkpoints'):p.mkdir(parents=True)
            c=dict(training=str(training),previous_training=str(previous),run=str(run))
            state=dict(entries={})
            def cp(epoch, value):
                return dict(epoch=epoch, model_args={'dim':1}, model={'weight':torch.tensor([float(value)])})
            # A temporary file is never treated as a ready checkpoint.
            torch.save(cp(25,25),training/'epoch_0025.pt.tmp')
            watcher.discover(c,state);self.assertEqual(state['entries'],{})
            (training/'epoch_0025.pt.tmp').rename(training/'epoch_0025.pt')
            watcher.discover(c,state);self.assertEqual(list(state['entries']),['epoch-25'])
            torch.save(cp(30,30),training/'epoch_0030.pt')
            best=cp(27,best_value);best.pop('epoch');torch.save(best,training/'best.pt')
            (previous/'metrics.jsonl').write_text(json.dumps(dict(epoch=18,validation={'loss':0.2}))+'\n')
            (training/'metrics.jsonl').write_text(json.dumps(dict(epoch=27,validation={'loss':0.1}))+'\n')
            watcher.discover(c,state)
            entry=state['entries']['best']
            self.assertEqual(entry.get('alias_of'),expected_alias)
            self.assertNotIn('epoch',torch.load(training/'best.pt',weights_only=False))
            if expected_alias is None:
                self.assertEqual(torch.load(run/'checkpoints/best.pt',weights_only=False)['epoch'],27)
            saved=json.dumps(state,sort_keys=True);watcher.discover(c,state)
            self.assertEqual(saved,json.dumps(state,sort_keys=True))

    def test_dedup_25(self):self.check_discovery(25,'epoch-25')
    def test_dedup_30(self):self.check_discovery(30,'epoch-30')
    def test_distinct_best(self):self.check_discovery(27,None)


if __name__=='__main__':unittest.main()
