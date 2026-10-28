import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from pepclip.utils import file_sha256, write_json

spec = importlib.util.spec_from_file_location('split_embedding_pairs', Path(__file__).resolve().parents[1] / 'scripts/split_embedding_pairs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SplitTests(unittest.TestCase):
    def test_deterministic_coverage_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / 'input.fasta'
            alphabet = 'ACDEFGHIKLMNPQRSTVWY'
            records = [(f'id-{i}', alphabet[i // 20] + alphabet[i % 20] + 'ACDEFGHIKL') for i in range(100)]
            fasta.write_text(''.join(f'>{sid}\n{seq}\n' for sid, seq in records))
            paths = [root / 'a', root / 'b']
            def metadata():
                for path in paths:
                    path.mkdir(exist_ok=True)
                    write_json(path / 'ids.json', [sid for sid, _ in records])
                    write_json(path / 'metadata.json', dict(schema_version=2, shape=[100, 4],
                        provenance=dict(run_identity=dict(source_hashes=dict(fasta=file_sha256(fasta)))),
                        ids_sha256=file_sha256(path / 'ids.json'), features_sha256='fixture-not-read'))
            metadata()
            first = module.split_pairs(fasta, *paths, root / 'first')
            second = module.split_pairs(fasta, *paths, root / 'second')
            self.assertEqual(first, second)
            self.assertEqual(sum(first['counts'].values()), 100)
            self.assertEqual(first['counts']['test'], 0)
            with (root / 'first/pairs.jsonl').open() as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len({r['identity_id'] for r in rows}), 100)
            self.assertEqual({r['split'] for r in rows}, {'train', 'val'})
            with self.assertRaises(FileExistsError):
                module.split_pairs(fasta, *paths, root / 'first')
            records[-1] = (records[-1][0], records[0][1])
            fasta.write_text(''.join(f'>{sid}\n{seq}\n' for sid, seq in records))
            metadata()
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                module.split_pairs(fasta, *paths, root / 'duplicate')
            self.assertFalse((root / 'duplicate/metadata.json').exists())


if __name__ == '__main__':
    unittest.main()
