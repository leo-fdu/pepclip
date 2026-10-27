"""Byte-preserving streamed assembly without NumPy or pretrained models."""
import base64
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest

scripts=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location("stream_embedding_return",scripts/"stream_embedding_return.py")
stream=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=stream
spec.loader.exec_module(stream)


class StreamReturnTests(unittest.TestCase):
    def test_file_partition_supports_four_and_eight_disjoint_balanced_flows(self):
        files=[f"{m}/chunks/{i:06d}.f32" for m in stream.MODELS for i in range(8)]
        sizes={f:(40 if f.startswith("esm2/") else 25) for f in files}
        for connections in (1,2,4,8):
            groups=stream.partition_transfer_files(files,connections,sizes)
            flat=[f for group in groups for f in group]
            self.assertEqual(len(groups),connections)
            self.assertCountEqual(flat,files)
            self.assertEqual(len(flat),len(set(flat)))
            loads=[sum(sizes[f] for f in group) for group in groups]
            self.assertLessEqual(max(loads)-min(loads),max(sizes.values()))

    def test_parallel_transfer_flows_keep_outputs_and_exit_codes_separate(self):
        commands=[[sys.executable,"-c","print('flow-one')"],
                  [sys.executable,"-c","print('flow-two'); raise SystemExit(3)"]]
        results=stream.transfer_commands(commands,"test-placeholder")
        self.assertEqual([code for code,_ in results],[0,3])
        self.assertIn("flow-one",results[0][1])
        self.assertIn("flow-two",results[1][1])

    def test_out_of_order_rows_preserve_header_and_float_bytes(self):
        text="{'descr': '<f4', 'fortran_order': False, 'shape': (4, 2), }"
        padding=128-10-len(text)-1
        header=b"\x93NUMPY\x01\x00"+struct.pack("<H",len(text)+padding+1)+(text+" "*padding+"\n").encode()
        raw=struct.pack("<8f",1.0,-2.0,0.0,1e-8,100.0,0.3,5.0,-7.0)
        info=dict(shape=[4,2],dtype="float32",offset=len(header),file_bytes=len(header)+len(raw),
                  header_base64=base64.b64encode(header).decode())
        manifest=dict(run_fingerprint="source-a",models={m:info for m in stream.MODELS})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            stream.initialize_local(root,manifest)
            piece=root/"piece.f32"
            piece.write_bytes(raw[16:])
            stream.write_range(root/"esm2/features.npy",len(header)+16,piece)
            # A resumed receiver must keep already received rows intact.
            stream.initialize_local(root,manifest)
            piece.write_bytes(raw[:16])
            stream.write_range(root/"esm2/features.npy",len(header),piece)
            self.assertEqual((root/"esm2/features.npy").read_bytes(),header+raw)
            self.assertEqual((root/"grover/features.npy").read_bytes(),header+bytes(len(raw)))
            self.assertFalse((root/"esm2/metadata.json").exists())
            with self.assertRaisesRegex(ValueError,"different source"):
                stream.initialize_local(root,{**manifest,"run_fingerprint":"source-b"})


if __name__=="__main__":
    unittest.main()
