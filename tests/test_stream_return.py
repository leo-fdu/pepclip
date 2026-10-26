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
spec.loader.exec_module(stream)


class StreamReturnTests(unittest.TestCase):
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
