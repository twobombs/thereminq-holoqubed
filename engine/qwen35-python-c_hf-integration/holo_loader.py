import numpy as np
import zipfile
import io
import zstandard

class HoloQueryPlanner:
    def __init__(self, holo_file_path: str):
        self.archive = zipfile.ZipFile(holo_file_path, 'r')
        
        # Scan for all available Phase Space layers (using the Real amplitude component as the anchor)
        self.layers = [name.replace('.w_real.npy.zst', '') 
                       for name in self.archive.namelist() 
                       if name.endswith('.w_real.npy.zst')]
        
        # In the Geometry Forge, we bypassed the global dictionary to allow 
        # faster, lock-free multiprocessing. We just use standard decompression now.
        self.decompressor = zstandard.ZstdDecompressor()
        
        print(f"Successfully mounted Phase Space archive. {len(self.layers)} complex layers detected.")

    def _read_and_decompress(self, filename: str) -> np.ndarray:
        try:
            compressed_bytes = self.archive.read(filename)
            raw_bytes = self.decompressor.decompress(compressed_bytes)
            buf = io.BytesIO(raw_bytes)
            return np.load(buf, allow_pickle=False)
        except KeyError:
            # Fallback check for Bypassed Dense layers (like embeddings)
            if filename.endswith(".rows.npy.zst") or filename.endswith(".cols.npy.zst"):
                raise RuntimeError(f"Geometry missing: {filename}")
            
            fallback_dense = filename.replace('.w_real.npy.zst', '.npy.zst').replace('.w_imag.npy.zst', '.npy.zst')
            try:
                 compressed_bytes = self.archive.read(fallback_dense)
                 raw_bytes = self.decompressor.decompress(compressed_bytes)
                 buf = io.BytesIO(raw_bytes)
                 return np.load(buf, allow_pickle=False)
            except KeyError:
                 raise RuntimeError(f"File {filename} not found in archive.")
                 
        except zstandard.ZstdError as e:
            raise RuntimeError(f"Zstd decompression failed on {filename}. Data corruption? Error: {e}")
