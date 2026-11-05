import numpy as np
import zipfile
import io
import zstandard

class HoloQueryPlanner:
    def __init__(self, holo_file_path: str):
        self.archive = zipfile.ZipFile(holo_file_path, 'r')
        
        # Scan for all available sparse layers
        self.layers = [name.replace('.weights.npy.zst', '') 
                       for name in self.archive.namelist() 
                       if name.endswith('.weights.npy.zst')]
        
        try:
            # Mount the dictionary directly from inside the ZIP file
            dict_bytes = self.archive.read('_zstd_dictionary.dict')
            self.zstd_dict = zstandard.ZstdCompressionDict(dict_bytes)
            self.decompressor = zstandard.ZstdDecompressor(dict_data=self.zstd_dict)
            print("Successfully mounted embedded Zstd dictionary from archive.")
        except KeyError:
            raise RuntimeError(f"Corrupt .holo archive: _zstd_dictionary.dict is missing from {holo_file_path}!")

    def _read_and_decompress(self, filename: str) -> np.ndarray:
        try:
            compressed_bytes = self.archive.read(filename)
            raw_bytes = self.decompressor.decompress(compressed_bytes)
            buf = io.BytesIO(raw_bytes)
            return np.load(buf, allow_pickle=False)
        except zstandard.ZstdError as e:
            raise RuntimeError(f"Zstd decompression failed on {filename}. Dictionary mismatch or data corruption. Error: {e}")

    def _fetch_layer_data(self, layer_name: str):
        coords = self._read_and_decompress(f"{layer_name}.coords.npy.zst")
        weights = self._read_and_decompress(f"{layer_name}.weights.npy.zst")
        return coords, weights
