import numpy as np

class SizeHierarchy:
    def __init__(self, volume_size,
                 chunk_size=None,
                 shard_size=None):
        self.volume_size = volume_size
        self.chunk_size = chunk_size
        self.shard_size = shard_size


    @property
    def chunk_size(self):
        return self._chunk_size


    @chunk_size.setter
    def chunk_size(self, cs):
        self._chunk_size = cs
        if cs is None:
            self.grid_shape_in_chunks = None
            self.bits_xyz = None
        else:
            self.grid_shape_in_chunks = np.ceil(np.divide(self.volume_size,
                                                          self._chunk_size))
            self.bits_xyz = np.ceil(np.log2(
                np.maximum(0, self.grid_shape_in_chunks-1)))


    @chunk_size.deleter
    def chunk_size(self):
        del self._chunk_size
        del self.grid_shape_in_chunks
        del self.bits_xyz


    @property
    def shard_size(self):
        return self._shard_size


    @shard_size.setter
    def shard_size(self, ss):
        self._shard_size = ss
        if ss is None:
            self.grid_shape_in_shards = None
        else:
            self.grid_shape_in_shards =  np.ceil(np.divide(
                self.volume_size, self.shard_size)).astype(int)


    @shard_size.deleter
    def shard_size(self):
        del self._shard_size
        del self.grid_shape_in_shards


    def to_dict(self):
        result = dict(volume_size=self.volume_size,
                      chunk_size=self.chunk_size,
                      shard_size=self.shard_size,
                      grid_shape_in_chunks=self.grid_shape_in_chunks,
                      grid_shape_in_shards=self.grid_shape_in_shards,
                      bits_xyz=self.bits_xyz)
        return result


    def compute_shard_size(self, preshift_bits, minishard_bits):
        # This estimation of shard_size requires x,y,z
        # non-zero bits all more than (preshift_bits+minishard_bits)/3
        shard_size_in_chunks = 2 ** int((preshift_bits+minishard_bits)/3)
        self.shard_size = np.multiply(shard_size_in_chunks,
                                      self.chunk_size).astype(int)
