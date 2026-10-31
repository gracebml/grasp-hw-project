"""Low-memory access to the Grasp-Anything(++) zip archives without extracting them.

The archives hold 1M-4.4M tiny members. ``zipfile.ZipFile`` builds one ``ZipInfo`` per member (several GB of RAM;
it was OOM-killed on this machine), so the central directory is streamed instead and only wanted members are kept.
"""
from __future__ import annotations

import os
import struct
import zlib

_EOCD = b"PK\x05\x06"
_EOCD64_LOC = b"PK\x06\x07"
_CDH = b"PK\x01\x02"
_CDH_FMT = struct.Struct("<4sHHHHHHIIIHHHHHII")  # 46 bytes


class ZipReader:
    def __init__(self, path):
        self.path = str(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size

    def _pread(self, offset, n):
        return os.pread(self.fd, n, offset)

    def _central_directory(self):
        tail_len = min(self.size, 1 << 16)
        tail = self._pread(self.size - tail_len, tail_len)
        i = tail.rfind(_EOCD)
        if i < 0:
            raise ValueError(f"{self.path}: end of central directory not found")
        cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
        if 0xFFFFFFFF in (cd_size, cd_off):
            loc = tail[i - 20:i]
            if loc[:4] != _EOCD64_LOC:
                raise ValueError(f"{self.path}: zip64 locator not found")
            rec = self._pread(struct.unpack("<Q", loc[8:16])[0], 56)
            cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
        return cd_off, cd_size

    def entries(self, chunk=16 << 20):
        """Yield ``(name, local_header_offset, compressed_size, method)`` for every member."""
        cd_off, cd_size = self._central_directory()
        buf, i, pos, end = b"", 0, cd_off, cd_off + cd_size
        while True:
            if len(buf) - i < (1 << 16) and pos < end:
                n = min(chunk, end - pos)
                buf, i, pos = buf[i:] + self._pread(pos, n), 0, pos + n
            if len(buf) - i < _CDH_FMT.size:
                return
            (sig, _, _, _, method, _, _, _, csize, usize,
             nlen, elen, clen, _, _, _, lho) = _CDH_FMT.unpack_from(buf, i)
            if sig != _CDH:
                return
            name = buf[i + 46:i + 46 + nlen].decode("utf-8")
            if 0xFFFFFFFF in (csize, usize, lho):
                usize, csize, lho = _zip64_sizes(buf[i + 46 + nlen:i + 46 + nlen + elen], usize, csize, lho)
            yield name, lho, csize, method
            i += _CDH_FMT.size + nlen + elen + clen

    def read(self, lho, csize, method):
        nlen, elen = struct.unpack("<HH", self._pread(lho + 26, 4))
        data = self._pread(lho + 30 + nlen + elen, csize)
        if method == 0:
            return data
        if method == 8:
            return zlib.decompress(data, -15)
        raise NotImplementedError(f"compression method {method}")


def _zip64_sizes(extra, usize, csize, lho):
    j = 0
    while j + 4 <= len(extra):
        hid, hlen = struct.unpack("<HH", extra[j:j + 4])
        if hid == 1:
            vals = iter(struct.unpack(f"<{hlen // 8}Q", extra[j + 4:j + 4 + hlen - hlen % 8]))
            usize = next(vals) if usize == 0xFFFFFFFF else usize
            csize = next(vals) if csize == 0xFFFFFFFF else csize
            lho = next(vals) if lho == 0xFFFFFFFF else lho
            break
        j += 4 + hlen
    return usize, csize, lho


def stem(name: str) -> str:
    return name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
