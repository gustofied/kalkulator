#!/usr/bin/env python3
"""Pack the official Limite 1B Violetto checkpoint for the browser runtime.

The packer intentionally depends only on Python's standard library and NumPy.
It reads safetensors directly, applies the model's inference-time fusions, and
writes deterministic, range-friendly binary shards plus a JSON manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


SOURCE_REPO = "paradigma-inc/limite-1b-violetto"
SOURCE_REVISION = "b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
SOURCE_SHA256 = "9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5"
CONFIG_SHA256 = "d080b1f7a04a44c44cb0deb1d3fcb7ce99212fce5573f5e15b74b7f84b74463d"
DEFAULT_INPUT = Path("/tmp/limite-official/model.safetensors")
DEFAULT_OUTPUT = Path("/tmp/limite-q4")
ALIGNMENT = 256
BLOCK_SIZE = 32
SHARD_MAX_BYTES = 112 * 1024 * 1024


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def product(shape: list[int]) -> int:
    return math.prod(shape) if shape else 1


class SafeTensorFile:
    """Minimal, read-only safetensors reader with zero-copy tensor views."""

    ITEM_BYTES = {"BF16": 2, "F16": 2, "F32": 4}

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("rb")
        header_size_bytes = self._handle.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        self.header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = self._handle.read(self.header_size)
        if len(header_bytes) != self.header_size:
            raise ValueError(f"{path} has a truncated safetensors header")
        self.header = json.loads(header_bytes)
        self.metadata = self.header.pop("__metadata__", {})
        self.data_start = 8 + self.header_size
        self.file_size = path.stat().st_size
        self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._validate()

    def _validate(self) -> None:
        data_bytes = self.file_size - self.data_start
        intervals: list[tuple[int, int, str]] = []
        for name, info in self.header.items():
            dtype = info.get("dtype")
            shape = info.get("shape")
            offsets = info.get("data_offsets")
            if dtype not in self.ITEM_BYTES:
                raise ValueError(f"Unsupported dtype {dtype!r} for {name}")
            if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"Malformed metadata for {name}")
            start, end = offsets
            expected = product(shape) * self.ITEM_BYTES[dtype]
            if start < 0 or end < start or end > data_bytes or end - start != expected:
                raise ValueError(f"Invalid data range for {name}: {offsets}, expected {expected} bytes")
            intervals.append((start, end, name))
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"Overlapping tensors: {previous[2]} and {current[2]}")

    def info(self, name: str) -> dict[str, Any]:
        try:
            return self.header[name]
        except KeyError as error:
            raise KeyError(f"Checkpoint is missing required tensor {name}") from error

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.info(name)["shape"])

    def float32(self, name: str, row_start: int | None = None, row_end: int | None = None) -> np.ndarray:
        info = self.info(name)
        shape = tuple(info["shape"])
        begin, _ = info["data_offsets"]
        offset = self.data_start + begin
        dtype = info["dtype"]
        if dtype == "BF16":
            raw = np.ndarray(shape, dtype="<u2", buffer=self._mmap, offset=offset)
            selected = raw if row_start is None else raw[row_start:row_end]
            bits = selected.astype(np.uint32)
            return np.left_shift(bits, 16).view(np.float32)
        if dtype == "F16":
            raw = np.ndarray(shape, dtype="<f2", buffer=self._mmap, offset=offset)
            selected = raw if row_start is None else raw[row_start:row_end]
            return selected.astype(np.float32)
        if dtype == "F32":
            raw = np.ndarray(shape, dtype="<f4", buffer=self._mmap, offset=offset)
            selected = raw if row_start is None else raw[row_start:row_end]
            return selected.astype(np.float32, copy=True)
        raise AssertionError(dtype)

    def scalar(self, name: str) -> float:
        value = self.float32(name)
        if value.shape != ():
            raise ValueError(f"Expected scalar {name}, found shape {value.shape}")
        return float(value)

    def close(self) -> None:
        self._mmap.close()
        self._handle.close()

    def __enter__(self) -> SafeTensorFile:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ShardWriter:
    """Writes aligned tensors without allowing a tensor to cross a shard."""

    def __init__(self, directory: Path, max_bytes: int) -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        self.shards: list[dict[str, Any]] = []
        self.tensors: dict[str, dict[str, Any]] = {}
        self._handle: Any = None
        self._path: Path | None = None
        self._index = -1
        self._position = 0

    def _open_next(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._index += 1
        filename = f"limite-q4-{self._index:02d}.bin"
        self._path = self.directory / filename
        self._handle = self._path.open("wb")
        self._position = 0
        self.shards.append({"file": filename, "url": filename})

    def _pad_to(self, target: int) -> None:
        if target < self._position:
            raise AssertionError("Cannot pad backwards")
        remaining = target - self._position
        zeroes = bytes(ALIGNMENT)
        while remaining:
            amount = min(remaining, len(zeroes))
            self._handle.write(zeroes[:amount])
            remaining -= amount
        self._position = target

    def _place(self, payload_bytes: int, scale_bytes: int = 0) -> tuple[int, int | None, int]:
        if self._handle is None:
            self._open_next()
        offset = align_up(self._position)
        scale_offset = align_up(offset + payload_bytes) if scale_bytes else None
        end = (scale_offset + scale_bytes) if scale_offset is not None else (offset + payload_bytes)
        if end > self.max_bytes and self._position:
            self._open_next()
            offset = 0
            scale_offset = align_up(payload_bytes) if scale_bytes else None
            end = (scale_offset + scale_bytes) if scale_offset is not None else payload_bytes
        if end > self.max_bytes:
            raise ValueError(
                f"A {end / (1024 * 1024):.2f} MiB tensor cannot fit in the "
                f"{self.max_bytes / (1024 * 1024):.2f} MiB shard limit"
            )
        self._pad_to(offset)
        return offset, scale_offset, end

    def write_small(
        self,
        name: str,
        value: np.ndarray,
        dtype: str = "f16",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if name in self.tensors:
            raise ValueError(f"Duplicate output tensor {name}")
        if dtype == "f16":
            encoded = np.asarray(value, dtype="<f2").tobytes(order="C")
        elif dtype == "f32":
            encoded = np.asarray(value, dtype="<f4").tobytes(order="C")
        else:
            raise ValueError(f"Unsupported output dtype {dtype}")
        offset, _, end = self._place(len(encoded))
        self._handle.write(encoded)
        self._position = end
        entry: dict[str, Any] = {
            "shard": self._index,
            "offset": offset,
            "offsetWords": offset // 4,
            "byteLength": len(encoded),
            "dtype": dtype,
            "shape": list(value.shape),
        }
        if extra:
            entry.update(extra)
        self.tensors[name] = entry

    def write_q4(
        self,
        name: str,
        shape: tuple[int, int],
        read_rows: Callable[[int, int], np.ndarray],
        chunk_rows: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if name in self.tensors:
            raise ValueError(f"Duplicate output tensor {name}")
        rows, columns = shape
        if columns % BLOCK_SIZE:
            raise ValueError(f"{name} has {columns} columns, not divisible by {BLOCK_SIZE}")
        blocks_per_row = columns // BLOCK_SIZE
        packed_bytes = rows * columns // 2
        scale_bytes = rows * blocks_per_row * 2
        offset, scale_offset, end = self._place(packed_bytes, scale_bytes)
        assert scale_offset is not None
        scales = np.empty((rows, blocks_per_row), dtype="<f2")
        written = 0
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            values = np.asarray(read_rows(start, stop), dtype=np.float32, order="C")
            expected_shape = (stop - start, columns)
            if values.shape != expected_shape:
                raise ValueError(f"{name} reader returned {values.shape}, expected {expected_shape}")
            blocks = values.reshape(stop - start, blocks_per_row, BLOCK_SIZE)
            max_abs = np.max(np.abs(blocks), axis=2)
            scale = max_abs / np.float32(7.0)
            safe_scale = np.where(scale == 0, np.float32(1.0), scale)
            quantized = np.rint(blocks / safe_scale[:, :, None])
            np.clip(quantized, -7, 7, out=quantized)
            signed = quantized.astype(np.int8).reshape(stop - start, columns)
            biased = (signed.astype(np.int16) + 8).astype(np.uint8)
            packed = biased[:, 0::2] | (biased[:, 1::2] << np.uint8(4))
            encoded = packed.tobytes(order="C")
            self._handle.write(encoded)
            written += len(encoded)
            scales[start:stop] = scale.astype("<f2")
        if written != packed_bytes:
            raise AssertionError(f"Wrote {written} Q4 bytes for {name}, expected {packed_bytes}")
        self._position = offset + packed_bytes
        self._pad_to(scale_offset)
        self._handle.write(scales.tobytes(order="C"))
        self._position = end
        entry: dict[str, Any] = {
            "shard": self._index,
            "offset": offset,
            "offsetWords": offset // 4,
            "byteLength": packed_bytes,
            "dtype": "q4_block32",
            "shape": [rows, columns],
            "scaleOffset": scale_offset,
            "scaleOffsetWords": scale_offset // 4,
            "scaleByteLength": scale_bytes,
            "scaleShape": [rows, blocks_per_row],
        }
        if extra:
            entry.update(extra)
        self.tensors[name] = entry

    def finish(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        for shard in self.shards:
            path = self.directory / shard["file"]
            size = path.stat().st_size
            if size > self.max_bytes:
                raise AssertionError(f"Shard {path} exceeds the configured limit")
            shard["byteLength"] = size
            shard["sha256"] = sha256_file(path)


def q4_raw(
    writer: ShardWriter,
    source: SafeTensorFile,
    input_name: str,
    output_name: str,
    chunk_rows: int,
    extra: dict[str, Any] | None = None,
) -> None:
    shape = source.shape(input_name)
    if len(shape) != 2:
        raise ValueError(f"Expected a matrix for {input_name}, found {shape}")
    writer.write_q4(
        output_name,
        shape,
        lambda start, stop: source.float32(input_name, start, stop),
        chunk_rows,
        extra,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args()

    started = time.monotonic()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    config_path = input_path.with_name("config.json").resolve()
    if not input_path.is_file():
        parser.error(f"checkpoint not found: {input_path}")
    if not config_path.is_file():
        parser.error(f"config not found: {config_path}")
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    print(f"Hashing {input_path} ...", flush=True)
    source_sha = sha256_file(input_path)
    if source_sha != SOURCE_SHA256:
        raise ValueError(f"Source SHA-256 is {source_sha}, expected {SOURCE_SHA256}")
    config_sha = sha256_file(config_path)
    if config_sha != CONFIG_SHA256:
        raise ValueError(f"Config SHA-256 is {config_sha}, expected {CONFIG_SHA256}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    layer_count = int(config["num_hidden_layers"])
    hidden_size = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    intermediate_size = int(config["intermediate_size"])
    value_layers = {int(index) for index in config["ve_layers"]}
    attention_gate_channels = int(config["attn_gate_channels"])
    value_gate_channels = int(config["ve_gate_channels"])

    if output_path.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output_path}; pass --force to replace it")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    output_path.mkdir(parents=True)

    writer = ShardWriter(output_path, SHARD_MAX_BYTES)
    consumed: set[str] = set()
    try:
        with SafeTensorFile(input_path) as source:
            def consume(*names: str) -> None:
                consumed.update(names)

            print("Packing tied embedding / LM head ...", flush=True)
            embed_name = "model.embed_tokens.weight"
            q4_raw(
                writer,
                source,
                embed_name,
                "embed_tokens",
                args.chunk_rows,
                {"roles": ["token_embedding", "lm_head"]},
            )
            consume(embed_name)

            print("Packing value embeddings ...", flush=True)
            value_embed_name = "model.value_embeds.weight"
            q4_raw(
                writer,
                source,
                value_embed_name,
                "value_embeds",
                args.chunk_rows,
                {"rowShape": [kv_heads, head_dim]},
            )
            consume(value_embed_name)

            for layer in range(layer_count):
                prefix = f"model.layers.{layer}"
                output_prefix = f"layers.{layer}"
                q_name = f"{prefix}.self_attn.q_proj.weight"
                k_name = f"{prefix}.self_attn.k_proj.weight"
                v_name = f"{prefix}.self_attn.v_proj.weight"
                o_name = f"{prefix}.self_attn.o_proj.weight"
                qkv_scale_name = f"{prefix}.self_attn.qkv_scale"
                o_scale_name = f"{prefix}.self_attn.o_scale"
                attention_gate_name = f"{prefix}.self_attn.attn_gate"
                value_gate_name = f"{prefix}.self_attn.ve_gate"
                xsa_name = f"{prefix}.self_attn.xsa_alpha"
                gate_name = f"{prefix}.mlp.gate_proj.weight"
                up_name = f"{prefix}.mlp.up_proj.weight"
                down_name = f"{prefix}.mlp.down_proj.weight"

                qkv_scale = source.scalar(qkv_scale_name)
                q = source.float32(q_name) * np.float32(qkv_scale)
                k = source.float32(k_name) * np.float32(qkv_scale)
                v = source.float32(v_name) * np.float32(qkv_scale)
                attention_gate_raw = source.float32(attention_gate_name)
                if attention_gate_raw.shape != (heads, attention_gate_channels):
                    raise ValueError(f"Unexpected attention gate shape in layer {layer}")
                attention_gate = np.zeros((heads, hidden_size), dtype=np.float32)
                attention_gate[:, :attention_gate_channels] = attention_gate_raw
                qkv_parts: list[dict[str, int | str]] = []
                cursor = 0
                qkv_arrays = []
                for part_name, array in (("q", q), ("k", k), ("v", v), ("attention_gate", attention_gate)):
                    rows = array.shape[0]
                    qkv_parts.append({"name": part_name, "rowStart": cursor, "rowCount": rows})
                    cursor += rows
                    qkv_arrays.append(array)
                if layer in value_layers:
                    value_gate_raw = source.float32(value_gate_name)
                    if value_gate_raw.shape != (kv_heads, value_gate_channels):
                        raise ValueError(f"Unexpected value gate shape in layer {layer}")
                    value_gate = np.zeros((kv_heads, hidden_size), dtype=np.float32)
                    value_gate[:, :value_gate_channels] = value_gate_raw
                    qkv_parts.append({"name": "value_gate", "rowStart": cursor, "rowCount": kv_heads})
                    cursor += kv_heads
                    qkv_arrays.append(value_gate)
                    consume(value_gate_name)
                elif value_gate_name in source.header:
                    raise ValueError(f"Unexpected value gate in layer {layer}")
                qkv = np.concatenate(qkv_arrays, axis=0)
                writer.write_q4(
                    f"{output_prefix}.qkv",
                    qkv.shape,
                    lambda start, stop, array=qkv: array[start:stop],
                    args.chunk_rows,
                    {"parts": qkv_parts},
                )
                consume(q_name, k_name, v_name, qkv_scale_name, attention_gate_name)

                o = source.float32(o_name) * np.float32(source.scalar(o_scale_name))
                writer.write_q4(
                    f"{output_prefix}.o",
                    o.shape,
                    lambda start, stop, array=o: array[start:stop],
                    args.chunk_rows,
                )
                consume(o_name, o_scale_name)

                gate = source.float32(gate_name)
                up = source.float32(up_name)
                gate_up = np.concatenate((gate, up), axis=0)
                writer.write_q4(
                    f"{output_prefix}.gate_up",
                    gate_up.shape,
                    lambda start, stop, array=gate_up: array[start:stop],
                    args.chunk_rows,
                    {
                        "parts": [
                            {"name": "gate", "rowStart": 0, "rowCount": intermediate_size},
                            {"name": "up", "rowStart": intermediate_size, "rowCount": intermediate_size},
                        ]
                    },
                )
                consume(gate_name, up_name)

                q4_raw(writer, source, down_name, f"{output_prefix}.down", args.chunk_rows)
                consume(down_name)

                xsa = np.tanh(source.float32(xsa_name))
                writer.write_small(
                    f"{output_prefix}.xsa_alpha",
                    xsa,
                    "f32",
                    {"transform": "tanh"},
                )
                consume(xsa_name)

                lambda_names = [
                    f"{prefix}.resid_lambda_attn",
                    f"{prefix}.post_lambda_attn",
                    f"{prefix}.resid_lambda_mlp",
                    f"{prefix}.post_lambda_mlp",
                ]
                lambdas = np.asarray([source.scalar(name) for name in lambda_names], dtype=np.float32)
                writer.write_small(
                    f"{output_prefix}.lambdas",
                    lambdas,
                    "f32",
                    {
                        "parts": [
                            "resid_lambda_attn",
                            "post_lambda_attn",
                            "resid_lambda_mlp",
                            "post_lambda_mlp",
                        ]
                    },
                )
                consume(*lambda_names)
                print(f"Packed layer {layer + 1:02d}/{layer_count}", flush=True)

            for suffix in ("dense1", "dense2", "dense2_mlp", "bias", "bias_mlp"):
                input_name = f"model.mudd.{suffix}"
                writer.write_small(f"mudd.{suffix}", source.float32(input_name), "f32")
                consume(input_name)

            missing = sorted(set(source.header) - consumed)
            unknown = sorted(consumed - set(source.header))
            if missing or unknown:
                raise ValueError(
                    f"Checkpoint contract mismatch: {len(missing)} unconsumed, "
                    f"{len(unknown)} unknown; missing={missing[:8]}, unknown={unknown[:8]}"
                )

        writer.finish()
        manifest = {
            "version": 1,
            "source": {
                "repo": SOURCE_REPO,
                "revision": SOURCE_REVISION,
                "file": "model.safetensors",
                "sha256": source_sha,
                "byteLength": input_path.stat().st_size,
            },
            "format": {
                "alignment": ALIGNMENT,
                "shardMaxBytes": SHARD_MAX_BYTES,
                "quantization": {
                    "dtype": "q4_block32",
                    "blockSize": BLOCK_SIZE,
                    "nibbleOrder": "low_first",
                    "storedValue": "signed_plus_8",
                    "signedRange": [-7, 7],
                    "scale": "max_abs_div_7",
                    "scaleDtype": "f16",
                    "matrixLayout": "output_major",
                },
            },
            "model": config,
            "shards": writer.shards,
            "tensors": writer.tensors,
        }
        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        if writer._handle is not None:
            writer._handle.close()
        shutil.rmtree(output_path, ignore_errors=True)
        raise

    elapsed = time.monotonic() - started
    packed_bytes = sum(shard["byteLength"] for shard in writer.shards)
    print(
        f"Wrote {len(writer.tensors)} tensors in {len(writer.shards)} shards: "
        f"{packed_bytes / (1024 * 1024):.2f} MiB in {elapsed:.1f}s",
        flush=True,
    )
    for shard in writer.shards:
        print(f"{shard['sha256']}  {shard['file']}  {shard['byteLength']} bytes")
    print(f"Manifest: {output_path / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
