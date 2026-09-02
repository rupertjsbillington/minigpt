"""
Global shuffle and mixing of three datasets, writing raw Parquet, then
tokenizing into .bin shards for the nanoGPT training loop.

Features:
  - No global tokenizer state.
  - Binary length-prefixed intermediate records instead of base64.
  - Final-document truncation to hit dataset token targets.
  - Precise UTF-8 byte budgeting for Parquet row-group buffering.
  - Safe temp cleanup.
  - Raw output:      ./raw/mixed_train_XXXXXX.parquet
  - Tokenized output (for training):
        /data/data_mix/train_shards/shard_XXXXX.bin
        /data/data_mix/val.bin
        /data/data_mix/meta.pkl

Intermediate binary record format:
    [key: uint64][token_count: uint32][text_len: uint64][utf8_text_bytes]
"""

import os
import glob
import pickle
import struct
import random
import shutil
import heapq
import tempfile
import contextlib
from array import array
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tiktoken
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import scan_cache_dir
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FINE_PDFS_TARGET = 500_000_000
DCLM_TARGET = 300_000_000
FINE_WEB_EDU_TARGET = 200_000_000

# Approximate token budget per raw Parquet shard.
RAW_SHARD_TOKEN_LIMIT = 100_000_000

SEED = 42

# External-sort chunk limits.
CHUNK_SIZE = 1_000_000
MAX_CHUNK_BYTES = 512 * 1024 * 1024  # 512 MiB approximate payload per chunk

# Parquet write buffering.
RAW_BATCH_DOCS = 10_000
RAW_BATCH_BYTES = 256 * 1024 * 1024  # 256 MiB UTF-8 bytes per row-group buffer

# If True, token targets match the original tokenized-pipeline behavior:
# one artificial separator/EOT token is counted per document.
# If False, only visible text tokens are counted.
COUNT_SEPARATOR_TOKEN = True

# --- Tokenized (.bin) output configuration --------------------------------
# Final training-ready dataset location consumed by train.py.
DATA_MIX_DIR = os.path.join('data', "data_mix")
TRAIN_BIN_DIR = os.path.join(DATA_MIX_DIR, "train_shards")
VAL_BIN_PATH = os.path.join(DATA_MIX_DIR, "val.bin")
META_PATH = os.path.join(DATA_MIX_DIR, "meta.pkl")

# Token budget per output .bin shard (matches RAW_SHARD_TOKEN_LIMIT).
BIN_SHARD_TOKEN_LIMIT = 100_000_000

# Tokens held out for validation (taken from the head of the already
# globally-shuffled stream, which is statistically equivalent to the tail).
VAL_TOKENS = 1_638_400

# tiktoken's Rust core releases the GIL, so threads speed up re-tokenization
# while pool.map keeps the shuffled stream order deterministic.
TOKENIZE_THREADS = 8
# ---------------------------------------------------------------------------

SCRIPT_DIR = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)

RAW_DIR = os.path.join(SCRIPT_DIR, "raw")

# ---------------------------------------------------------------------------
# Binary intermediate record format
# ---------------------------------------------------------------------------
#   key:         uint64
#   token_count: uint32
#   text_len:    uint64
HEADER_FORMAT = ">QIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_TOKEN_COUNT = (1 << 32) - 1


# ---------------------------------------------------------------------------
# Tokenizer encapsulation
# ---------------------------------------------------------------------------
class TokenCounter:
    """
    Encapsulates tokenizer access so that no tokenizer state is stored in
    module-level globals.
    """

    def __init__(self, encoding_name: str = "gpt2", count_separator_token: bool = True):
        self.encoding_name = encoding_name
        self.count_separator_token = count_separator_token
        self.enc = tiktoken.get_encoding(encoding_name)
        self.eot = None
        if self.count_separator_token:
            self.eot = getattr(self.enc, "eot_token", None)
            if self.eot is None:
                self.eot = self.enc._special_tokens["<|endoftext|>"]

    def encode_document(self, text: str):
        """
        Encode a document for token counting/truncation.
        If count_separator_token is True, a leading separator/EOT token is
        included to match the original tokenized-pipeline accounting.
        """
        tokens = self.enc.encode_ordinary(text)
        if self.count_separator_token:
            return [self.eot] + tokens
        return tokens

    def decode_document_tokens(self, tokens):
        """
        Decode a possibly truncated token sequence back to best-effort text.
        """
        if not tokens:
            return ""
        if (
            self.count_separator_token
            and self.eot is not None
            and tokens[0] == self.eot
        ):
            tokens = tokens[1:]
        try:
            return self.enc.decode(tokens)
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def prepare_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(TRAIN_BIN_DIR, exist_ok=True)

def clean_outputs():
    """
    Remove old mixed_train_* Parquet outputs so reruns do not leave stale shards.
    """
    if not os.path.isdir(RAW_DIR):
        return
    for fn in os.listdir(RAW_DIR):
        if fn.startswith("mixed_train_") and fn.endswith(".parquet"):
            path = os.path.join(RAW_DIR, fn)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass


def clean_bin_outputs():
    """
    Remove old tokenized .bin shards / val.bin / meta.pkl so reruns do not
    leave stale training data behind.
    """
    if os.path.isdir(TRAIN_BIN_DIR):
        for fn in os.listdir(TRAIN_BIN_DIR):
            if fn.endswith(".bin"):
                try:
                    os.remove(os.path.join(TRAIN_BIN_DIR, fn))
                except OSError:
                    pass
    for path in (VAL_BIN_PATH, META_PATH):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def coerce_to_text(value):
    """
    Convert common Hugging Face `text` field variants into a string.
    Returns None if there is nothing usable.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        parts = []
        for item in value:
            s = coerce_to_text(item)
            if s:
                parts.append(s)
        return " ".join(parts) if parts else None
    return str(value)


def write_temp_record(temp_file, rng: random.Random, token_count: int, text_bytes: bytes):
    """
    Write one binary temp record.
    temp_file must be opened in binary mode, e.g. open(path, "wb").
    """
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if token_count > MAX_TOKEN_COUNT:
        raise ValueError("token_count too large for binary temp format")
    key = rng.getrandbits(64)
    header = struct.pack(
        HEADER_FORMAT,
        key,
        int(token_count),
        len(text_bytes),
    )
    temp_file.write(header)
    temp_file.write(text_bytes)


# ---------------------------------------------------------------------------
# Step 1: Stream datasets into temporary binary file
# ---------------------------------------------------------------------------
def process_dataset_to_temp(
    name: str,
    target_tokens: int,
    temp_file,
    rng: random.Random,
    tokenizer: TokenCounter,
) -> int:
    """
    Stream one dataset and write documents until target_tokens is reached.
    The final document is truncated if it crosses the target.
    """
    if target_tokens <= 0:
        return 0

    print(f"\nProcessing {name} (target {target_tokens} tokens)...")
    ds = load_dataset(
        name,
        split="train",
        streaming=True,
    )

    tokens_written = 0
    skipped_bad = 0
    skipped_token_count_too_large = 0
    pbar = tqdm(total=target_tokens, unit="tok", desc=name)

    for example in ds:
        text = coerce_to_text(example.get("text", None))
        if not text:
            continue

        # Encode text and token ids. If the text contains problematic Unicode,
        # sanitize once and retry.
        try:
            tokens = tokenizer.encode_document(text)
            raw_bytes = text.encode("utf-8")
        except Exception:
            try:
                text = text.encode("utf-8", errors="replace").decode("utf-8")
                tokens = tokenizer.encode_document(text)
                raw_bytes = text.encode("utf-8")
            except Exception:
                skipped_bad += 1
                continue

        token_count = len(tokens)
        if token_count <= 0:
            continue
        if token_count > MAX_TOKEN_COUNT:
            skipped_token_count_too_large += 1
            continue

        if tokens_written + token_count <= target_tokens:
            write_temp_record(temp_file, rng, token_count, raw_bytes)
            tokens_written += token_count
            pbar.update(token_count)
            if tokens_written >= target_tokens:
                break
        else:
            remaining = target_tokens - tokens_written
            if remaining <= 0:
                break
            # Truncate the final document to exactly fill the dataset budget.
            truncated_tokens = tokens[:remaining]
            truncated_text = tokenizer.decode_document_tokens(truncated_tokens)
            # Best-effort UTF-8 bytes for Parquet output.
            truncated_bytes = truncated_text.encode("utf-8", errors="replace")
            write_temp_record(
                temp_file,
                rng,
                remaining,
                truncated_bytes,
            )
            tokens_written += remaining
            pbar.update(remaining)
            break

    pbar.close()

    if skipped_bad:
        print(f"  Skipped {skipped_bad} unreadable documents.")
    if skipped_token_count_too_large:
        print(
            f"  Skipped {skipped_token_count_too_large} documents whose token "
            f"count exceeded {MAX_TOKEN_COUNT}."
        )
    if tokens_written < target_tokens:
        print(
            f"  Warning: only wrote {tokens_written} tokens for {name}; "
            f"target was {target_tokens}."
        )
    else:
        print(f"  Wrote {tokens_written} tokens for {name}")

    return tokens_written


# ---------------------------------------------------------------------------
# Step 2: Split binary temp file into sorted binary chunks
# ---------------------------------------------------------------------------
def split_and_sort_chunks_binary(temp_file_path: str, chunk_dir: str) -> int:
    """
    Read the binary temp file, split into chunks by record count and approximate
    byte size, sort each chunk by key, and write binary chunk files.
    """
    print("\nSplitting and sorting binary chunks...")
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_count = 0
    records = []
    approx_bytes = 0

    def flush_chunk():
        nonlocal chunk_count, records, approx_bytes
        if not records:
            return
        records.sort(key=lambda r: r[0])
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_count:05d}.bin")
        with open(chunk_path, "wb") as cf:
            for key, token_count, text_bytes in records:
                cf.write(
                    struct.pack(
                        HEADER_FORMAT,
                        key,
                        token_count,
                        len(text_bytes),
                    )
                )
                cf.write(text_bytes)
        chunk_count += 1
        records = []
        approx_bytes = 0

    with open(temp_file_path, "rb") as f:
        while True:
            header = f.read(HEADER_SIZE)
            if not header:
                break
            if len(header) != HEADER_SIZE:
                raise EOFError("Incomplete record header in temporary binary file")
            key, token_count, text_len = struct.unpack(HEADER_FORMAT, header)
            text_bytes = f.read(text_len)
            if len(text_bytes) != text_len:
                raise EOFError("Incomplete record payload in temporary binary file")
            records.append((key, token_count, text_bytes))
            approx_bytes += HEADER_SIZE + text_len
            if len(records) >= CHUNK_SIZE or approx_bytes >= MAX_CHUNK_BYTES:
                flush_chunk()

    flush_chunk()
    print(f"  Created {chunk_count} sorted binary chunks.")
    return chunk_count


# ---------------------------------------------------------------------------
# Step 3: Merge sorted binary chunks and write raw Parquet shards
# ---------------------------------------------------------------------------
def iter_binary_records(file_obj):
    """
    Yield records from an already-open binary chunk file.
    """
    while True:
        header = file_obj.read(HEADER_SIZE)
        if not header:
            break
        if len(header) != HEADER_SIZE:
            raise EOFError("Incomplete record header in binary chunk file")
        key, token_count, text_len = struct.unpack(HEADER_FORMAT, header)
        text_bytes = file_obj.read(text_len)
        if len(text_bytes) != text_len:
            raise EOFError("Incomplete record payload in binary chunk file")
        yield key, token_count, text_bytes


class RawShardWriter:
    """
    Streams raw text into Parquet shards.
    Shard boundaries are determined by token_count metadata stored in the
    temporary shuffled records.
    buffer_bytes tracks UTF-8 byte length, not Python character count.
    """

    def __init__(
        self,
        out_dir: str,
        shard_token_limit: int,
        batch_docs: int,
        batch_bytes: int,
    ):
        self.out_dir = out_dir
        self.shard_token_limit = shard_token_limit
        self.batch_docs = batch_docs
        self.batch_bytes = batch_bytes
        self.schema = pa.schema([pa.field("text", pa.string())])
        self.shard_index = 0
        self.writer = None
        self.buffer = []
        self.buffer_bytes = 0
        self.shard_docs = 0
        self.shard_tokens = 0
        self.total_docs = 0
        self.total_tokens = 0
        self.closed = False

    def _shard_path(self) -> str:
        return os.path.join(
            self.out_dir,
            f"mixed_train_{self.shard_index:06d}.parquet",
        )

    def _flush_buffer(self):
        if not self.buffer:
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(self._shard_path(), self.schema)
        table = pa.table({"text": pa.array(self.buffer, type=pa.string())})
        self.writer.write_table(table)
        self.buffer = []
        self.buffer_bytes = 0

    def _finalize_shard(self):
        if self.shard_docs == 0:
            return
        self._flush_buffer()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.shard_index += 1
        self.shard_docs = 0
        self.shard_tokens = 0

    def add(self, text: str, token_count: int, text_bytes_len=None):
        """
        Add one document to the current shard.
        If text_bytes_len is omitted, it is computed as len(text.encode("utf-8")).
        Passing text_bytes_len avoids an extra encode when the UTF-8 byte length
        is already known.
        """
        if self.closed:
            raise RuntimeError("RawShardWriter is already closed")
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        if text_bytes_len is None:
            text_bytes_len = len(text.encode("utf-8"))
        if text_bytes_len < 0:
            raise ValueError("text_bytes_len must be non-negative")

        # Close current shard before adding a document that would overflow it.
        # If token_count itself is larger than the shard limit, it becomes an
        # oversized single-document shard instead of failing.
        if (
            self.shard_docs > 0
            and self.shard_tokens + token_count > self.shard_token_limit
        ):
            self._finalize_shard()

        self.buffer.append(text)
        self.buffer_bytes += text_bytes_len
        self.shard_docs += 1
        self.shard_tokens += token_count
        self.total_docs += 1
        self.total_tokens += token_count

        if (
            len(self.buffer) >= self.batch_docs
            or self.buffer_bytes >= self.batch_bytes
        ):
            self._flush_buffer()

    def close(self):
        if self.closed:
            return
        self._finalize_shard()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.closed = True


def merge_and_write_binary(
    chunk_dir: str,
    chunk_count: int,
    output_dir: str,
    expected_tokens=None,
):
    """
    Merge sorted binary chunks and write globally shuffled raw Parquet shards.
    """
    if chunk_count == 0:
        print("No chunks to merge.")
        return

    print("\nMerging sorted binary chunks and writing raw Parquet dataset...")

    writer = RawShardWriter(
        out_dir=output_dir,
        shard_token_limit=RAW_SHARD_TOKEN_LIMIT,
        batch_docs=RAW_BATCH_DOCS,
        batch_bytes=RAW_BATCH_BYTES,
    )

    with contextlib.ExitStack() as stack:
        chunk_files = []
        for i in range(chunk_count):
            chunk_path = os.path.join(chunk_dir, f"chunk_{i:05d}.bin")
            chunk_files.append(
                stack.enter_context(open(chunk_path, "rb"))
            )

        merged_iter = heapq.merge(
            *(iter_binary_records(f) for f in chunk_files),
            key=lambda item: item[0],
        )

        pbar = tqdm(total=expected_tokens, unit="tok", desc="Final raw dataset")
        try:
            for _key, token_count, text_bytes in merged_iter:
                # Decode UTF-8 for Parquet.
                #
                # For valid UTF-8, the original byte length is exact.
                # For invalid UTF-8, decode with replacement and compute the
                # exact UTF-8 length of the string that will actually be stored.
                try:
                    text = text_bytes.decode("utf-8")
                    text_bytes_len = len(text_bytes)
                except UnicodeDecodeError:
                    text = text_bytes.decode("utf-8", errors="replace")
                    text_bytes_len = len(text.encode("utf-8"))

                writer.add(text, token_count, text_bytes_len)
                pbar.update(token_count)
        finally:
            writer.close()
            pbar.close()

    print(
        f"Final raw dataset written to {output_dir} | "
        f"docs={writer.total_docs} | tokens={writer.total_tokens}"
    )


# ---------------------------------------------------------------------------
# Step 4: Tokenize raw Parquet shards into .bin shards for training
# ---------------------------------------------------------------------------
def iter_parquet_text_batches(path, batch_docs):
    """Yield lists of non-empty text strings from one raw Parquet shard."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_docs):
        texts = batch.column("text").to_pylist()
        texts = [t for t in texts if t]
        if texts:
            yield texts


def _encode_batch(texts, tokenizer):
    """Encode a batch of documents. Worker function for the thread pool."""
    out = []
    for text in texts:
        try:
            toks = tokenizer.encode_document(text)
        except Exception:
            continue
        if toks:
            out.append(toks)
    return out


def _write_bin(tokens_buf, path):
    """Write an array('H') token buffer to disk as raw uint16."""
    arr = np.frombuffer(tokens_buf, dtype=np.uint16)
    arr.tofile(path)
    return len(arr)


def tokenize_raw_shards_to_bin(tokenizer: TokenCounter, val_target: int) -> int:
    """
    Tokenize the globally shuffled raw Parquet shards into uint16 .bin shards
    readable by the shard-aware training loop in train.py.

    Layout written:
        /data/data_mix/train_shards/shard_XXXXX.bin
        /data/data_mix/val.bin
        /data/data_mix/meta.pkl

    The first ~val_target tokens of the (already globally shuffled) stream are
    held out for validation; the remainder becomes the training stream.
    Documents are encoded with TokenCounter.encode_document, i.e. with the
    same leading EOT separator that the mixing pipeline counted, so token
    accounting stays consistent end-to-end.
    """
    shard_paths = sorted(glob.glob(os.path.join(RAW_DIR, "mixed_train_*.parquet")))
    if not shard_paths:
        raise FileNotFoundError(f"No raw parquet shards found in {RAW_DIR}")

    print(f"\nTokenizing {len(shard_paths)} raw parquet shards -> {DATA_MIX_DIR}")

    train_buf = array("H")   # uint16 token buffer (2 bytes/token)
    val_buf = array("H")
    shard_idx = 0
    train_tokens = 0
    val_tokens = 0
    val_filled = val_target <= 0

    def flush_train(force=False):
        nonlocal train_buf, shard_idx, train_tokens
        while len(train_buf) >= BIN_SHARD_TOKEN_LIMIT:
            chunk = train_buf[:BIN_SHARD_TOKEN_LIMIT]
            train_buf = train_buf[BIN_SHARD_TOKEN_LIMIT:]
            out_path = os.path.join(TRAIN_BIN_DIR, f"shard_{shard_idx:05d}.bin")
            train_tokens += _write_bin(chunk, out_path)
            shard_idx += 1
        if force and len(train_buf) > 0:
            out_path = os.path.join(TRAIN_BIN_DIR, f"shard_{shard_idx:05d}.bin")
            train_tokens += _write_bin(train_buf, out_path)
            shard_idx += 1
            train_buf = array("H")

    def route(tokens):
        nonlocal val_buf, val_tokens, val_filled
        if not val_filled:
            need = val_target - val_tokens
            if len(tokens) <= need:
                val_buf.extend(tokens)
                val_tokens += len(tokens)
                if val_tokens >= val_target:
                    val_filled = True
                return
            # Document straddles the train/val boundary: split it.
            val_buf.extend(tokens[:need])
            val_tokens += need
            val_filled = True
            tokens = tokens[need:]
        train_buf.extend(tokens)
        if len(train_buf) >= BIN_SHARD_TOKEN_LIMIT:
            flush_train()

    # tiktoken releases the GIL, so a thread pool speeds this up while
    # pool.map preserves the (already globally shuffled) stream order.
    with ThreadPoolExecutor(max_workers=TOKENIZE_THREADS) as pool:
        for path in tqdm(shard_paths, desc="Tokenizing parquet shards"):
            batches = iter_parquet_text_batches(path, RAW_BATCH_DOCS)
            for token_lists in pool.map(
                lambda b: _encode_batch(b, tokenizer), batches
            ):
                for toks in token_lists:
                    route(toks)

    flush_train(force=True)

    # Write the validation split.
    val_arr = np.frombuffer(val_buf, dtype=np.uint16)
    val_arr.tofile(VAL_BIN_PATH)

    # Metadata consumed by train.py (derives vocab_size for from-scratch init).
    meta = {
        "vocab_size": tokenizer.enc.n_vocab,          # 50257
        "tokenizer": tokenizer.encoding_name,
        "train_tokens": train_tokens,
        "val_tokens": int(len(val_arr)),
        "num_train_shards": shard_idx,
        "bin_shard_tokens": BIN_SHARD_TOKEN_LIMIT,
        "count_separator_token": tokenizer.count_separator_token,
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(
        f"Tokenized dataset written to {DATA_MIX_DIR} | "
        f"train_shards={shard_idx} | train_tokens={train_tokens:,} | "
        f"val_tokens={len(val_arr):,}"
    )
    return train_tokens + len(val_arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    prepare_dirs()
    clean_outputs()
    clean_bin_outputs()

    # Tokenizer state is created here and passed explicitly.
    tokenizer = TokenCounter(
        encoding_name="gpt2",
        count_separator_token=COUNT_SEPARATOR_TOKEN,
    )
    rng = random.Random(SEED)

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="mix_temp_")
        temp_file_path = os.path.join(temp_dir, "all_records.bin")

        total_written = 0
        with open(temp_file_path, "wb") as temp_f:
            total_written += process_dataset_to_temp(
                "codelion/finepdfs-1B",
                FINE_PDFS_TARGET,
                temp_f,
                rng,
                tokenizer,
            )
            total_written += process_dataset_to_temp(
                "codelion/dclm-baseline-1B",
                DCLM_TARGET,
                temp_f,
                rng,
                tokenizer,
            )
            total_written += process_dataset_to_temp(
                "codelion/fineweb-edu-1B",
                FINE_WEB_EDU_TARGET,
                temp_f,
                rng,
                tokenizer,
            )

        if total_written == 0:
            print("No tokens collected; exiting.")
            return

        chunk_dir = os.path.join(temp_dir, "chunks")
        chunk_count = split_and_sort_chunks_binary(temp_file_path, chunk_dir)

        # Free disk space before merge.
        os.remove(temp_file_path)

        merge_and_write_binary(
            chunk_dir,
            chunk_count,
            output_dir=RAW_DIR,
            expected_tokens=total_written,
        )

        # Step 4: tokenize the globally shuffled raw parquet shards into the
        # .bin shards consumed by train.py at /data/data_mix.
        val_target = min(VAL_TOKENS, total_written // 10)
        tokenize_raw_shards_to_bin(tokenizer, val_target)

    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        cache_info = scan_cache_dir()
        revisions_to_delete = []

        for repo in cache_info.repos:
            if repo.repo_id in {"codelion/finepdfs-1B", "codelion/dclm-baseline-1B", "codelion/fineweb-edu-1B"}:
                for revision in repo.revisions:
                    revisions_to_delete.append(revision.commit_hash)

        if revisions_to_delete:
            delete_strategy = cache_info.delete_revisions(*revisions_to_delete)
            delete_strategy.execute()
        print("Temporary files cleaned up.")

if __name__ == "__main__":
    main()