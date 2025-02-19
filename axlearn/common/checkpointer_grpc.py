# Copyright © 2025 Apple Inc.
"""gRPC support for array serialization.

This module implements "parameter servers"--a cluster of gRPC servers with in-memory TensorStore
of the array shards.

Each cluster of parameter servers are typically used by a single trainer job to save checkpoints
and are usually not shared by different trainer jobs. Trainer processes save checkpoints to the
servers by writing array shards to their corresponding "local" servers (each trainer process
corresponds to one local server). The first parameter server also acts as the "primary" server
of the cluster and stores the addresses and ports of all servers and the metadata of checkpoints.

Any number of clients can read checkpoints from servers. They can discover the local servers and
metadata from the primary server and read the array shards from the local servers directly.

# Design

## Notable properties

This design doesn't depend on any filesystem support. Everything happens in-memory or in-network.

## Path handling

Almost all functions/methods in this module expect a grpc://addr:port/... path. When reading
checkpoints, the address in the path should be the primary gRPC server's address, although tensor
shards maybe stored on other gRPC servers.

## Server side

1. Each host either starts a local gRPC server using its own IP or wait for a remote "local" server
   to start.
2. Each worker waits for the first gRPC server (i.e. the primary server) to start.
3. When dumping the checkpoint, each host stores all the addressable shards to the colocated gRPC
   server (no cross-host communication).
4. During the first checkpoint, each host also writes the tensorstore gRPC spec to the primary gRPC
   server. This spec contains the shard to gRPC server mapping information.
5. When a dump is finished, an index file is written to the primary server.

This design reuses `array_serialization`, so support for advanced features such as
`max_data_shard_degree` (which may help to load balance network workload) comes for free.

This design also reuses `Checkpointer` and `TensorstoreStateStorage`, so features such as multiple
checkpoints and garbage collection comes for free.

## Client side

The client has zero dependency on jax and only depends on tensorstore and numpy. The information
that the client needs to know out-of-band includes the address of the primary server and the keys
of the tensors that it wants to read.

1. The client initialize a `TensorstoreGrpcClient` with the primary gRPC server address.
2. During the first read, the client queries the primary gRPC server for the shard-gRPC server
   mappings and cache them. This assumes that all checkpoints written at the serve-side have
   identical structure.
3. The client optionally waits for the index file to be generated.
4. The client then creates a tensorstore handle with the shard-gRPC mapping spec. This handle can
   be used to read any slices of any tensors. Tensorstore will automatically orchestrate the
   optimal connections from client to the gRPC server(s) located in each jax host that store these
   slices.

For axlearn-based client, we conveniently provide a `TensorstoreStateStorage` compatible API --
`GRPCTensorStoreStateStorage` -- that can be used as part of `TensorstoreStateStorageBuilder` to
load checkpoints.

## Tensorstore spec details

Currently, we use tensorstore's zarr driver to store sharded arrays. For an array fully sharded in
a (2, 2) mesh, the zarr format will produce the following files:

```
.zarray # metadata file
0.0
0.1
1.0
1.1
```

These file names will be appended to the path specified for the underlying kvstore. In the current
checkpointer, this path is the path in the model state tree prepended by
`base_dir/step_number/gda`, e.g. `base_dir/step_00000010/gda/decoder/emb/token_emb/weight`. So, the
final paths passed to the underlying kvstore driver when using the zarr format is
`base_dir/step_00000010/gda/decoder/emb/token_emb/weight/0.0`, ...

Tensorstore provides a kvstack driver that allows multiple drivers to be used to as a single
kvstore. This driver can match the path and route a specific driver. Therefore, we can start a gRPC
server on each host and use this routing mechanism to route the shards to the locally running gRPC
server. This avoids cross-host communication and helps to load-balance storage. Alternatively, the
"local" gRPC server can be on a remote machine to reduce the memory pressure of trainer hosts.
The gRPC server of worker 0 can be designated as the primary gRPC server and stores extra metadata
information.

As a concrete example, the following spec is used to store the shard `3.0` of an array `x1` located
on one of the host. This array is sharded four times in the first axis. Note that the metadata file
is routed to the primary gRPC server as there should be only one copy of it.

```python
{
    "driver": "zarr",
    "kvstore": {
        "driver": "kvstack",
        "path": "/prefix/step_00000010/gda/x1",
        "layers": [
            {
                # local gRPC server.
                "base": {"driver": "tsgrpc_kvstore", "address": "localhost:9836"},
                "exact": "/prefix/step_00000010/gda/x1/3.0",
                "strip_prefix": 0,
            },
            {
                # primary gRPC server.
                "base": {"driver": "tsgrpc_kvstore", "address": "localhost:9833"},
                "exact": "/prefix/step_00000010/gda/x1/.zarray",
                "strip_prefix": 0,
            },
        ],
    },
    "metadata": {"compressor": {"id": "zstd"}, "shape": (16, 128), "chunks": (4, 128)},
    "dtype": "float32",
}
```

When we want to read any tensor from these gRPC servers, all we need to do is to collect the
tensorstore spec for that tensor, and merge all the `layers`. In this way, tensorstore will have
the information about where each shard is stored. It can establish connections to the required
gRPC server depending on the slice to read. In my implementation, all hosts will write the spec
to the primary server, so it's easy for the client to retrieve them.

Although the tensorstore spec will be different when we use a different base dir or step number,
if we assume the checkpoint structure and tensor shapes will stay the same, then we can safely
reuse the tensorstore spec of the first checkpoint. Hence, we can dump the tensorstore spec
to the primary server once during the first checkpoint, and strip the base dir and step number.
The client only retrieves the spec once from the primary server. To read newer checkpoints, it
just need to prepend the base dir and step number of the newer checkpoint to the cached spec,
and pass that to tensorstore.
"""
import asyncio
import copy
import os
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Optional

import jax
import tensorstore as ts
from absl import logging

from axlearn.common.array_serialization import BoundedDataShardedAsyncCheckpointManager
from axlearn.common.checkpointer import (
    STEP_NUM_DIGITS,
    Checkpointer,
    CheckpointValidationType,
    TensorStoreStateStorage,
    build_step_dir,
    check_state_structure,
    multihost_utils,
    parse_step_from_dir,
)
from axlearn.common.config import ConfigOr, config_class, maybe_instantiate

# Indicates a gRPC path, i.e. a path that starts with "grpc://address:port".
GRPCPath = str


def _loop_retry(fn, *, interval=0.5, timeout_seconds=300):
    retry_cnt = round(timeout_seconds / interval)
    for i in range(retry_cnt):
        try:
            return fn()
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            if i == retry_cnt - 1:
                raise RuntimeError(f"Retry reached timeout={timeout_seconds}s") from e
            time.sleep(interval)


# TODO(hanzhi-zhou): implements read lock to prevent garbage collection happening while the client
# is reading checkpoints.
class TensorstoreGrpcClient:
    """Client for accessing tensors stored in a cluster of gRPC servers."""

    @classmethod
    def parse_path(cls, path: GRPCPath) -> tuple[str, str]:
        """Parses a gRPC path. Returns (address, absolute path).

        Examples: grpc://127.0.0.2:123/path/a -> ("127.0.0.2:123", "/path/a")
                  grpc://127.0.0.2:123 -> ("127.0.0.2:123", "/")
        """
        # Note: we don't use urllib to parse this to relax the character constraints in the path.
        protocol = "grpc://"
        if not path.startswith(protocol):
            raise ValueError(f"{path=} must starts with grpc://")
        addr = path[len(protocol) :].split("/", maxsplit=1)
        if len(addr) == 1:
            return addr[0], "/"
        return addr[0], "/" + addr[1]

    @classmethod
    def _get_json_store(cls, path: GRPCPath, *, read=None, create=None):
        addr, path = cls.parse_path(path)
        return ts.open(
            ts.Spec(
                {
                    "driver": "json",
                    "kvstore": {"driver": "tsgrpc_kvstore", "address": addr},
                    "path": path,
                }
            ),
            open=True,
            read=read,
            create=create,
        )

    @classmethod
    def read_json_sync(cls, path: GRPCPath) -> Any:
        """Similar to `read_json_async`, but blocks until the result is read."""
        return cls._get_json_store(path, read=True).result().read().result().item()

    @classmethod
    async def read_json_async(cls, path: GRPCPath) -> Any:
        """Reads a JSON file at `path`.

        Returns a Python object after JSON deserialization.
        """
        store = await cls._get_json_store(path, read=True)
        return (await store.read()).item()

    @classmethod
    def write_json_sync(cls, path: GRPCPath, *, data: Any):
        """Serializes `data` to a JSON file at `path`."""
        cls._get_json_store(path, create=True).result().write(data).result()

    @classmethod
    def exists(cls, path: GRPCPath) -> bool:
        """Checks whether a file at `path` exists without reading it."""
        addr, path = cls.parse_path(path)
        store = ts.KvStore.open({"driver": "tsgrpc_kvstore", "address": addr}).result()
        return len(store.list(ts.KvStore.KeyRange(path, path + "\0")).result()) > 0

    @classmethod
    def wait_for_exists(cls, path: GRPCPath, **kwargs):
        """Waits for file at `path` to exist."""

        def retry_fn():
            if cls.exists(path):
                return
            raise FileNotFoundError(f"{path} does not exist.")

        _loop_retry(retry_fn, **kwargs)

    @classmethod
    def _get_exclusive_max(cls, path: str) -> str:
        """Computes the exclusive max for ts.KvStore.KeyRange for a path prefix `path`.

        As an example, `ts.KvStore.KeyRange(path, cls._get_exclusive_max(path))` should match
        all paths starting with `path`.
        """
        return path[:-1] + chr(ord(path[-1]) + 1)

    @classmethod
    async def rm_async(cls, path: GRPCPath, *, recursive: bool = False) -> None:
        """Removes file at `path`.

        If `recursive` is True, remove all files whose path starts with `path`.
        """
        addr, path = cls.parse_path(path)
        store = await ts.KvStore.open({"driver": "tsgrpc_kvstore", "address": addr})
        if recursive:
            await store.delete_range(ts.KvStore.KeyRange(path, cls._get_exclusive_max(path)))
        else:
            await store.delete_range(ts.KvStore.KeyRange(path, path + "\0"))

    @classmethod
    def list_checkpoints(cls, ckpt_dir: GRPCPath, *, include_not_committed: bool) -> list[str]:
        """Lists checkpoints under `ckpt_dir`.

        Args:
            ckpt_dir: Base dir containing all the checkpoints.
            include_not_committed: If True, include non-committed checkpoints.

        Returns:
            A list of unsorted checkpoint steps without `ckpt_dir` prefix, e.g.
                ["step_00000200", "step_00000100"].

        Raises:
            ValueError if the gRPC server is not reachable.
        """
        primary_addr, ckpt_dir = cls.parse_path(ckpt_dir)
        store = ts.KvStore.open({"driver": "tsgrpc_kvstore", "address": primary_addr}).result()
        # This result contains all paths that start with ckpt_dir/step_*.
        result: list[bytes] = store.list(
            ts.KvStore.KeyRange(
                build_step_dir(ckpt_dir, step=0),
                cls._get_exclusive_max(build_step_dir(ckpt_dir, step=int("9" * STEP_NUM_DIGITS))),
            )
        ).result()
        out = set()
        for r in result:
            r = r.decode()[len(ckpt_dir) :].lstrip("/").split("/")
            if include_not_committed:
                out.add(r[0])
            else:
                if r[1] == "index":
                    out.add(r[0])
        return list(out)

    def __init__(self):
        self._specs = None
        self._all_addresses = None
        self._split_regex = re.compile(r"step_[0-9]{8}/?")

    def build_specs(self, ckpt_dir: GRPCPath) -> dict[str, dict[str, Any]]:
        """Build the tensorstore specs for all tensors stored in `ckpt_dir`.

        This is done by prepending `ckpt_dir` to the paths in the cached specs within this class.
        See also `self._fetch_and_cache_specs`.

        Returns:
            A dict that maps the absolute path of each tensor in `ckpt_dir` without the
            grpc://addr prefix (e.g. `/ckpt_dir/gda/x`) to its tensorstore spec. Each tensorstore
            spec can be directly passed to `ts.open` to read any slice of the tensor.
        """
        self._fetch_and_cache_specs(ckpt_dir)
        _, ckpt_dir = self.parse_path(ckpt_dir)
        out = {}
        for path, spec in self._specs.items():
            spec = copy.deepcopy(spec)
            spec["kvstore"]["path"] = os.path.join(ckpt_dir, spec["kvstore"]["path"])
            for layer in spec["kvstore"]["layers"]:
                layer["exact"] = os.path.join(ckpt_dir, layer["exact"])
            out[os.path.join(ckpt_dir, path)] = spec
        return out

    def _fetch_and_cache_specs(self, path: GRPCPath):
        """Gathers the gRPC specs from all hosts and cache them.

        This method first queries the primary server to get the number of hosts, and then queries
        the specs written by each host. The path prefix in each tensorstore spec is removed so that
        the spec can be reused for different path prefix, e.g. different checkpoint steps. However,
        this imposes the restriction that checkpoints written at different steps must have
        identical structure, and tensor shapes must be the same.

        Does nothing if the specs are already cached.

        Args:
            path: A GRPCPath that contains the primary server's address.
        """
        if self._specs is not None:
            return

        addr, _ = self.parse_path(path)

        async def fetch_specs():
            data = await self.read_json_async(f"grpc://{addr}/init_info.json")
            return await asyncio.gather(
                *(
                    self.read_json_async(f"grpc://{addr}/specs/rank_{i}.json")
                    for i in range(data["process_count"])
                )
            )

        consolidated_specs = {}
        all_addresses = set()
        for specs in asyncio.run(fetch_specs()):
            for spec in specs:
                # Strip ckpt_dir prefix.
                # /ckpt_dir/step_xxxxxxxx/a/b/c -> a/b/c.
                path = spec["kvstore"]["path"] = self._split_regex.split(spec["kvstore"]["path"])[1]
                for layer in spec["kvstore"]["layers"]:
                    layer["exact"] = self._split_regex.split(layer["exact"])[1]

                # Merge the kvstack layers from all hosts.
                if path in consolidated_specs:
                    consolidated_specs[path]["kvstore"]["layers"].extend(spec["kvstore"]["layers"])
                else:
                    consolidated_specs[path] = spec
        for spec in consolidated_specs.values():
            for layer in spec["kvstore"]["layers"]:
                all_addresses.add(layer["base"]["address"])
        self._all_addresses = list(all_addresses)
        self._specs = consolidated_specs

    def remove_checkpoint(self, ckpt_dir: GRPCPath):
        """Removes checkpoint from all gRPC servers at `ckpt_dir`.

        The index file is removed first to prevent a partially removed checkpoint being recognized
        as a full checkpoint.
        """
        self._fetch_and_cache_specs(ckpt_dir)

        async def rm_checkpoint():
            # Remove index first.
            await self.rm_async(os.path.join(ckpt_dir, "index"))

            fut = []
            _, path = self.parse_path(ckpt_dir)
            for addr in self._all_addresses:
                fut.append(self.rm_async(f"grpc://{addr}{path}", recursive=True))
            await asyncio.gather(*fut)

        asyncio.run(rm_checkpoint())


def _check_kvstore_server_binary():
    if shutil.which("kvstore_server_main") is None:
        raise ValueError("'kvstore_server_main' must be in PATH.")


def default_kvstore_server(local_server_addr: str) -> subprocess.Popen:
    """The default kvstore server.

    This function requires `kvstore_server_main` in PATH and starts a kvstore gRPC server at
    `local_server_addr` using the memory kvstore driver.
    """
    _check_kvstore_server_binary()
    return subprocess.Popen(
        args=[
            "kvstore_server_main",
            f'--spec={{"bind_addresses": ["{local_server_addr}"], "base": "memory://"}}',
        ],
    )


class GRPCAsyncCheckpointManager(BoundedDataShardedAsyncCheckpointManager):
    """Like `BoundedDataShardedAsyncCheckpointManager`, but backed by gRPC servers."""

    def __init__(self, cfg: "GRPCTensorStoreStateStorage.Config"):
        super().__init__(
            max_concurrent_gb=cfg.max_concurrent_gb,
            timeout_secs=cfg.timeout_secs,
            max_data_shard_degree=cfg.max_data_shard_degree,
            shard_threshold_bytes=cfg.shard_threshold_bytes,
        )
        self._local_server_proc = None
        self._local_server_addr = cfg.local_server_addr
        self._primary_server_addr = None
        self._server_timeout_secs = cfg.server_timeout_seconds
        # Local server addr can be None when we're only restoring and not writing.
        if self._local_server_addr is None and cfg.local_server_builder is not None:
            raise ValueError("Cannot start a local server when local_server_addr is None!")

        if cfg.local_server_builder is not None:
            self._local_server_proc = maybe_instantiate(cfg.local_server_builder)(
                cfg.local_server_addr
            )

        # Wait for local server to be up and running.
        if cfg.local_server_addr is not None:
            _loop_retry(
                lambda: TensorstoreGrpcClient.write_json_sync(
                    f"grpc://{cfg.local_server_addr}/init_info.json",
                    data={"process_count": jax.process_count()},
                ),
                timeout_seconds=cfg.server_timeout_seconds,
            )

    def _tensorstore_spec_modifier(self, spec, *, shard_infos):
        """Modified the tensorstore specs according to the shard_infos.

        This method modifies the tensorstore spec for an array such that
        1. The locally addressable shards are stored in the local gRPC server.
        2. The zarr metadata is stored in the primary gRPC server.

        The metadata is only written by the first jax process as all other processes open
        tensorstore with assume_metadata=True.
        """
        prev_kvstore = spec["kvstore"]
        primary_addr, path = TensorstoreGrpcClient.parse_path(prev_kvstore["path"])
        self._primary_server_addr = primary_addr
        layers = [
            dict(
                base=dict(driver="tsgrpc_kvstore", address=self._local_server_addr),
                exact=os.path.join(path, info.shard_coordinate()),
                strip_prefix=0,
            )
            for info in shard_infos
        ]
        if jax.process_index() == 0:
            layers.append(
                dict(
                    base=dict(driver="tsgrpc_kvstore", address=self._primary_server_addr),
                    exact=os.path.join(path, ".zarray"),
                    strip_prefix=0,
                )
            )
        spec["kvstore"] = dict(driver="kvstack", path=path, layers=layers)

    def _tensorstore_spec_log_fn(self, specs):
        TensorstoreGrpcClient.write_json_sync(
            path=f"grpc://{self._primary_server_addr}/specs/rank_{jax.process_index()}.json",
            data=specs,
        )

    def stop(self):
        """Stops the internal local gRPC server."""
        super().stop()
        if self._local_server_proc is not None:
            self._local_server_proc.terminate()
            self._local_server_proc.wait()
            self._local_server_proc = None


def write_index_file(*, ckpt_dir: GRPCPath, index: Any):
    TensorstoreGrpcClient.write_json_sync(
        os.path.join(ckpt_dir, "index"),
        data=index,
    )


class GRPCTensorStoreStateStorage(TensorStoreStateStorage):
    """Serialize tensors from/to gRPC-backed tensorstore.

    Supports all functionality of TensorStoreStateStorage except saving tf iterators and pygrain
    checkpoints.

    It's required that all calls to `save_to_dir` and `restore_from_dir` uses the same gRPC
    address.
    """

    @config_class
    class Config(TensorStoreStateStorage.Config):
        """Configures GRPCTensorStoreStateStorage.

        Attributes:
            local_server_addr: The address and port of the local gRPC server that may be started.
                Required if `save_to_dir` is needed or `local_server_builder` is set. Optional if
                only requires `restore_from_dir`.
            local_server_builder: If set, it should start a local gRPC server that binds to
                `local_server_addr` and return the subprocess associated with it. Otherwise,
                assumes a gRPC server at `local_server_addr` is reachable or will be reachable in
                `server_timeout_seconds` after this class is instantiated.
            server_timeout_seconds: Timeout in seconds when trying to reach primary or local
                server during init.
        """

        local_server_addr: Optional[str] = None
        local_server_builder: Optional[ConfigOr[Callable[[str], subprocess.Popen]]] = None
        server_timeout_seconds: int = 300

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._manager = GRPCAsyncCheckpointManager(cfg)
        self._client = TensorstoreGrpcClient()
        self._primary_server_addr = None

    def _spec_from_path(self, ckpt_path: GRPCPath):
        spec = {"driver": "zarr", "kvstore": {"path": ckpt_path}}
        return spec

    def _check_same_grpc_addr(self, ckpt_dir: GRPCPath):
        """Checks if we're using the same gRPC address for every save and restore call.

        Also, waits for the primary server to be reachable for the first call to this method.
        """
        addr, _ = TensorstoreGrpcClient.parse_path(ckpt_dir)
        if self._primary_server_addr is None:
            self._primary_server_addr = addr
            TensorstoreGrpcClient.wait_for_exists(f"grpc://{addr}/init_info.json")
        elif self._primary_server_addr != addr:
            raise ValueError(
                "GRPCTensorStoreStateStorage requires saving to or restoring from the same gRPC "
                f"address. Previous address: {self._primary_server_addr}. New address: {addr}. "
                "If this is intended, please instantiate new instances for each address."
            )

    def save_to_dir(self, *, step, state, ckpt_dir: GRPCPath, on_commit_callback=write_index_file):
        if self.config.local_server_addr is None:
            raise ValueError("local_server_addr must be specified to use `save_to_dir`.")
        self._check_same_grpc_addr(ckpt_dir)
        start_time = time.perf_counter()
        spec = self._get_spec(step, state, ckpt_dir)

        def commit():
            on_commit_callback(ckpt_dir=ckpt_dir, index=spec.index)
            logging.info(
                "Serialization of %s completed in %s seconds.",
                ckpt_dir,
                time.perf_counter() - start_time,
            )

        self._manager.serialize(
            spec.gda_values,
            spec.tensorstore_specs,
            on_commit_callback=commit,
        )

    def restore_from_dir(
        self, step, state, *, ckpt_dir: GRPCPath, validation=CheckpointValidationType.EXACT
    ):
        self._check_same_grpc_addr(ckpt_dir)
        spec = self._get_spec(step, state, ckpt_dir)
        logging.info("Restoring checkpoint from directory %s", ckpt_dir)
        index_file = self._client.read_json_sync(os.path.join(ckpt_dir, "index"))
        check_state_structure(index_file, target_structure=spec.index, validation=validation)
        consolidated_spec = self._client.build_specs(ckpt_dir)
        for i, tsspec in enumerate(spec.tensorstore_specs):
            _, path = TensorstoreGrpcClient.parse_path(tsspec["kvstore"]["path"])
            spec.tensorstore_specs[i] = consolidated_spec.pop(path)
        assert len(consolidated_spec) == 0

        return self._restore_tensorstore_state(state, ckpt_dir=ckpt_dir, spec=spec)

    def stop(self):
        """Stops the internal gRPC server."""
        super().stop()
        self._manager.stop()


# pylint: disable=protected-access
class GRPCCheckpointer(Checkpointer):
    """A checkpointer that can save to/restore from a cluster of gRPC servers.

    Supports all functionality of Checkpointer except saving tf iterators and pygrain checkpoints.

    The `dir` config should encode the address of the primary gRPC server and the path prefix
    within that server, e.g. "grpc://123.456.789.123:1234/any/path/prefix". The reachability of the
    gRPC server is not checked until the first call to `save` or `restore`.
    """

    _storage: GRPCTensorStoreStateStorage

    @config_class
    class Config(Checkpointer.Config):
        storage: GRPCTensorStoreStateStorage.Config = GRPCTensorStoreStateStorage.default_config()

    def __init__(self, cfg: Config, *, parent):
        cfg.index_writer = write_index_file
        if not cfg.dir.startswith("grpc://"):
            raise ValueError("GRPCCheckpointer.Config.dir must start with grpc://")
        super().__init__(cfg, parent=parent)

    @classmethod
    def checkpoint_steps(cls, base_dir: GRPCPath):
        return [parse_step_from_dir(d) for d in cls.checkpoint_paths(base_dir)]

    @classmethod
    def checkpoint_paths(cls, base_dir: GRPCPath) -> list[GRPCPath]:
        try:
            ckpts = TensorstoreGrpcClient.list_checkpoints(base_dir, include_not_committed=False)
        except ValueError:
            # Server not reachable.
            return []
        return [os.path.join(base_dir, step) for step in ckpts]

    @classmethod
    def _all_checkpoint_paths(cls, base_dir: GRPCPath):
        return TensorstoreGrpcClient.list_checkpoints(base_dir, include_not_committed=True)

    # HACK: override a classmethod with an instance method.
    def cleanup_checkpoint(self, ckpt_dir: GRPCPath, *, sync: bool = True):
        if jax.process_index() == 0:
            self._storage._client.remove_checkpoint(ckpt_dir)
        if sync:
            multihost_utils.sync_global_devices(f"grpc_{ckpt_dir}_cleanup")

    def _index_exists(self, ckpt_dir: GRPCPath):
        return TensorstoreGrpcClient.exists(os.path.join(ckpt_dir, "index"))
