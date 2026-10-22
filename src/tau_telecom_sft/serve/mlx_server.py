"""mlx_lm.server with its memory capped, so it can share the Mac with a training job.

Left alone it keeps 10 KV caches (about 1 GB each at 9K tokens for the 0.6B) and MLX's buffer
cache, which reached 22 GB beside the trainer's 21 GB and swapped the 32 GB Mac. This caps MLX's
buffer cache and hands every other argument to mlx_lm.server unchanged.
"""

from __future__ import annotations

import sys

CACHE_LIMIT = 2 << 30  # bytes MLX may keep for reuse after freeing


def main() -> None:
    import mlx.core as mx
    from mlx_lm.server import main as serve

    mx.set_cache_limit(CACHE_LIMIT)
    sys.argv[0] = "mlx_lm.server"
    serve()


if __name__ == "__main__":
    main()
