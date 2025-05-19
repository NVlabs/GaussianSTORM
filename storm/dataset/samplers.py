import itertools
from typing import Any, Optional

import torch
from torch.utils.data.sampler import Sampler

import storm.utils.distributed as distributed


def _get_torch_dtype(size: int) -> Any:
    """Return the appropriate PyTorch dtype based on size."""
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """
    Generate the indices of a random permutation.

    This matches PyTorch's CPU implementation.
    See: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    """
    dtype = _get_torch_dtype(size)
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()
        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


class InfiniteSampler(Sampler):
    def __init__(
        self,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        """
        A sampler that infinitely yields indices for a dataset.

        Args:
            sample_count (int): Number of samples in the dataset.
            shuffle (bool): Whether to shuffle indices.
            seed (int): Seed for random generator.
            start (Optional[int]): Starting index for sampling.
            step (Optional[int]): Step size for sampling.
            advance (int): Number of indices to skip at the start.
        """
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_world_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        """Yield indices based on the specified configuration."""
        iterator = self._shuffled_iterator() if self._shuffle else self._iterator()
        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        """Generate indices sequentially."""
        assert not self._shuffle
        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        """Generate shuffled indices."""
        assert self._shuffle
        generator = torch.Generator().manual_seed(self._seed)
        while True:
            iterable = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(iterable, self._start, None, self._step)


class NoPaddingDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        """
        A distributed sampler without padding.
        Used for distributed evaluation, i.e., when the dataset is not divisible by the number of
        replicas but we still want to evaluate on the full dataset.

        Args:
            dataset: The dataset to sample from.
            num_replicas (int): Number of replicas in distributed setting.
            rank (int): Rank of the current process.
            shuffle (bool): Whether to shuffle indices.
        """
        self.dataset = dataset
        self.num_replicas = distributed.get_world_size() if num_replicas is None else num_replicas
        self.rank = distributed.get_global_rank() if rank is None else rank
        self.shuffle = shuffle
        self.num_samples = len(self.dataset) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        self.rank_start = self.rank * self.num_samples
        self.rank_end = (
            (self.rank + 1) * self.num_samples
            if self.rank < self.num_replicas - 1
            else len(self.dataset)
        )

    def __iter__(self):
        """Yield indices for the current rank."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(0)
            indices = torch.randperm(len(indices), generator=g).tolist()
        indices = indices[self.rank_start : self.rank_end]
        return iter(indices)

    def __len__(self):
        """Return the number of samples for the current rank."""
        return self.num_samples
