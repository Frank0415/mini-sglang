import torch


class ReqToTokenPool:
    def __init__(self, total_reqs: int, max_context_len: int, device: torch.device):
        assert total_reqs > 0, "total_reqs must be positive"
        assert max_context_len > 0, "max_context_len must be positive"
        self.total_reqs = total_reqs
        self.max_context_len = max_context_len
        self.device = device

        # Only use row 1 - total_reqs, leave 0 for reserved
        self.req_to_token = torch.zeros(
            (total_reqs + 1, max_context_len),
            dtype=torch.int32,
            device=device,
        )
        # Row number is managed by CPU, mapping content is stored on the specified device
        self.free_slots = list[int](range(1, total_reqs + 1))

    def available_slots(self) -> int:
        return len(self.free_slots)

    def alloc_rows(self,need_size:int) -> list[int] | None:
        if need_size > self.available_slots():
            return None
        alloced_slots = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return alloced_slots

    def write(
        self,
        indices: tuple[int, slice],
        values: torch.Tensor,
    ) -> None:
        row, positions = indices
        assert 1 <= row <= self.total_reqs, "request row out of range"
        start = 0 if positions.start is None else positions.start
        stop = self.max_context_len if positions.stop is None else positions.stop
        assert positions.step in (None, 1), "request positions must be contiguous"
        assert 0 <= start <= stop <= self.max_context_len, (
            "request positions exceed max_context_len"
        )
        assert values.ndim == 1 and values.numel() == stop - start, (
            "one slot is required for each request position"
        )
        self.req_to_token[indices[0], indices[1]] = values

    def free_rows(self,indices:list[int]) -> None:
        assert all(1 <= row <= self.total_reqs for row in indices), (
            "request row to free out of range"
        )
        assert len(set(indices)) == len(indices), "duplicate request rows in free"
        assert set(indices).isdisjoint(self.free_slots), "request row already free"
        self.free_slots.extend(indices)

    def clear(self) -> None:
        self.free_slots = list[int](range(1, self.total_reqs + 1))
