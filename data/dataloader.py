import torch.utils.data


class InfiniteBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        while True:
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch


class FastDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, num_workers, drop_last, 
                 collate_fn=None, pin_memory=True, sampler=None):
        if sampler is None:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        self.sampler_origin = sampler  # to access DistributedSampler later
        batch_sampler = InfiniteBatchSampler(sampler, batch_size, drop_last)

        super().__init__(dataset=dataset, 
                         batch_sampler=batch_sampler, 
                         num_workers=num_workers,
                         collate_fn=collate_fn, 
                         pin_memory=pin_memory)

        self.data_iter = super().__iter__()
        self.data_idx = 0

    def __iter__(self):
        while self.data_idx < len(self.data_iter):
            self.data_idx += 1
            yield next(self.data_iter)

        self.data_idx = 0
