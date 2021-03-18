import torch
from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self,
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
                **kwargs):
        from . import demodata
        super(RandomSampler, self).__init__(
            num, pos_fraction, neg_pos_ub, add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))

    def random_choice(self, gallery, num):
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device())
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds
    
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)
    
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)