import torch
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler

def build_optimizer(args, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    if args.optim == 'Adam':
        optimizer =torch.optim.Adam(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
             {'params': ed_params, 'lr': args.lr_ed}],
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    elif args.optim == 'AdamW':
        optimizer =torch.optim.AdamW(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
             {'params': ed_params, 'lr': args.lr_ed}],
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
             {'params': ed_params, 'lr': args.lr_ed}],
            weight_decay=args.weight_decay,
            momentum = 0.9,
            nesterov=True,
        )
    return optimizer

def build_optimizer_cls(args, model):
    if args.optim == 'Adam':
        optimizer =torch.optim.Adam(
            [{'params': model.model.parameters(), 'lr': args.lr_ve},
             {'params': model.head.parameters(), 'lr':  args.lr_ed}],
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    elif args.optim == 'AdamW':
        optimizer =torch.optim.AdamW(
            [{'params': model.model.parameters(), 'lr': args.lr_ve},
             {'params': model.head.parameters(), 'lr': args.lr_ed}],
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            [{'params': model.model.parameters(), 'lr': args.lr_ve},
             {'params': model.head.parameters(), 'lr':  args.lr_ed}],
            weight_decay=args.weight_decay,
            momentum = 0.9,
            nesterov=True,
        )
    return optimizer


def build_lr_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    print('In optimizer:')
    print('num_step:',num_steps)
    print('decay_step:',decay_steps)
    print('warm_up_step:',warmup_steps)
    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
