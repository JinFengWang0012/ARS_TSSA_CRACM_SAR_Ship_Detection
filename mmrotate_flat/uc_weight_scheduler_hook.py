import math

from mmengine.hooks import Hook

from mmrotate.registry import HOOKS


@HOOKS.register_module()
class UCWeightSchedulerHook(Hook):
    """Dynamically schedule the UCR unit-circle restriction weight."""

    priority = 'NORMAL'

    def __init__(self,
                 start_weight: float,
                 end_weight: float,
                 begin_epoch: int = 0,
                 end_epoch: int = 12,
                 schedule: str = 'cosine') -> None:
        self.start_weight = float(start_weight)
        self.end_weight = float(end_weight)
        self.begin_epoch = int(begin_epoch)
        self.end_epoch = int(end_epoch)
        self.schedule = schedule
        if self.end_epoch < self.begin_epoch:
            raise ValueError('end_epoch must be >= begin_epoch')
        if self.schedule not in ('linear', 'cosine'):
            raise ValueError("schedule must be 'linear' or 'cosine'")

    def _get_model(self, runner):
        model = runner.model
        return model.module if hasattr(model, 'module') else model

    def _resolve_weight(self, epoch: int) -> float:
        if epoch <= self.begin_epoch:
            return self.start_weight
        if epoch >= self.end_epoch:
            return self.end_weight

        progress = (epoch - self.begin_epoch) / max(
            self.end_epoch - self.begin_epoch, 1)
        if self.schedule == 'linear':
            factor = progress
        else:
            factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.start_weight + (self.end_weight - self.start_weight) * factor

    def before_train_epoch(self, runner) -> None:
        model = self._get_model(runner)
        bbox_head = getattr(model, 'bbox_head', None)
        angle_coder = getattr(bbox_head, 'angle_coder', None)
        restrict_loss = getattr(angle_coder, 'loss_angle_restrict', None)
        if restrict_loss is None or not hasattr(restrict_loss, 'loss_weight'):
            return

        current_weight = self._resolve_weight(runner.epoch)
        restrict_loss.loss_weight = current_weight
        runner.logger.info(
            f'UCWeightSchedulerHook set loss_angle_restrict.loss_weight='
            f'{current_weight:.6f} at epoch {runner.epoch + 1}.')
