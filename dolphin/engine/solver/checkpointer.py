import os
from dolphin.utils import logger

log = logger.get()

class Checkpointer(object):
    
    def __init__(self,
                interval=-1,
                by_epoch=True,
                save_optimizer=True,
                out_dir=None,
                max_keep_ckpts=-1,
                **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs

    def after_train_epoch(self, solver):
        if not self.by_epoch or (solver.epoch + 1) % self.interval != 0:
            return
        
        log.info(f'Saving checkpoint at {solver.epoch + 1} epoch')
        if not self.out_dir:
            self.out_dir = solver.work_dir
        solver.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get('filename_tmpl', 'epoch_{}.pth')
            current_epoch = solver.epoch + 1
            for epoch in range(current_epoch - self.max_keep_ckpts, 0, -1):
                ckpt_path = os.path.join(self.out_dir,
                                        filename_tmpl.format(epoch))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break