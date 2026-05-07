from ai4flow.model.gaussian_diffusion import *
from ai4flow.model.unet import *
from ema_pytorch import EMA
from pathlib import Path
from torch.optim import Adam
from accelerate import Accelerator
import os
from datetime import datetime


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_dl,
            *,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=5000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True
    ):
        super().__init__()
        
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        # self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        dl = self.accelerator.prepare(train_dl)
        self.dl = cycle(dl)

        # optimizer

        optim_klass = Adam
        self.opt = optim_klass(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.results_folder = Path(results_folder)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    x, y = next(self.dl)
                    # x:condition, y:target
                    x, y = x.float().to(device), y.float().to(device)
                    # print("### main trainig loop x:", x.shape, "### x type ", type(x))
                    
                    # Use this if executed with accelerate: `accelerate launch main.py`
                    self.model.module.self_condition = x
                    self.model.module.model.self_condition = x

                    # Use this if directly executed with python: `python main.py`
                    # self.model.self_condition = x
                    # self.model.model.self_condition = x
                    # print("###type", type(self.model), "### str", self.model.__str__())
                    # print("###type for model.model", type(self.model.module.model), "### str for model.model", self.model.module.model.__str__())

                    with self.accelerator.autocast():
                        loss = self.model(y)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.6f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                pbar.update(1)

        accelerator.print('training complete')
