import os
import sys

proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)

from ai4flow.utils.trainer import *
from ai4flow.dataset.airfoil_dataset import TurbDataset
from torch.utils.data import DataLoader
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

train_path = os.path.join(proj_dir, "data", "airfoil_stable", "train_6k/")

if "__main__" == __name__:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps_model = Unet(dim=64, channels=3, dim_mults=(1, 2, 4, 8)).to(device)

    diffuser = GaussianDiffusion(
        eps_model,
        image_size=(128, 128),
        timesteps=1000,
        auto_normalize=False
    ).to(device)

    # create pytorch test object with dfp dataset
    batch_size = 16
    prop = None
    data_set = TurbDataset(dataProp=prop, dataDir=train_path)
    train_dl = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    print("Training batches: {}".format(len(train_dl)))

    trainer = Trainer(
        diffuser,
        train_dl,
        train_lr=8e-5,
        train_num_steps=700000,  # total training steps
        ema_decay=0.995,  # exponential moving average decay
        save_and_sample_every=5000
    )

    trainer.train()
