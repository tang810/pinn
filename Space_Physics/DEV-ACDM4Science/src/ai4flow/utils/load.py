from ai4flow.utils.trainer import *
import dataset
import numpy as np
from torch.utils.data import DataLoader

if "__main__" == __name__:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create pytorch test object with dfp dataset
    batch_size = 18
    data_set = dataset.TurbDataset(mode=2, dataDir="./test/", dataDirTest="./test/")
    val_dl = DataLoader(data_set, batch_size=batch_size, shuffle=False, drop_last=True)
    x, y = next(iter(val_dl))
    x, y = x.float().to(device), y.float().to(device)

    eps_model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
    eps_model.load_state_dict(torch.load('./checkpoint_6k/diffusion-700000.pth'))

    diffuser = GaussianDiffusion(
        eps_model,
        image_size=(128, 128),
        timesteps=1000,
        auto_normalize=False
    ).to(device)

    eps_model.self_condition = x
    y_pred = diffuser.sample(batch_size=batch_size)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(batch_size, 3, 128, 128)
    y = y.detach().cpu().numpy()
    y = y.reshape(batch_size, 3, 128, 128)
    np.save('y_pred_new.npy', y_pred)
    np.save('y_pred_ori.npy', y)
