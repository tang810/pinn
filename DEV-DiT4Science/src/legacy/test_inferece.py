import os
import sys
from dit4science.inference import inference
import torch
from dit4science.model.dit_models import DiT_models
from dit4science.data.dataset_dit import TurbDataset
from dit4science.download import find_model
from dit4science.model.autoencoder import AutoencoderKL

class DiTTest:
    def __init__(self, data_path, model_ckpt, vae_ckpt_cond, vae_ckpt_res, model_type,channel_idx=-1, prop=None, image_size=128, num_classes=1000, cfg_scale=40):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_path = data_path
        self.channel_idx = channel_idx
        self.prop = prop
        self.image_size = image_size
        self.model_ckpt = model_ckpt
        self.num_classes = num_classes
        self.cfg_scale = cfg_scale
        self.latent_size = self.image_size // 8
        self.vae_ckpt_cond = vae_ckpt_cond
        self.vae_ckpt_res = vae_ckpt_res
        self.model_type = model_type

        # Load the dataset
        self.dataset = TurbDataset(
            dataProp=self.prop, dataDir=self.data_path,
            gaussian_norm=False, channel_idx=self.channel_idx
        )

        # Load the model
        self.model = self.load_model(self.model_type)

        # Load the VAE models
        self.vae_cond = self.load_vae(self.vae_ckpt_cond)
        self.vae_res = self.load_vae(self.vae_ckpt_res)

    def load_model(self,model_type):
        
        model = DiT_models[model_type](
            input_size=self.latent_size,
            num_classes=self.num_classes,
        ).to(self.device)
        state_dict = find_model(self.model_ckpt)
        model.load_state_dict(state_dict)
        return model

    def load_vae(self, vae_ckpt):
        config = {
            "monitor": "val/rec_loss",
            "embed_dim": 4,
            "lossconfig": {
                "target": "dit4science.loss.LPIPSWithDiscriminator",
                "params": {
                    "disc_start": 50001,
                    "kl_weight": 0.000001,
                    "disc_weight": 0.5
                }
            },
            "ddconfig": {
                "double_z": True,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 2, 4, 4],  # num_down = len(ch_mult) - 1
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0
            }
        }
        vae_model = AutoencoderKL(**config).to(self.device)
        vae_model.init_from_ckpt(vae_ckpt)
        return vae_model

    def run_inference(self, num_sampling_steps=250):
        inference(self.dataset, self.model, self.vae_cond, self.vae_res, latent_size=self.latent_size, num_sampling_steps=num_sampling_steps)

if __name__ == "__main__":
    #### Set paths and parameters
    model_type = "DiT-L/2"            #### F.B: DiT-L/2, DiT-XL/2, DiT-B/2...
    data_path = "data/test/"  #### F.B: test data path
    ckpt = "weights/6300000.pt"            #### F.B: ../checkpoints/6300000.pt (Note that this corresponds to the model_type)
    vae_ckpt_cond = "weights/cond.ckpt"     #### F.B: ../checkpoints/vae_cond.ckpt 
    vae_ckpt_res = "weights/func.ckpt"       #### F.B: ../checkpoints/vae_res.ckpt 
    # Create handler and run inference
    handler = DiTTest(data_path, ckpt, vae_ckpt_cond, vae_ckpt_res,model_type)
    handler.run_inference(num_sampling_steps=250)
