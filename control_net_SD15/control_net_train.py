import warnings
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from control_net_SD15.cldm.logger import ImageLogger
from control_net_SD15.cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import time
from datetime import datetime, timedelta
import json
import cv2
import os
import numpy as np
from torch.utils.data import Dataset

class Custom_Dataset(Dataset):
    def __init__(self, prompt_json_path, image_data_path):
        self.data = []
        self.image_data_path = image_data_path

        if not os.path.exists(prompt_json_path):
            raise FileNotFoundError(f"Prompt JSON file not found at: {prompt_json_path}")
        if not os.path.isdir(image_data_path):
            raise NotADirectoryError(f"Image data directory not found at: {image_data_path}")
        with open(prompt_json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source_path = os.path.join(self.image_data_path, source_filename)
        target_path = os.path.join(self.image_data_path, target_filename)
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        if source is None or target is None:
            raise FileNotFoundError(f"Could not read source or target image at index {idx}. Paths: {source_path}, {target_path}")
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == "__main__":

    ######## Configs ##########
    # torch precision
    torch.set_float32_matmul_precision('medium')

    # training data path and checkpoint path
    image_data_path= 'Data/fill50k'
    prompt_json_path='Data/fill50k/prompt.json'
    resume_path =    'checkpoints/controlnet_sd1.5_ini.ckpt'

    # training hyperparameter
    save_memory = False
    # max_steps can be set as -1 --train until used every sample
    batch_size = 4
    max_steps = 6000
    max_epochs = 10

    #LR
    learning_rate = 1e-5
    #log and checkpoint setting
    not_logger = True
    logger_freq = 300
    save_ckpt_every_n_steps = 1000
    save_top_k = -1
    save_weights_only = True
    save_last = True
    sd_locked = True
    only_mid_control = False
    ######## Configs ##########


    model = create_model('models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # ckpt_callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        every_n_train_steps=save_ckpt_every_n_steps,
        save_weights_only=save_weights_only,
        save_top_k=save_top_k,
        filename='my_controlnet_sd15_{epoch:03d}_{step:06d}_{val_loss:.4f}',
        save_last=save_last
    )
    # Misc
    dataset = Custom_Dataset(image_data_path=image_data_path,prompt_json_path=prompt_json_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq, disabled=not_logger)
    trainer = pl.Trainer(accelerator='gpu', devices='auto', precision=32, max_steps=max_steps, max_epochs=max_epochs,
                         callbacks=[logger, checkpoint_callback])
    # Ignore warnings
    warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")
    warnings.filterwarnings("ignore",
                            ".*The dataloader, train_dataloader, does not have many workers which may be a bottleneck*")
    warnings.filterwarnings("ignore",
                            ".*You defined a `validation_step` but have no `val_dataloader`. Skipping val loop*")
    warnings.filterwarnings("ignore",
                            ".*in your `training_step` but the value needs to be floating point. Converting it to torch.float32*")
    # Train!
    t_start = time.time()
    print('training begin')
    trainer.fit(model, dataloader)
    sec = timedelta(seconds=int(time.time() - t_start))
    d = datetime(1, 1, 1) + sec
    print('training end')
    print("time spend: %d days %d hours %d minutes %d seconds" % (d.day - 1, d.hour, d.minute, d.second))