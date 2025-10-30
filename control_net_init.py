import sys
import os
import torch
from omegaconf import OmegaConf
from util import instantiate_from_config
import safetensors.torch  # 导入 safetensors

def get_state_dict(d):
    return d.get('state_dict', d)

# 保持 load_state_dict 不变（虽然主程序不使用它，但如果其他地方调用它，它仍然有用）
def load_state_dict(ckpt_path, location='cuda'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

if __name__ == '__main__':
    # edit your config here
    # you can use .ckpt or .safetensors to train
    pretrained_weight_path = 'checkpoints/v1-5-pruned.safetensors'  # <-- **修改文件名为 .safetensors**
    # the place you want to generate the weight of ini control net
    output_weight_path = 'checkpoints/controlnet_sd1.5_ini.ckpt'
    # put the model config here
    model_config = 'models/cldm_v15.yaml'

    model = create_model(config_path=model_config)

    # load .safetensor
    _, extension = os.path.splitext(pretrained_weight_path)
    if extension.lower() == ".safetensors":
        pretrained_weights = safetensors.torch.load_file(pretrained_weight_path, device='cpu')
        print(f'Loaded safetensors from [{pretrained_weight_path}]')
    elif extension.lower() == ".ckpt":
        pretrained_weights = torch.load(pretrained_weight_path, map_location=torch.device('cpu'))
        print(f'Loaded ckpt from [{pretrained_weight_path}]')
    else:
        raise ValueError(f"Unsupported file format: {extension}")

    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    scratch_dict = model.state_dict()

    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_weight_path)
    print('Done.')