from safetensors.torch import load_file

modela = load_file("/storage/dev/nyanko/naifu-flux/checkpoint/model_e0_s1000.safetensors")
modelb = load_file("/storage/local/tmpfiles/checkpoint/model_e6_s1000.safetensors")

modela = {k.replace("model.", ""): v for k, v in modela.items()}
modelb = {k.replace("model.", ""): v for k, v in modelb.items()}
modelc = load_file("/storage/dev/nyanko/flux-dev/flux1-dev.sft")
from termcolor import colored

def compare_models_with_colors_one_line(modela, modelb, modelc):
    # 逐层比较权重偏移
    for layer in modela:
        weights_a = modelc[layer]
        weights_b = modelb[layer]
        weights_c = modela[layer]
        
        # 计算与modela相比的百分比偏移
        shift_b = ((weights_b - weights_a) / weights_a.abs()).mean() * 100
        shift_c = ((weights_c - weights_a) / weights_a.abs()).mean() * 100
        
        # 设置颜色和符号
        shift_b_sign = '+' if shift_b > 0 else '-'
        shift_c_sign = '+' if shift_c > 0 else '-'
        
        color_b = 'green' if shift_b > 0 else 'red'
        color_c = 'green' if shift_c > 0 else 'red'
        
        shift_b_str = f"{shift_b_sign}{abs(shift_b):.2f}%"
        shift_c_str = f"{shift_c_sign}{abs(shift_c):.2f}%"
        
        colored_shift_b = colored(shift_b_str, color_b)
        colored_shift_c = colored(shift_c_str, color_c)
        
        # 单行打印结果
        print(f"Layer: {layer} | Model_camera: {colored_shift_b} | Model_fulla: {colored_shift_c}")

# 使用示例：假设你已经有modela, modelb, modelc三个state_dict
compare_models_with_colors_one_line(modela, modelb, modelc)