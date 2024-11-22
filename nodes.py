
import os
import torch
from PIL import Image
import folder_paths
from comfy.utils import ProgressBar, common_upscale

class CXH_StyleModelApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "style_model": ("STYLE_MODEL",),
            "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            "strength": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01
            }),
        }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"
    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, clip_vision_output, style_model, conditioning, strength):
        # 获取 style model 的条件向量，仅计算一次
        style_cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        style_dim = style_cond.shape[1]
        
        # 平滑缩放函数
        def smooth_scale(x, strength):
            return x * (strength ** 2) if strength <= 1.0 else x * (1.0 + (strength - 1.0) ** 0.5)


        # 对 style_cond 应用平滑缩放
        scaled_style = smooth_scale(style_cond, strength)

        # 初始化返回值
        updated_conditioning = []

        for orig_cond, extra_data in conditioning:
            orig_dim = orig_cond.shape[1]

            # 分离基础条件和样式条件
            if orig_dim > style_dim:
                # 如果原始条件已经包含样式向量，则截取基础部分
                base_conditioning = orig_cond[:, :-style_dim]
            else:
                # 否则，整个向量都是基础条件
                base_conditioning = orig_cond

             # 对样式条件进行平滑缩放
            scaled_style = smooth_scale(style_cond, strength)

            # 合并条件
            combined_conditioning = torch.cat((base_conditioning, scaled_style), dim=1)
            updated_conditioning.append([combined_conditioning, extra_data])


        return (updated_conditioning,)

    
