import folder_paths
import comfy.sd
from .sd import load_diffusion_model
import torch


class FluxModCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "guidance_ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "ExtraModels/FluxMod"
    TITLE = "FluxModCheckpointLoader"

    def load_unet(self, unet_name, guidance_ckpt_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model_patcher = load_diffusion_model(unet_path, model_options=model_options)

        print(type(model_patcher.model))

        guidance_ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", guidance_ckpt_name)

        state_dict = comfy.utils.load_torch_file(guidance_ckpt_path)
        model_patcher.get_model_object("diffusion_model").distilled_guidance_layer.load_state_dict(state_dict)

        return (model_patcher,)


NODE_CLASS_MAPPINGS = {
    "FluxModCheckpointLoader" : FluxModCheckpointLoader
}
