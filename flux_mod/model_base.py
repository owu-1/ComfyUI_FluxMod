from comfy.model_base import Flux as FluxOriginal, ModelType, BaseModel
from .model import Flux as FluxModel

class Flux(FluxOriginal):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super(FluxOriginal, self).__init__(model_config, model_type, device=device, unet_model=FluxModel)
