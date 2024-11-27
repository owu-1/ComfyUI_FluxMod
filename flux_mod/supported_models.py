from comfy.supported_models import Flux as FluxOriginal, FluxSchnell as FluxSchnellOriginal
from . import model_base
from comfy.model_base import ModelType

class Flux(FluxOriginal):
    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Flux(self, device=device)
        return out

class FluxSchnell(FluxSchnellOriginal):
    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.Flux(self, model_type=ModelType.FLOW, device=device)
        return out
