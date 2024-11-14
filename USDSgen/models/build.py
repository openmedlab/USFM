from mmengine.config import Config, ConfigDict
from mmseg.registry import MODELS
from omegaconf import DictConfig, OmegaConf
from timm.models import create_model

from usdsgen.modules.backbone.segbackbone import HVITBackbone4Seg
from usdsgen.modules.backbone.vision_transformer import build_vit
from usdsgen.modules.head.seg.ATMHead import ATMHead
from usdsgen.modules.losses import atm_loss
from usdsgen.utils.modelutils import load_pretrained

# import model modules


def build_model(config, logger):
    if config.model.model_type == "FM":
        if config.task == "Cls":
            model = build_vit(config.model.model_cfg, logger)
        elif config.task == "Seg":
            model = MODELS.build(
                ConfigDict(OmegaConf.to_container(config.model.model_cfg, resolve=True))
            )
            load_pretrained(config.model.model_cfg.backbone, model.backbone, logger)
        else:
            raise NotImplementedError(f"Unknown model: {config.model}")
    else:
        model = create_model(**config.model.model_cfg)
    return model
