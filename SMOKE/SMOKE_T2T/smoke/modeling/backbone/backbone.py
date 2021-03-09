from collections import OrderedDict

from torch import nn

from smoke.modeling import registry
from . import dla
from timm.models.helpers import load_pretrained, load_checkpoint
from .T2T_ViT.T2T_ViT import T2T_ViT

def _cfg(url='/remote-home/xieziyang/Code/SMOKE/SMOKE_T2T/smoke/modeling/backbone/T2T_ViT/pretrained_pth/80.7_T2T_ViT_t_14.pth', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 1280, 384), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}

@registry.BACKBONES.register("DLA-34-DCN")
def build_dla_backbone(cfg):
    body = dla.DLA(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
    return model

@registry.BACKBONES.register("T2t_vit_t_14")
def build_T2T_backbond(cfg, pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    body = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    body.default_cfg = default_cfgs['T2t_vit_t_14']
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
