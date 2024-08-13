import torch
import torch.nn as nn
from fairseq.models.transformer import TransformerEncoderLayerBase
from fairseq import utils

class WeightedMultilingualTransformerEncoderLayer(nn.Module):
    """Weighted multilingual transformer layer: mixes src, tgt, and shared representations."""
    def __init__(self, cfg, layer=0, pretrained_weights_path=None):
        super().__init__()
        self.shared_token = "__shared__"
        # Filter for unique languages.
        languages = [self.get_src(lp) for lp in cfg.lang_pairs]
        languages = [self.shared_token] + sorted(set(languages))
        self.models = nn.ModuleDict({lang: TransformerEncoderLayerBase(cfg, layer=layer) for lang in languages})
        self.mixing_params = nn.Parameter(torch.ones([2]))

        # Load pre-trained weights if provided
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)


    def load_pretrained_weights(self, pretrained_weights_path):
        pretrained_state_dict = torch.load(pretrained_weights_path, map_location=torch.device('cuda'))
        for lang, model in self.models.items():
            model.load_state_dict(pretrained_state_dict, strict=False)

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[torch.Tensor] = None):
        shared_repr = self.models[self.shared_token].forward(x, encoder_padding_mask, attn_mask)
        src_repr = self.models[self.lang_pair].forward(x, encoder_padding_mask, attn_mask)
        mixing_weights = utils.softmax(self.mixing_params, dim=-1)
        return mixing_weights[0] * shared_repr + mixing_weights[1] * src_repr 
