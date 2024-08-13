import torch
import torch.nn as nn
from fairseq.models.transformer import TransformerEncoderLayerBase

class LanguageSpecificEncoderLayer(nn.Module):
    def __init__(self, args, layer=0):
        super().__init__()
        self.index_language = args.language_specific_layers[layer]
        all_languages = sorted(set(args.langs))
        self.models = nn.ModuleDict({lang: TransformerEncoderLayerBase(args, layer=layer) for lang in all_languages})


    def forward(self, x, encoder_padding_mask, attn_mask: Optional[torch.Tensor] = None):
        return self.models[self.lang].forward(x, encoder_padding_mask, attn_mask)


class MultilingualTransformer(nn.Module):
    def __init__(self, args, num_layers, lang_pairs, pretrained_weights_path=None):
        super().__init__()
        self.num_layers = num_layers
        self.langs = langs
        
        # Create a stack of language-specific encoder layers
        self.encoder_layers = nn.ModuleList([
            LanguageSpecificEncoderLayer(args, layer=i) for i in range(num_layers)
        ])
        
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)

    def forward(self, src, src_mask, langs):
        # Pass the source through the stack of encoder layers
        x = src
        for layer in self.encoder_layers:
            layer.langs = langs
            x = layer(x, src_mask)
        
        return x

    def load_pretrained_weights(self, pretrained_weights_path):
        pretrained_state_dict = torch.load(pretrained_weights_path, map_location=torch.device('cuda'))
        for layer in self.encoder_layers:
            for lang, model in layer.models.items():
                model.load_state_dict(pretrained_state_dict, strict=False)
