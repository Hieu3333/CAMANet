from torch import nn
import torch
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor, nn
from copy import deepcopy as clone

# patch_feats : (B,2*Ns,feat_size)
    # gbl_feats : (B,feat_size)
    # logits : (B,n_classes)
    # cams : (B,n_classes,2*Ns)
class ForeBackLearning(nn.Module):
    def __init__(self, norm=None,dropout=None):
        super(ForeBackLearning, self).__init__()
        self.norm = norm #LayerNorm (feat_size)
        self.dropout = dropout
        if norm:
            self.fore_norm = norm
            self.back_norm = clone(norm)
        if dropout:
            self.fore_dropout = dropout
            self.back_dropout = clone(dropout)

    def forward(self,patch_feats,cam,logits):
        logits = torch.sigmoid(logits) #(B,n_classes) Signoid here but they use Softmax in the paper??
        labels = (logits >= 0.5).float() # Binary classification (1 if prob ≥ 0.5, else 0)
        cam = labels.unsqueeze(-1) * cam #(B,n_classes) -> (B,n_classes,1); (B,n_classes,1) * (B,n_classes,2*Ns) -> (B,n_classes,2*Ns)
        fore_map, _ = torch.max(cam, dim=1, keepdim=True) #(B,1,2*Ns)
        fore_map = self._normalize(fore_map) #fore_map is d_v in the paper
        back_map = 1-fore_map
        fore_rep = torch.matmul(fore_map, patch_feats) #(B,1,2*Ns)@(B,2*Ns,feat_size) -> (B,1,feat_size)
        back_rep = torch.matmul(back_map, patch_feats) # (B,1,feat_size )
        if self.norm:
            fore_rep = self.fore_norm(fore_rep)
            back_rep = self.back_norm(back_rep)
        if self.dropout:
            fore_rep = self.fore_dropout(fore_rep)
            back_rep = self.back_dropout(back_rep)
        return fore_rep, back_rep, fore_map.squeeze(1)

    def _normalize(self, cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization."""
        cams.sub_(cams.min(-1).values[(..., None)])
        cams_max = cams.max(-1).values[(..., None)]
        cams_max[cams_max<1e-12] = 1e-12
        cams.div_(cams_max)
        return cams