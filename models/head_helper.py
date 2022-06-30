import torch
from torch import nn
from slowfast.models.head_helper import ResNetBasicHead, TransformerBasicHead
from slowfast.models.head_helper import MLPHead

class ResNetBasicHead(ResNetBasicHead):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        # Get features
        feat = x.clone().detach()
        #feat = feat.mean(3).mean(2).reshape(feat.shape[0], -1)
        
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if self.detach_final_fc:
            x = x.detach()
        if self.l2norm_feats:
            x = nn.functional.normalize(x, dim=1, p=2)

        if (
            x.shape[1:4] == torch.Size([1, 1, 1])
            and self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        ):
            x = x.view(x.shape[0], -1)

        x_proj = self.projection(x)

        time_projs = []
        if self.predictors:
            x_in = x_proj
            for proj in self.predictors:
                time_projs.append(proj(x_in))

        if not self.training:
            if self.act is not None:
                x_proj = self.act(x_proj)
            # Performs fully convlutional inference.
            if x_proj.ndim == 5 and x_proj.shape[1:4] > torch.Size([1, 1, 1]):
                x_proj = x_proj.mean([1, 2, 3])

        x_proj = x_proj.view(x_proj.shape[0], -1)

        if time_projs:
            return [x_proj] + time_projs, feat
        else:
            return x_proj, feat