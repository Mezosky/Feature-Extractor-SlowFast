"""
This module manages the step forward for different pySlowFast models. 
The idea is to modify the forward step to obtain the characteristics of the videos.

ToDo:
- Check the features for the models.
"""

import math
import torch
from slowfast.models.video_model_builder import ResNet, SlowFast, MViT, X3D
from slowfast.models import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ResnetFeat(ResNet):
    def forward(self, x, bboxes=None):
        x = x[:]  # avoid pass by reference
        x = self.s1(x)
        x = self.s2(x)
        y = []  # Don't modify x list in place due to activation checkpoint.
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            y.append(pool(x[pathway]))
        x = self.s3(y)
        x = self.s4(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x, feat = self.head(x)
        return x, feat

@MODEL_REGISTRY.register()
class SlowFastFeat(SlowFast):
    def forward(self, x, bboxes=None):
        x = x[:]
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
            return x
        else:
            x, feat = self.head(x)
            return x, feat

@MODEL_REGISTRY.register()
class MvitFeat(MViT):

    def _forward_reversible(self, x):
        """
        Reversible specific code for forward computation.
        """
        # rev does not support cls token or detection
        assert not self.cls_embed_on
        assert not self.enable_detection

        x = self.rev_backbone(x)

        if self.use_mean_pooling:
            x = self.fuse(x)
            x = x.mean(1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.fuse(x)
            x = x.mean(1)

        feat = x.clone().detach()
        x = self.head(x)

        return x, feat

    # def forward(self, x):
    #     x = x[0]
    #     x, bcthw = self.patch_embed(x)

    #     T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
    #     H, W = bcthw[-2], bcthw[-1]
    #     B, N, C = x.shape

    #     if self.cls_embed_on:
    #         cls_tokens = self.cls_token.expand(
    #             B, -1, -1
    #         )  # stole cls_tokens impl from Phil Wang, thanks
    #         x = torch.cat((cls_tokens, x), dim=1)

    #     if self.use_abs_pos:
    #         if self.sep_pos_embed:
    #             pos_embed = self.pos_embed_spatial.repeat(
    #                 1, self.patch_dims[0], 1
    #             ) + torch.repeat_interleave(
    #                 self.pos_embed_temporal,
    #                 self.patch_dims[1] * self.patch_dims[2],
    #                 dim=1,
    #             )
    #             if self.cls_embed_on:
    #                 pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
    #             pos_embed = self._get_pos_embed(pos_embed, bcthw)
    #             x = x + pos_embed
    #         else:
    #             pos_embed = self._get_pos_embed(self.pos_embed, bcthw)
    #             x = x + pos_embed

    #     if self.drop_rate:
    #         x = self.pos_drop(x)

    #     if self.norm_stem:
    #         x = self.norm_stem(x)
        
    #     thw = [T, H, W]
    #     for blk in self.blocks:
    #         x, thw = blk(x, thw)

    #     x = self.norm(x)
    #     if self.cls_embed_on:
    #         x = x[:, 0]
    #     else:
    #         x = x.mean(1)

    #     feat = x.clone().detach()

    #     x = self.head(x)
    #     return x, feat
    
    def forward(self, x):

        x = x[0]
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        if self.enable_rev:
            x, feat = self._forward_reversible(x)

        else:
            for blk in self.blocks:
                x, thw = blk(x, thw)

            if self.enable_detection:
                assert not self.enable_rev

                x = self.norm(x)
                if self.cls_embed_on:
                    x = x[:, 1:]

                B, _, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])

                feat = x.clone().detach()
                x = self.head([x], bboxes)

            else:
                if self.use_mean_pooling:
                    if self.cls_embed_on:
                        x = x[:, 1:]
                    x = x.mean(1)
                    x = self.norm(x)
                elif self.cls_embed_on:
                    x = self.norm(x)
                    x = x[:, 0]
                else:  # this is default, [norm->mean]
                    x = self.norm(x)
                    x = x.mean(1)

                feat = x.clone().detach()
                x = self.head(x)

        return x, feat

@MODEL_REGISTRY.register()
class X3DFeat(X3D):  
    def forward(self, x, bboxes=None):
        for module in self.children():
            if "X3DHead" in module._get_name():
                x, feat = module(x)
            else:
                x = module(x)
        return x, feat