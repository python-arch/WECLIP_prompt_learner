import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import class_names, new_class_names_coco, BACKGROUND_CATEGORY_COCO
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_coco_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP_Plus.PAR import PAR
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)

_tokenizer = _Tokenizer()                    
SOT_TEXT = _tokenizer.encoder['<|startoftext|>']
EOT_TEXT = _tokenizer.encoder['<|endoftext|>']
class PromptLearner(nn.Module):
    def __init__(self, classnames, ctx_len=16, embed_dim=512, device='cuda',
                 class_specific=True, max_seq_len=77):
        super().__init__()
        self.classnames = classnames
        self.ctx_len = ctx_len
        self.class_specific = class_specific
        self.n_cls = len(classnames)
        self.max_seq_len = max_seq_len  # CLIP's maximum sequence length

        ctx_shape = (self.n_cls if class_specific else 1,
                     ctx_len, embed_dim)
        self.context = nn.Parameter(torch.randn(ctx_shape) * 0.02)

        with torch.no_grad():
            suffix_tokens = clip.tokenize(
                [f"{name}." for name in classnames])
        self.register_buffer("suffix_tokens", suffix_tokens)

        self.register_buffer("token_prefix",
                             torch.tensor([SOT_TEXT], dtype=torch.long))
        self.text_embed_dim = embed_dim
        self.device = device

    def forward(self, clip_model):
        sot, eot = SOT_TEXT, EOT_TEXT
        pad = 0

        n_ctx = self.ctx_len
        n_cls = self.n_cls
        device = self.device
        max_seq_len = self.max_seq_len

        if not self.class_specific:
            ctx = self.context.expand(n_cls, -1, -1)
        else:
            ctx = self.context

        suf = self.suffix_tokens.to(device)
        suf_len = (self.suffix_tokens != pad).sum(-1).max().item() - 1

        full = torch.full((n_cls, max_seq_len), pad,
                        dtype=torch.long, device=device)

        actual_seq_len = 1 + n_ctx + suf_len + 1  # sot + context + suffix + eot
        
        if actual_seq_len > max_seq_len:
            available_suf_len = max_seq_len - 1 - n_ctx - 1
            if available_suf_len <= 0:
                raise ValueError(f"Context length {n_ctx} is too long for max sequence length {max_seq_len}")
            suf_len = min(suf_len, available_suf_len)
            actual_seq_len = 1 + n_ctx + suf_len + 1

        full[:, 0] = sot
        full[:, 1 + n_ctx : 1 + n_ctx + suf_len] = suf[:, 1:1 + suf_len]
        full[:, 1 + n_ctx + suf_len] = eot

        x = clip_model.token_embedding(full)             # FP16 under autocast
        x = torch.cat([x[:, :1], ctx.to(x.dtype), x[:, 1+n_ctx:]], dim=1)               # ctx → FP16
        x = x + clip_model.positional_embedding.to(x.dtype)  # ← now FP16 + FP16
        
        # print(x.dtype)
        x = x.permute(1, 0, 2)  # [seq_len, batch, dim]
        x, _ = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, dim]
        x = clip_model.ln_final(x)
        
        eot_mask = full == eot
        eot_pos = eot_mask.float().argmax(-1)
        
        text_feats = x[torch.arange(n_cls), eot_pos] @ clip_model.text_projection
        
        return text_feats / text_feats.norm(dim=-1, keepdim=True)
class WeCLIP_Plus(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, dino_model=None, dino_fts_dim=768, decoder_layers=3,
                 embedding_dim=256, in_channels=512, dataset_root_path=None, clip_flag=16, ctx_len=16, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dino_fts_fuse_dim = dino_fts_dim
        self.clip_flag = clip_flag


        self.encoder, _ = clip.load(clip_model, device=device)
        
        self.encoder = self.encoder.float()

        for name, param in self.encoder.named_parameters():
            if clip_flag == 14 and '23' not in name:
                param.requires_grad=False
            if clip_flag == 16 and "11" not in name:
                 param.requires_grad=False
            if any(x in name for x in ['token_embedding', 'positional_embedding', 'ln_final', 'text_projection']) or ('transformer' in name and 'visual' not in name):
                    param.requires_grad = True

        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', dino_model)

        for name, param in self.dino_encoder.named_parameters():
            param.requires_grad = False

        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)
        
        self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)
        
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=decoder_layers, heads=8, output_dim=self.num_classes)

        self.root_path = None if dataset_root_path is None else os.path.join(dataset_root_path, 'SegmentationClass')
        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform, clip_flag=clip_flag)
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        self.iter_num = 0
        self.require_all_fts = True
        
        self.prompt_learner_fg = PromptLearner(new_class_names_coco, ctx_len,
                                               self.encoder.text_projection.shape[1],
                                               device=device, class_specific=True)
        self.prompt_learner_bg = PromptLearner(BACKGROUND_CATEGORY_COCO, ctx_len,
                                               self.encoder.text_projection.shape[1],
                                               device=device, class_specific=False)

    def get_param_groups(self):
        param_groups = [[], [], [], [], []]

        for param in list(self.prompt_learner_fg.parameters()):
            param_groups[4].append(param)
        for param in list(self.prompt_learner_bg.parameters()):
            param_groups[4].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        for param in list(self.dino_decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, img, img_names='2007_000032', mode='train'):
        all_img_tokens_list = []
        cam_list = []
        b, c, h, w = img.shape
        if not self.training:
            self.encoder.eval()
        else:
            self.encoder.visual.eval()  # Only visual part
        self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

        with torch.no_grad():
            dino_img_h, dino_img_w = (h//14)*14, (w//14)*14
            dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
            dino_ftses = self.dino_encoder.forward_features(dino_img)
            dino_fts = dino_ftses['x_norm_patchtokens']

        fg_text_features = self.prompt_learner_fg(self.encoder)
        bg_text_features = self.prompt_learner_bg(self.encoder)
        
        fts_all_stack = torch.stack(fts_all, dim=0)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        
        if self.require_all_fts==True:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//self.clip_flag, w //self.clip_flag)

        all_img_tokens = all_img_tokens[-1].unsqueeze(0)

        fts = self.decoder_fts_fuse(all_img_tokens)
        _, _, fts_h, fts_w = fts.shape

        if isinstance(dino_fts, list):
            for d_i, dino_fts_single in enumerate(dino_fts):
                dino_fts_single = dino_fts_single.reshape([b, dino_img_h // 14, dino_img_w // 14, -1]).permute(0, 3, 1, 2)
                dino_fts[d_i] = dino_fts_single

            dino_fts = torch.stack(dino_fts)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts)
            dino_h, dino_w = dino_img_h // 14, dino_img_w // 14

        else:
            dino_fts = dino_fts.reshape([b, dino_img_h//14, dino_img_w//14, -1]).permute(0,3,1,2)
            _, _, dino_h, dino_w = dino_fts.shape
            dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))
        
        dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)

        seg_clip, seg_attn_weight_list_clip = self.decoder(fts)
        seg_dino, seg_attn_weight_list_dino = self.decoder(dino_fts)

        clip_dino_fts = torch.cat([fts, dino_fts], dim=1)

        seg_dino_prob = F.softmax(0.5*seg_dino+0.5*seg_clip, dim=1)
        seg_dino_prob = seg_dino_prob.detach()

        attn_fts = F.interpolate(clip_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        if self.training:
            for i, img_name in enumerate(img_names):
                img_path = os.path.join(self.root_path, 'train', str(img_name) + '.png')
                img_i = img[i]
                cam_fts = cam_fts_all[i]
                cam_attn = attn_weight_stack[i]

                seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

                require_seg_trans = True
                seg_dino_cam = seg_dino_prob[i]

                cam_refined_list, keys, w, h = perform_single_coco_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                              bg_text_features, fg_text_features,
                                                                              self.grad_cam,
                                                                              mode=mode,
                                                                              require_seg_trans=require_seg_trans,
                                                                              seg_dino_cam=seg_dino_cam,
                                                                              clip_flag=self.clip_flag
                                                                              )

                cam_dict = generate_cam_label(cam_refined_list, keys, w, h)

                cams = cam_dict['refined_cam'].cuda()

                bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()

                cams = torch.cat([bg_score, cams], dim=0).cuda()

                valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
                valid_key = torch.from_numpy(valid_key).cuda()

                with torch.no_grad():
                    cam_labels = _refine_cams(self.par, img[i], cams, valid_key)

                cam_list.append(cam_labels)

            all_cam_labels = torch.stack(cam_list, dim=0)

        if self.training:
            return seg_clip, seg_dino, all_cam_labels, attn_pred, fg_text_features, bg_text_features
        else:
            return seg_clip, seg_dino


if __name__=="__main__":
    wetr = WeCLIP_Plus(num_classes=20, clip_model='ViT-B/16', dino_model='dinov2_vitb14', 
                       dataset_root_path='path/to/dataset', embedding_dim=256).cuda()
    wetr.train()
    dummy_input = torch.randn(2, 3, 512, 512, device="cuda") 
    
    loss = wetr(dummy_input)[0].sum()
    loss.backward()
    print("Prompt FG grad:", any(p.grad is not None for p in wetr.prompt_learner_fg.parameters()))
    print("Prompt BG grad:", any(p.grad is not None for p in wetr.prompt_learner_bg.parameters()))
    
   