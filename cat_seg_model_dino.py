# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple, List, Dict

import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce
from detectron2.config import configurable
from detectron2.config import CfgNode as CN
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom
from safetensors.torch import load_file
from einops import rearrange
import numpy as np
import os, sys
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
# Make the local dinov3 package importable as top-level `dinov3`.
# The repository contains `cat_seg/dinov3/dinov3/...`. Adding
# the parent folder `cat_seg/dinov3` to sys.path lets `import dinov3` work.
_dinov3_parent = os.path.join(os.path.dirname(__file__), "dinov3")
if _dinov3_parent not in sys.path:
    sys.path.insert(0, _dinov3_parent)
# _metaclip2_parent = os.path.join(os.path.dirname(__file__), "metaclip2")
# if _metaclip2_parent not in sys.path:
#     sys.path.insert(0, _metaclip2_parent)

_nerve_parent = os.path.join(os.path.dirname(__file__), "nerve")
if _nerve_parent not in sys.path:
    sys.path.insert(0, _nerve_parent)

_sam3_parent = os.path.join(os.path.dirname(__file__), "sam3")
if _sam3_parent not in sys.path:
    sys.path.insert(0, _sam3_parent)

_score_parent = os.path.join(os.path.dirname(__file__), "SCORE")
if _score_parent not in sys.path:
    sys.path.insert(0, _score_parent)

from .sam3.sam3 import build_sam3_image_model
from .sam3.sam3.model.sam3_image_processor import Sam3Processor
from .maskadapter.mask_adapter import MASKAdapterHead
from .nerve.diffusion_model.stable_diffusion import diffusion
from transformers import AutoImageProcessor, AutoModel, AutoProcessor
# from .PVTV2.pvtv2 import pvt_v2_b1, pvt_v2_b3, pvt_v2_b2
from .SCORE.score.modeling.backbone.clip_rs import CLIP_RS

# Helper: load safetensors into a model (best-effort, non-strict)
try:
    from safetensors.torch import load_file as _safetensors_load
except Exception:
    _safetensors_load = None

def _load_safetensors_into_model(model: nn.Module, path: str, device: str | None = None):
    """Try to load a safetensors file into model. This is best-effort and uses strict=False.

    It will attempt to load with and without common prefixes like 'model.' or 'backbone.'.
    If `safetensors` is not installed or file missing, this is a no-op.
    """
    if _safetensors_load is None:
        # safetensors not installed
        return False
    if not os.path.isfile(path):
        return False

    # load on cpu first
    try:
        state = _safetensors_load(path, device="cpu")
    except Exception:
        return False

    def try_load(state_dict):
        # move tensors to model device if requested
        if device is not None:
            state_dict = {k: v.to(device) if hasattr(v, 'to') else v for k, v in state_dict.items()}
        try:
            model.load_state_dict(state_dict, strict=False)
            return True
        except Exception:
            return False

    # try direct
    if try_load(state):
        return True

    # try common prefix strips
    prefixes = ["model.", "backbone.", "module."]
    for p in prefixes:
        new = {k[len(p):] if k.startswith(p) else k: v for k, v in state.items()}
        if try_load(new):
            return True

    # give up but log first few keys to help debugging
    try:
        sample_keys = list(state.keys())[:10]
        print(f"safetensors load: could not auto-match keys; sample keys: {sample_keys}")
    except Exception:
        pass
    return False

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1)
        )

    def forward(self, x, x_dion):
        token_dim = x[:, :1, :]
        x = rearrange(x[:, 1:, :], "B (H W) C -> B C H W", H=24)
        weighting_dion = F.adaptive_avg_pool2d(x_dion, 1)
        weighting_dion = self.fc1(weighting_dion)
        y = x * weighting_dion
        y = torch.cat([token_dim, rearrange(y, "B C H W-> B (H W) C ", H=24)], dim=1)
        return y
    

@META_ARCH_REGISTRY.register()
class CATSeg(nn.Module):
    @staticmethod
    def _build_clip_rs_cfg(rsclip_model_name: str, rsclip_pretrained_weights: str) -> CN:
        cfg = CN()
        cfg.MODEL = CN()
        cfg.MODEL.SCORE = CN()
        cfg.MODEL.SCORE.RSCLIP_MODEL_NAME = rsclip_model_name
        cfg.MODEL.SCORE.RSCLIP_PRETRAINED_WEIGHTS = rsclip_pretrained_weights
        return cfg

    def _extract_rs_clip_features(self, clip_images_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Keep RS-CLIP input behavior aligned with SCORE: use 224x224 images.
        # clip_input_rs = F.interpolate(
        #     clip_images_tensor,
        #     size=(224, 224),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        clip_input_rs = clip_images_tensor
        rs_outputs = self.clip_rs(clip_input_rs)
        # print("rs_outputs keys:", rs_outputs["clip_vis_dense"].shape)
        clip_out_rs = F.interpolate(
            rs_outputs["clip_vis_dense"],
            size=(24, 24),
            mode="bilinear",
            align_corners=False,
        )
        if clip_out_rs.dtype != self.linear6.weight.dtype:
            clip_out_rs = clip_out_rs.to(dtype=self.linear6.weight.dtype)
        return self.linear6(clip_out_rs)
        return {
            "rs_clip_vis_dense": rs_outputs["clip_vis_dense"],
            "rs_clip_global": F.normalize(rs_outputs["clip_global"], dim=-1),
        }

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        rsclip_model_name: str,
        rsclip_pretrained_weights: str,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone
        # clip_rs_cfg = self._build_clip_rs_cfg(rsclip_model_name, rsclip_pretrained_weights)
        # self.clip_rs = CLIP_RS(clip_rs_cfg)
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        # self.image_encoder = vit_base()
        # self.image_encoder1 = vit_base()
        # self.image_encoder2 = vit_base()
        # self.image_encoder3 = vit_base()
        # Build RS-CLIP with local RemoteCLIP checkpoint support.

        # Load DINOv3 backbone from the local model directory.
        safetensors_path = os.path.join(os.path.dirname(__file__), "QLIP")
        self.clip_rs = AutoModel.from_pretrained(
            safetensors_path,
            ignore_mismatched_sizes=True
        )
        peft_config = LoraConfig(
            r=4, 
            lora_alpha=32, 
            target_modules=["k_proj", "q_proj", "v_proj"],  # 这里的名称需匹配 DINO 内部层名
            lora_dropout=0.05,
            bias="none"
        )
        self.clip_rs = get_peft_model(self.clip_rs, peft_config)
        # self.processor = AutoProcessor.from_pretrained(safetensors_path)

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)

        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        # self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        # self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        self.upsample3 = nn.ConvTranspose2d(512, self.proj_dim, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(512, self.proj_dim, kernel_size=1, stride=1)
        self.upsample5 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.upsample6 = nn.ConvTranspose2d(320, 256, kernel_size=2, stride=2)

        self.down_channel = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=1)
        self.down_channel1 = nn.ConvTranspose2d(1024*4, 768, kernel_size=1, stride=1)
        self.down_channel2 = nn.ConvTranspose2d(1024*4, 768, kernel_size=1, stride=1)

        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))

        self.linear = nn.Sequential(
            nn.Linear(768, 512),
        )
        self.linear１ = nn.Sequential(
            nn.Linear(768, 512),
        )
        self.linear２ = nn.Sequential(
            nn.Linear(768, 512),
        )
        self.linear３ = nn.Sequential(
            nn.Linear(768, 512),
        )
        # self.cov = nn.Conv2d(768, self.proj_dim, kernel_size=1, stride=1, bias=False)
        self.shared_token = nn.Parameter(torch.zeros(1, 1, self.proj_dim))
        self.linear4 = nn.Sequential(
            nn.Linear(1024, 768),
        )
        self.linear5 = nn.Sequential(
            nn.Linear(1024, 768),
        )
        self.linear6 = nn.ConvTranspose2d(self.proj_dim, 512, kernel_size=1, stride=1)
        self.linear7 = nn.ConvTranspose2d(1024, self.proj_dim, kernel_size=1, stride=1)
        self.linear8 = nn.ConvTranspose2d(1024, self.proj_dim, kernel_size=1, stride=1)
        self.linearcat1 = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.linearcat2 = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.linearcat3 = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.linearcat4 = nn.Sequential(
            nn.Linear(1024, 512),
        )
        cfg = dict(
            clip_model_name='clip_base',
            mask_in_chans=64,
            num_channels=24,
            use_checkpoint=False,
            num_output_maps=1,
        )

        self.mask_adapter = MASKAdapterHead(**cfg)
        # model = build_sam3_image_model(
        #     bpe_path=f"/home/frank/JIAYUANLI/OVRS/OVRS/cat_seg/sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
        #     checkpoint_path='/home/frank/JIAYUANLI/OVRS/OVRS/cat_seg/sam3/pretrained_weights/sam3.pt', 
        #     device="cuda"
        # )
        # self.processor = Sam3Processor(model, confidence_threshold=0.5, device="cuda")
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        self.num_queries = len(self.class_texts)
        self.attention_layers_to_use = [-2, -4, -6]
        self.attn_refine = diffusion(
                        attention_layers_to_use=[-2, -4, -6],
                        model="v2.1", time_step=45,
                        device="cuda", dtype=torch.float16)
        self.attention_thr = 0.2

        self.class_txt_list = []
        for idx in range(len(self.class_texts)):
            self.class_txt_list.append('a photo of ' + str(self.class_texts[idx]))
    def refinement(self, ori_img, pred_mask, out):
        print("pred_mask shape before refinement:", pred_mask.shape)
        print("ori_img shape before refinement:", ori_img.shape)
        pred_mask = F.interpolate(pred_mask[None], size=(64, 64), mode='bilinear',
                                      align_corners=False)[0].flatten(-2).float()
        cross_att = pred_mask.transpose(0, 1)

        class_txt = ""
        # for idx in range(len(self.class_texts)):
        #     class_txt += self.class_texts[idx] + ", "
        self.attn_refine(ori_img.to(self.device), "")
        self_att = torch.cat(
            [self.attn_refine.attention_maps[idx][0] for idx in self.attention_layers_to_use]).float()

        print("self_att shape after cat:", self_att.shape)
        self_att /= torch.amax(self_att, dim=-2, keepdim=True) + 1e-5
        self_att = torch.where(self_att < self.attention_thr, 0, self_att)
        self_att /= self_att.sum(dim=-1, keepdim=True) + 1e-5

        # if self.refinement == "mean":
        # self_att = self_att.mean(0)
        # elif self.refinement == "selection":
        #     self_att = self_att[self.attention_idx]
        # else:
        self_att = reduce(torch.matmul, self_att, torch.eye(self_att.shape[-1], device="cuda"))
        # self_att = F.interpolate(self_att[None].unsqueeze(0), size=(9216, 4096), mode='bilinear',
        #                               align_corners=False)[0]
        print("self_att shape before matmul:", self_att.shape)
        print("cross_att shape before matmul:", cross_att.shape)
        pred_mask = (self_att @ cross_att).transpose(0, 1).reshape(-1, 64, 64)
        if out:
            pred_mask = F.interpolate(pred_mask[None], size=(96, 96), mode='bilinear',
                                      align_corners=False)[0]
        else:
            pred_mask = F.interpolate(pred_mask[None], size=(24, 24), mode='bilinear',
                                        align_corners=False)[0]
        print("pred_mask shape after refinement:", pred_mask.shape)
        return pred_mask

    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "rsclip_model_name": cfg.MODEL.SCORE.RSCLIP_MODEL_NAME,
            "rsclip_pretrained_weights": cfg.MODEL.SCORE.RSCLIP_PRETRAINED_WEIGHTS,
        }
    def get_hf_intermediate_layers(self, model, pixel_values, layer_indexes):
        """
        针对 Hugging Face DINOv3 模型的中间层提取补丁
        参数:
            model: 加载好的 DINOv3 模型对象
            pixel_values: 输入图像 Tensor [B, 3, H, W]
            layer_indexes: 想要提取的层索引列表, 例如 [3, 5, 8, 11]
        返回:
            List[Tensor]: 每个 Tensor 形状为 [B, C, h, w]
        """
        # 1. 前向传播并要求输出隐藏层
        outputs = model(pixel_values, output_hidden_states=True)
        import math
        # Hugging Face 的 hidden_states 包含 (embed_outputs + all_layer_outputs)
        # 所以索引 i 实际上对应的是第 i 层的输出
        all_layers = outputs.hidden_states 
        print("all_layers length:", len(all_layers))
        selected_outs = []
        for idx in layer_indexes:
            # 获取对应层的特征: [Batch, Tokens, Dim]
            feat = all_layers[idx] 
            # 2. 剥离 [CLS] token (假设 [CLS] 在第一个位置)
            # DINOv3 通常有 1 个 CLS token，剩下的就是空间 Patch
            spatial_feat = feat[:, 5:, :] 
            # 3. 计算特征图的长宽 (Reshape)
            B, N, C = spatial_feat.shape
            h = w = int(math.sqrt(N))
            # 转换维度: [B, N, C] -> [B, C, N] -> [B, C, h, w]
            print("spatial_feat shape before reshape:", spatial_feat.shape)
            spatial_feat = spatial_feat.transpose(1, 2).reshape(B, C, h, w)
            selected_outs.append(spatial_feat)
        return selected_outs

    @property
    def device(self):
        return self.pixel_mean.device

    def _get_vision_hidden_states(self, pixel_values: torch.Tensor):
        """Return vision hidden states from the current HF/PEFT model stack.

        Some wrapped model variants may return `None` for
        `self.clip_rs.vision_model(...).hidden_states` even when
        `output_hidden_states=True`. This helper tries multiple compatible
        output paths before giving up.
        """
        hidden_states = None
        try:
            vision_outputs = self.clip_rs.vision_model(
                pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = getattr(vision_outputs, "hidden_states", None)
        except Exception:
            hidden_states = None

        if hidden_states is not None:
            return hidden_states

        # Fallback: route through the full model output. Depending on model
        # type, vision hidden states may live under `vision_model_output`.
        try:
            model_outputs = self.clip_rs(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            if hasattr(model_outputs, "vision_model_output"):
                vision_output = getattr(model_outputs, "vision_model_output")
                hidden_states = getattr(vision_output, "hidden_states", None)
            if hidden_states is None:
                hidden_states = getattr(model_outputs, "hidden_states", None)
        except Exception:
            hidden_states = None

        return hidden_states

    def get_mask(self, image):
        b, c, h, w = image.shape
        seg_logits_all = []
        seg_logits = torch.zeros((self.num_queries, h, w), device="cuda")
        for i in range(image.shape[0]):
            image_i = image[i].squeeze(0)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float):
                inference_state = self.processor.set_image(image_i)
            
            for query_idx, query_word in enumerate(self.class_texts):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)
                semantic_logits = inference_state['masks_logits']
                if semantic_logits.shape != (h, w):
                        semantic_logits = F.interpolate(
                            semantic_logits, 
                            size=(h, w), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze()
                if semantic_logits.shape[0] != 0:
                    semantic_logits = torch.sum(semantic_logits, dim=0)
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
            seg_logits_all.append(seg_logits.unsqueeze(0))

        return torch.cat(seg_logits_all, dim=0)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        # for x in batched_inputs:
        #     print("batched_inputs image shape:", x["instances"]['image_width'])
        images = [x["image"].to(self.device) for x in batched_inputs]
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        self.layers = []

        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
        # clip_images_resized = F.interpolate(clip_images.tensor, size=256, mode='bilinear', align_corners=False)
        clip_images_resized_90 = torch.rot90(clip_images_resized, k=1, dims=(2, 3))
        clip_images_resized_180  = torch.rot90(clip_images_resized, k=2, dims=(2, 3))
        clip_images_resized_270 = torch.rot90(clip_images_resized, k=3, dims=(2, 3))
        # def to_01(x):
        #     return (x - x.min()) / (x.max() - x.min() + 1e-6)
        # inputs = self.processor(text=self.class_txt_list, images=to_01(clip_images_resized), return_tensors="pt", padding=True, do_rescale=False, do_normalize=False)
        # outputs = self.clip_rs(**inputs.to("cuda"))

        # clip_features = self.linear(self.sem_seg_head.predictor.clip_rs.vision_model(clip_images_resized).last_hidden_state)
        # clip_features1 = self.linear1(self.sem_seg_head.predictor.clip_rs.vision_model(clip_images_resized_90).last_hidden_state)
        # clip_features2 = self.linear2(self.sem_seg_head.predictor.clip_rs.vision_model(clip_images_resized_180).last_hidden_state)
        # clip_features3 = self.linear3(self.sem_seg_head.predictor.clip_rs.vision_model(clip_images_resized_270).last_hidden_state)
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        clip_features1 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized_90, dense=True)
        clip_features2 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized_180, dense=True)
        clip_features3 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized_270, dense=True)

        # hidden_states_clip = self.clip_rs.vision_model(clip_images_resized, output_hidden_states=True).hidden_states
        # hidden_states_clip_90 = self.clip_rs.vision_model(clip_images_resized_90, output_hidden_states=True).hidden_states
        # hidden_states_clip_180 = self.clip_rs.vision_model(clip_images_resized_180, output_hidden_states=True).hidden_states
        # hidden_states_clip_270 = self.clip_rs.vision_model(clip_images_resized_270, output_hidden_states=True).hidden_states
        # qclip_features =  self.clip_rs.visual_projection(hidden_states_clip[-1][:, :, :])
        # qclip_features1 = self.clip_rs.visual_projection(hidden_states_clip_90[-1][:, :, :])
        # qclip_features2 = self.clip_rs.visual_projection(hidden_states_clip_180[-1][:, :, :])
        # qclip_features3 = self.clip_rs.visual_projection(hidden_states_clip_270[-1][:, :, :])

        # clip_features += qclip_features
        # clip_features1 += qclip_features1
        # clip_features2 += qclip_features2
        # clip_features3 += qclip_features3
        # clip_features = self.clip_rs.visual_projection(self.clip_rs.vision_model(clip_images_resized).last_hidden_state[:, :, :])
        # clip_features1 = self.clip_rs.visual_projection(self.clip_rs.vision_model(clip_images_resized_90).last_hidden_state[:, :, :])
        # clip_features2 = self.clip_rs.visual_projection(self.clip_rs.vision_model(clip_images_resized_180).last_hidden_state[:, :, :])
        # clip_features3 = self.clip_rs.visual_projection(self.clip_rs.vision_model(clip_images_resized_270).last_hidden_state[:, :, :])

        
        print("clip_features shape:", clip_features.shape)
        # print("clip_features_rs global shape:", clip_features_rs["rs_clip_global"].shape)
        # logits1 = self.get_mask(clip_images_resized)

        # hidden_states = self._get_vision_hidden_states(clip_images_resized)

        # clip_features_dino = self.linear(hidden_states[23][:, 4:, :])
        # clip_features = self.linear(self.image_encoder(clip_images_resized).last_hidden_state[:, 4:, :])
        # # clip_features_dino = self.linear(self.image_encoder(clip_images_resized, output_hidden_states=True).hidden_states[11][:, 4:, :])
        # clip_features1 = self.linear1(self.image_encoder(clip_images_resized_90).last_hidden_state[:, 4:, :])
        # clip_features2 = self.linear2(self.image_encoder(clip_images_resized_180).last_hidden_state[:, 4:, :])
        # clip_features3 = self.linear3(self.image_encoder(clip_images_resized_270).last_hidden_state[:, 4:, :])
        # print("clip_features_dino shape:", clip_features_dino.shape)
        # clip_features += clip_features_dino
        # clip_features =  self.linearcat1(torch.cat([clip_features, clip_features_dino], dim=-1))
        # print(self.alphaparams[0])
        # print(self.betaparams[0])
        # clip_features = clip_features + clip_features_dino
        # token_dim = clip_features[:, :1, :]
        # clip_features_dino = clip_features[:, 1:, :]
        # clip_features_dino_tmp = clip_features_dino
        # clip_features_dino = rearrange(clip_features_dino, "B (H W) C -> B C H W", H=24)
        # clip_features_dino_90 = torch.rot90(clip_features_dino, k=1, dims=(2, 3))
        # clip_features_dino_180 = torch.rot90(clip_features_dino, k=2, dims=(2, 3))
        # clip_features_dino_270 = torch.rot90(clip_features_dino, k=3, dims=(2, 3))
        # clip_features_dino_90 = torch.cat((rearrange(clip_features_dino_90, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features_dino_180 = torch.cat((rearrange(clip_features_dino_180, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features_dino_270 = torch.cat((rearrange(clip_features_dino_270, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features = self.weights(clip_features, clip_features_dino)
        # clip_features = clip_features + self.linear1(clip_features_dino)
        # clip_features1 = clip_features1 + self.linear1(clip_features_dino_90)
        # clip_features2 = clip_features2 + self.linear2(clip_features_dino_180)
        # clip_features3 = clip_features3 + self.linear3(clip_features_dino_270)
        
        # clip_features1 = self.linearcat2(torch.cat([clip_features1, clip_features_dino_90], dim=-1))
        # clip_features2 = self.linearcat3(torch.cat([clip_features2, clip_features_dino_180], dim=-1))
        # clip_features3 = self.linearcat4(torch.cat([clip_features3, clip_features_dino_270], dim=-1))

        # clip_features1 = self.image_encoder(clip_images_resized_90).last_hidden_state[:, 4:, :]
        # clip_features2 = self.image_encoder(clip_images_resized_180).last_hidden_state[:, 4:, :]
        # clip_features3 = self.image_encoder(clip_images_resized_270).last_hidden_state[:, 4:, :]
        # clip_features = self.image_encoder(clip_images_resized)[11]["patch_tokens_norm"]
        # clip_features1 = self.image_encoder(clip_images_resized_90)[11]["patch_tokens_norm"]
        # clip_features2 = self.image_encoder(clip_images_resized_180)[11]["patch_tokens_norm"]
        # clip_features3 = self.image_encoder(clip_images_resized_270)[11]["patch_tokens_norm"]
        # B = clip_features1.shape[0]
        # shared = self.shared_token.expand(B, -1, -1) 
        # clip_features = torch.cat([clip_features, shared], dim=1)
        # clip_features1 = torch.cat([clip_features1, shared], dim=1)
        # clip_features2 = torch.cat([clip_features2, shared], dim=1)
        # clip_features3 = torch.cat([clip_features3, shared], dim=1)
        # print("clip_features shape:", clip_features.shape)
        # image_features = self.linear8(torch.cat([clip_features[:, 1:, :], clip_features_dino_tmp], dim=-1))
        # image_features = clip_features_dino_tmp
        # image_features = clip_features[:, 1:, :] + clip_features_dino_tmp
        # print("image_features shape:", image_features.shape)
        # CLIP ViT features for guidance
        # image_features = clip_features_s[:, 1:, :]
        # image_features_1 = clip_features[:, 1:, :]
        image_features = clip_features[:, 1:, :]
        # print("image_features shape:", image_features.shape)
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24) 
        for i in range(clip_images_resized.shape[0]):
            res3[i] = self.refinement(clip_images_resized[i], res3[i], False)
        # res4 = rearrange(hidden_states_clip[self.layer_indexes[0]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # res5 = rearrange(hidden_states_clip[self.layer_indexes[1]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # res3 = self.down_channel(torch.cat([rearrange(image_features, "B (H W) C -> B C H W", H=24), 
        # rearrange(image_features_1, "B (H W) C -> B C H W", H=24)], dim=1))
        # res3 = rearrange(hidden_states_clip[self.layer_indexes[0]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # hidden_90 = rearrange(hidden_states_clip_90[self.layer_indexes[0]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # hidden_180 = rearrange(hidden_states_clip_180[self.layer_indexes[0]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # hidden_270 = rearrange(hidden_states_clip_270[self.layer_indexes[0]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # hidden_90 = torch.rot90(hidden_90, k=3, dims=(2, 3))
        # hidden_180 = torch.rot90(hidden_180, k=2, dims=(2, 3))
        # hidden_270 = torch.rot90(hidden_270, k=1, dims=(2, 3))
        # res4 = rearrange(hidden_states_clip[self.layer_indexes[0]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # res4 = self.down_channel1(torch.cat([res4, hidden_90, hidden_180, hidden_270], dim=1))

        # hidden_90_1 = rearrange(hidden_states_clip_90[self.layer_indexes[1]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # hidden_180_1 = rearrange(hidden_states_clip_180[self.layer_indexes[1]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # hidden_270_1 = rearrange(hidden_states_clip_270[self.layer_indexes[1]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # hidden_90_1 = torch.rot90(hidden_90_1, k=3, dims=(2, 3))
        # hidden_180_1 = torch.rot90(hidden_180_1, k=2, dims=(2, 3))
        # hidden_270_1 = torch.rot90(hidden_270_1, k=1, dims=(2, 3))
        # res5 = rearrange(hidden_states_clip[self.layer_indexes[1]][:, 1:, :], "B (H W) C -> B C H W", H=24)
        # res5 = self.down_channel2(torch.cat([res5, hidden_90_1, hidden_180_1, hidden_270_1], dim=1))

        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)

        print("res5 shape before upsample:", res5.shape)
        print("res4 shape before upsample:", res4.shape)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        print("res4 after shape:", res4.shape)
        print("res5 after shape:", res5.shape)
        features = {'res5': res5, 'res4': res4, 'res3': res3}

        outputs = self.sem_seg_head([clip_features, clip_features1, clip_features2, clip_features3], features)
        outputs = self.mask_adapter(F.interpolate(res3, size=(outputs.shape[-2], outputs.shape[-1]), mode="bilinear", align_corners=False), outputs) + outputs
        # for i in range(clip_images_resized.shape[0]):
        #     outputs[i] = self.refinement(clip_images_resized[i], outputs[i], True)
        # 1) compute (I - Δ·F)^{-1}

        # 保存原始 raw outputs（未经过 sigmoid / postprocess），以便外部工具使用（例如 CAM 分析）
        # try:
        #     self._last_raw_outputs = outputs
        # except Exception:
        #     self._last_raw_outputs = None
        # # 保存输入图像的一份拷贝（CPU, HWC, uint8）用于可视化叠加
        # try:
        #     img_tensor = batched_inputs[0]["image"].cpu()
        #     img_np = img_tensor.permute(1, 2, 0).numpy()
        #     if img_np.dtype != 'uint8':
        #         img_np = np.clip(img_np, 0, 255).astype('uint8')
        #     self._last_input_image = img_np
        # except Exception:
        #     self._last_input_image = None
        if self.training:
            # --- 语义分割损失 ---
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)

            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss_sem = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg": loss_sem}

            return losses

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])
            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results

    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        print("ENTER CATSeg.forward, sliding_window=", self.sliding_window)
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        
        # 旋转90度
        clip_images_90 = torch.rot90(clip_images, k=1, dims=(2, 3))
        # 旋转180度
        clip_images_180  = torch.rot90(clip_images, k=2, dims=(2, 3))
        # 旋转270度
        clip_images_270 = torch.rot90(clip_images, k=3, dims=(2, 3))
        
        self.layers = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        clip_features1 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_90, dense=True)
        clip_features2 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_180, dense=True)
        clip_features3 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_270, dense=True)

        clip_features_dino = self.linear(self.image_encoder(clip_images).last_hidden_state[:, 4:, :])
        # clip_features_dino = self.linear(self.image_encoder(clip_images_resized, output_hidden_states=True).hidden_states[11][:, 4:, :])
        # clip_features_dino_90 = self.linear1(self.image_encoder(clip_images_resized_90).last_hidden_state[:, 4:, :])
        # clip_features_dino_180 = self.linear2(self.image_encoder(clip_images_resized_180).last_hidden_state[:, 4:, :])
        # clip_features_dino_270 = self.linear3(self.image_encoder(clip_images_resized_270).last_hidden_state[:, 4:, :])
        print("clip_features_dino shape:", clip_features_dino.shape)
        print("clip_features shape before adding dino:", clip_features.shape)
        # clip_features += clip_features_dino
        # clip_features =  self.linearcat1(torch.cat([clip_features, clip_features_dino], dim=-1))
        # print(self.alphaparams[0])
        # print(self.betaparams[0])
        # clip_features = clip_features + clip_features_dino
        token_dim = clip_features_dino[:, :1, :]
        clip_features_dino = clip_features_dino[:, 1:, :]
        clip_features_dino_tmp = clip_features_dino
        clip_features_dino = rearrange(clip_features_dino, "B (H W) C -> B C H W", H=24)
        # clip_features_dino_90 = torch.rot90(clip_features_dino, k=1, dims=(2, 3))
        # clip_features_dino_180 = torch.rot90(clip_features_dino, k=2, dims=(2, 3))
        # clip_features_dino_270 = torch.rot90(clip_features_dino, k=3, dims=(2, 3))
        # clip_features_dino_90 = torch.cat((rearrange(clip_features_dino_90, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features_dino_180 = torch.cat((rearrange(clip_features_dino_180, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features_dino_270 = torch.cat((rearrange(clip_features_dino_270, "B C H W -> B (H W) C"), token_dim), dim=1)
        # clip_features = self.weights(clip_features, clip_features_dino)
        # clip_features = clip_features + self.linear1(clip_features_dino)
        # clip_features1 = clip_features1 + self.linear1(clip_features_dino_90)
        # clip_features2 = clip_features2 + self.linear2(clip_features_dino_180)
        # clip_features3 = clip_features3 + self.linear3(clip_features_dino_270)
        
        # clip_features1 = self.linearcat2(torch.cat([clip_features1, clip_features_dino_90], dim=-1))
        # clip_features2 = self.linearcat3(torch.cat([clip_features2, clip_features_dino_180], dim=-1))
        # clip_features3 = self.linearcat4(torch.cat([clip_features3, clip_features_dino_270], dim=-1))

        # clip_features1 = self.image_encoder(clip_images_resized_90).last_hidden_state[:, 4:, :]
        # clip_features2 = self.image_encoder(clip_images_resized_180).last_hidden_state[:, 4:, :]
        # clip_features3 = self.image_encoder(clip_images_resized_270).last_hidden_state[:, 4:, :]
        # clip_features = self.image_encoder(clip_images_resized)[11]["patch_tokens_norm"]
        # clip_features1 = self.image_encoder(clip_images_resized_90)[11]["patch_tokens_norm"]
        # clip_features2 = self.image_encoder(clip_images_resized_180)[11]["patch_tokens_norm"]
        # clip_features3 = self.image_encoder(clip_images_resized_270)[11]["patch_tokens_norm"]
        # B = clip_features1.shape[0]
        # shared = self.shared_token.expand(B, -1, -1) 
        # clip_features = torch.cat([clip_features, shared], dim=1)
        # clip_features1 = torch.cat([clip_features1, shared], dim=1)
        # clip_features2 = torch.cat([clip_features2, shared], dim=1)
        # clip_features3 = torch.cat([clip_features3, shared], dim=1)
        print("clip_features shape:", clip_features.shape)
        image_features = self.linear8(torch.cat([clip_features[:, 1:, :], clip_features_dino_tmp], dim=-1))
        # image_features = clip_features_dino_tmp
        # image_features = clip_features[:, 1:, :] + clip_features_dino_tmp
        # print("image_features shape:", image_features.shape)
        # CLIP ViT features for guidance
        
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        # res4_hidden = self.linear4(hidden_states[self.layer_indexes[0] ][:, 5:, :])
        # res5_hidden = self.linear5(hidden_states[self.layer_indexes[1] ][:, 5:, :])
        res4_hidden = self.linear4(self.image_encoder(clip_images, output_hidden_states=True).hidden_states[self.layer_indexes[0] ][:, 5:, :])
        res5_hidden = self.linear5(self.image_encoder(clip_images, output_hidden_states=True).hidden_states[self.layer_indexes[1] ][:, 5:, :])
        res4_hidden = rearrange(res4_hidden, "B (H W) C -> B C H W", H=24)
        res5_hidden = rearrange(res5_hidden, "B (H W) C -> B C H W", H=24)
        print("res4 shape:", res4_hidden.shape)
        # res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24) + res4_hidden
        # res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24) + res5_hidden
        res4 = self.linear6(torch.cat([rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24), res4_hidden], dim=1))
        res5 = self.linear7(torch.cat([rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24), res5_hidden], dim=1))

        # outs = self.annother_backbone(clip_images_resized)
        # res5, res4 = self.upsample5(outs[1]), self.upsample6(outs[2])
        # res3 = self.upsample3(outs[3])
        # res7, res6 = self.upsample6(outs[0]), self.upsample5(outs[1])
        # res4 = self.upsample1(res4)
        # res5 = self.upsample2(res5)
        # 假设 self.image_encoder 已是 Dinov3 模型实例，self.layer_indexes = [3, 7]
        # 在 forward 中，得到 intermediate layers（patch tokens 已 reshape 为 B,C,H,W）
        # outs = self.image_encoder.get_intermediate_layers(clip_images_resized, n=self.layer_indexes, reshape=True)
        # --- 替换旧的 self.image_encoder.get_intermediate_layers ---
        # outs = self.get_hf_intermediate_layers(
        #     self.image_encoder, 
        #     clip_images_resized, 
        #     self.layer_indexes
        # )
        # outs = self.image_encoder(clip_images_resized)[self.layer_indexes[0]]["patch_tokens_norm"], self.image_encoder(clip_images_resized)[self.layer_indexes[1]]["patch_tokens_norm"]
        # outs 是 tuple，顺序对应传入的 layer_indexes
        # res4 = rearrange(outs[0], "B (H W) C -> B C H W", H=24) # (B, C, H_patch, W_patch)
        # print("res4 shape before upsample:", res4.shape)
        # res5 = rearrange(outs[1], "B (H W) C -> B C H W", H=24) # (B, C, H_patch, W_patch)
        # 接下来与原代码一致：
        print("res5 shape before upsample:", res5.shape)
        print("res4 shape before upsample:", res4.shape)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        print("res4 after shape:", res4.shape)
        print("res5 after shape:", res5.shape)
        features = {'res5': res5, 'res4': res4, 'res3': res3}
        # res3 = rearrange(clip_features, "B (H W) C -> B C H W", H=24)
        # res4 = self.upsample1(rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24))
        # res5 = self.upsample2(rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24))

        # features = {'res5': res5, 'res4': res4, 'res3': res3,}


        outputs = self.sem_seg_head([clip_features, clip_features1, clip_features2,clip_features3], features)
        

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
