import os.path as osp
from collections import OrderedDict
import math, sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from transformers import BertModel, BertTokenizer
from typing import List

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # Loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class CaptionVecGen(nn.Module):
    """
    BERT [CLS] 토큰 임베딩을 반환합니다.
    """
    def __init__(self, device="cpu", model_name="bert-base-uncased"):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name).to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, captions: List[str]) -> torch.Tensor:
        """
        captions: 클래스마다 1개의 문자열 리스트, 길이 = n_cls
        returns: (n_cls, hidden_size) 텐서
        """
        inputs = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        # [CLS] 위치의 임베딩 반환
        return outputs.last_hidden_state[:, 0, :]

    

class PromptLearner(nn.Module):
    """
    CoCoOp + pi_token off-context 구조. 
    고정 프롬프트: 'a photo of a {class} is {pi_token}.'
    - 'a photo of a', 'is', '.'은 고정 (token embedding만 사용, 학습 X)
    - pi_token만 학습 (nn.Parameter)
    """
    def __init__(self, cfg, classnames, captions, clip_model, device='cpu'):
        super().__init__()
        self.device = device
        self.n_cls = len(classnames)
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # meta_net3: (BERT 클래스 설명 → pi_token)
        self.captions = captions
        self.caption_gen = CaptionVecGen(device)
        bert_dim = self.caption_gen.bert.config.hidden_size
        self.bert_emb = None

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.n_ctx = 1  # pi_token 한 개
        # self.ctx = nn.Parameter(torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype))
        # nn.init.normal_(self.ctx, std=0.02)


        self.meta_net3 = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(bert_dim, self.ctx_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(self.ctx_dim // 16, self.ctx_dim))
        ]))
        if cfg.TRAINER.LICOCOOP.PREC == "fp16":
            self.meta_net3.half()
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        # 고정 프롬프트: 'a photo of a {class} is {pi_token}.'
        ctx = "a photo of a"
        be = "is"   

        # 토큰화: 각 class에 대해 ['a photo of a', {class}, 'is', 'X']
        prompts = [f"{ctx} {cname} {be} X" for cname in classnames] # (n_cls, <77)
        tokenized_prompts = torch.cat([clip.tokenize(string) for string in prompts]) # (n_cls, <77)
        with torch.no_grad():
            embeddings = clip_model.token_embedding(tokenized_prompts).type(self.dtype) # (n_cls, <77, bert_dim)

        # 고정 prefix/suffix, class 토큰 idx 파악
        # token_prefix: 'a photo of a {class} is'까지
        # token_pi_idx: 'X' 위치 (pi_token 위치)
        # token_suffix: '.' 이후 토큰
        # (실제 위치는 tokenizer를 통해 확인, 아래는 예시)
        # e.g. for prompt = [SOS, a, photo, of, a, class, is, X, ., EOS]
        #      pi_token은 7번째 토큰 위치(0-based)

        # X 토큰 ID 가져오기
        x_token_id = clip.tokenize("X").squeeze()[1].item() # [SOS, X, 0, 0...] 

        # 각 클래스별로 X 토큰의 실제 위치 찾기
        pi_token_idx = [] # (n_cls, )
        for i in range(self.n_cls):
            idx = (tokenized_prompts[i] == x_token_id).nonzero(as_tuple=False)
            if len(idx) == 0:
                raise ValueError(f"Token 'X' not found in prompt {i} ({prompts[i]})")
            pi_token_idx.append(idx[-1].item())

        # prefix, suffix 각각 클래스별로 저장
        max_len = tokenized_prompts.shape[1]
        prefix_list = []
        suffix_list = []
        for i in range(self.n_cls):
            prefix = embeddings[i, :pi_token_idx[i], :]  # (prefix_len, dim)
            suffix = embeddings[i, pi_token_idx[i] + 1:max_len, :]  # (suffix_len, dim)
            prefix_list.append(prefix)
            suffix_list.append(suffix)

        # 패딩을 맞춰야 함 (prompt 길이 달라질 수 있음)
        # => prefix, suffix를 (n_cls, max_prefix_len, dim), (n_cls, max_suffix_len, dim)로 맞춘다
        max_prefix_len = max([p.shape[0] for p in prefix_list])
        min_suffix_len = min([s.shape[0] for s in suffix_list])

        print(f"max_prefix_len: {max_prefix_len}, min_suffix_len: {min_suffix_len}")
        assert max_prefix_len + min_suffix_len == 76, f"max_prefix_len + min_suffix_len must equal to 76"

        for i in range(self.n_cls):
            # prefix
            pad = (0, 0, 0, max_prefix_len - prefix_list[i].shape[0])
            prefix_list[i] = F.pad(prefix_list[i], pad)
            # suffix
            suffix_list[i] = suffix_list[i][:min_suffix_len, :]  # (suffix_len, dim)

        self.register_buffer("token_prefix", torch.stack(prefix_list, dim=0)) # (n_cls, max_prefix_len, dim)
        self.register_buffer("token_suffix", torch.stack(suffix_list, dim=0)) # (n_cls, max_suffix_len, dim)

        self.tokenized_prompts = tokenized_prompts
        # self.name_lens = name_lens  # 각 클래스 이름의 토큰 길이

    def forward(self):
        assert self.bert_emb is not None
        pi = self.meta_net3(self.bert_emb).unsqueeze(1)  # (n_cls, 1, ctx_dim)
        # self._last_pi = pi # for debugging
        # ctx 파라미터를 사용하지 않고, pi만 사용
        prompts = torch.cat(
            [
                self.token_prefix,     # (n_cls, prefix_len, ctx_dim)
                pi,                   # (n_cls, 1, ctx_dim) - pi_token
                self.token_suffix     # (n_cls, suffix_len, ctx_dim)
            ],
            dim=1
        ).type(self.dtype)
        return prompts
    

class CustomCLIP(nn.Module):
    """이미지 특성과 파이토큰 기반 텍스트 특성 간 내적"""
    def __init__(self, cfg, classnames, captions, clip_model, device='cpu'):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, captions, clip_model, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training and label is not None:
            return F.cross_entropy(logits, label)
        return logits
    
    
@TRAINER_REGISTRY.register()
class LiCoCoOp(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        captions = self.dm.dataset.captions  # 각 클래스별 외적 캡션 리스트
        device = self.device

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LICOCOOP.PREC == "fp32" or cfg.TRAINER.LICOCOOP.PREC == "amp":
            clip_model.float()

        print("Building PiToken-based custom CLIP")
        self.model = CustomCLIP(cfg, classnames, captions, clip_model, device)

        # freeze image/text encoder
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        enabled = [name for name, param in self.model.named_parameters() if param.requires_grad]
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LICOCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def before_train(self):
        super().before_train()

        pl = self.model.prompt_learner
        device = self.device

        # 1) BERT&meta_net3를 올리고 eval 모드
        pl.caption_gen.to(device).eval()
        pl.meta_net3.to(device).eval()

        # 2) 한 번만 캡션 벡터 → bias 계산
        with torch.no_grad():
            bert_emb = pl.caption_gen(pl.captions)   # (n_cls, bert_dim)
        pl.bert_emb = bert_emb.half()

    def forward_backward(self, batch):
        """
        모델 학습.
        """
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.LICOCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label) # forward

            # DEBUG
            # pi = self.model.prompt_learner._last_pi  # (n_cls, 1, ctx_dim)
            # pi_l2_loss = pi.pow(2).mean()  # 전체 평균
            # total_loss = loss + (1e-2) * pi_l2_loss  # lambda=1e-2는 조절 가능


            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

