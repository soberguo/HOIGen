import os

import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

# from datasets import build_dataset
from datasets.utils import build_data_loader
import clipnet as clip
from clipnet.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from torch.autograd import Variable
from hico_label import all_classnames, object_name, human_name, human_for_verb_name, object_seen_name, human_seen_name
from collections import Counter
import pickle

_tokenizer = _Tokenizer()
train_tranform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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
        x = x.permute(1, 0, 2)  # NLD -> LND  （256,77,512）->(77,256,512)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).float()

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner_hoi(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 5
        ctx_init = None
        # ctx_init = 'a photo of a person'
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.n_cls = n_cls
        self.n_ctx = n_ctx

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :].cuda()
            prompt_prefix = ctx_init
            self.n_ctx = n_ctx
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        # self.ctx = ctx_vectors # No prompt learning.
        self.prompt_prefix = prompt_prefix
        self.get_prefix_suffix_token(classnames, clip_model)

    def get_prefix_suffix_token(self, classnames, clip_model):
        prompt_prefix = self.prompt_prefix  # 'X X X X'
        classnames = [name.replace("_", " ") for name in classnames]
        name_token = [_tokenizer.encode(name) for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)(51,77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)  # (51,77,512)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self, bias, target):

        prefix = self.token_prefix[target]  # (256,1,512)
        suffix = self.token_suffix[target]  # (256,72,512)
        ctx = self.ctx  # (4,512)                    # (n_ctx, ctx_dim)
        bias = bias.unsqueeze(1)  # (256,1,512)          # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1,4,512)
        ctx_shifted = ctx + bias  # (256,4,512)        # (batch, n_ctx, ctx_dim)
        prompts = torch.cat([prefix, ctx_shifted, suffix], dim=1)  # (256,77,512)
        return prompts


class PromptLearner_h(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = None
        # ctx_init = 'a photo of a'
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.n_cls = n_cls
        self.n_ctx = n_ctx

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :].cuda()
            prompt_prefix = ctx_init
            self.n_ctx = n_ctx
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        #         self.ctx = ctx_vectors # No prompt learning.
        self.prompt_prefix = prompt_prefix
        self.get_prefix_suffix_token(classnames, clip_model)

    def get_prefix_suffix_token(self, classnames, clip_model):
        prompt_prefix = self.prompt_prefix  # 'X X X X'
        classnames = [name.replace("_", " ") for name in classnames]
        name_token = [_tokenizer.encode(name) for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)(51,77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)  # (51,77,512)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self, bias, target):

        prefix = self.token_prefix[target]  # (256,1,512)
        suffix = self.token_suffix[target]  # (256,72,512)
        ctx = self.ctx  # (4,512)                    # (n_ctx, ctx_dim)
        bias = bias.unsqueeze(1)  # (256,1,512)          # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1,4,512)
        ctx_shifted = ctx + bias  # (256,4,512)        # (batch, n_ctx, ctx_dim)
        prompts = torch.cat([prefix, ctx_shifted, suffix], dim=1)  # (256,77,512)
        return prompts


class PromptLearner_o(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = None
        # ctx_init = 'a photo of a'
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.n_cls = n_cls
        self.n_ctx = n_ctx

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :].cuda()
            prompt_prefix = ctx_init
            self.n_ctx = n_ctx
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        #         self.ctx = ctx_vectors # No prompt learning.
        self.prompt_prefix = prompt_prefix
        self.get_prefix_suffix_token(classnames, clip_model)

    def get_prefix_suffix_token(self, classnames, clip_model):
        prompt_prefix = self.prompt_prefix  # 'X X X X'
        classnames = [name.replace("_", " ") for name in classnames]
        name_token = [_tokenizer.encode(name) for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)(51,77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)  # (51,77,512)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self, bias, target):

        prefix = self.token_prefix[target]  # (256,1,512)
        suffix = self.token_suffix[target]  # (256,72,512)
        ctx = self.ctx  # (4,512)                    # (n_ctx, ctx_dim)
        bias = bias.unsqueeze(1)  # (256,1,512)          # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1,4,512)
        ctx_shifted = ctx + bias  # (256,4,512)        # (batch, n_ctx, ctx_dim)
        prompts = torch.cat([prefix, ctx_shifted, suffix], dim=1)  # (256,77,512)
        return prompts


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512 * 1, 2048),
            nn.ReLU(),
        )
        self.mean = nn.Linear(2048, 512)
        self.log_var = nn.Linear(2048, 512)

        self.apply(weights_init)

    def forward(self, x):
        #         x = torch.cat([x, a], dim=1)
        x = self.net(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        n_ctx = 4
        self.net = nn.Sequential(
            nn.Linear(512 * 1, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512 * 1),
        )
        self.apply(weights_init)

    def forward(self, x):
        out = self.net(x)
        return out


def vae_loss(recon_x, x, mean, log_var, target):
    REC = (recon_x - x).pow(2).sum(1).mean()
    KLD = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()
    return (REC + 1 * KLD)


class mlp_net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def mseloss(x, y):
    mseloss = nn.MSELoss()
    return mseloss(x, y)


class mlp_hoi_o_h_net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def find_key_by_value(my_dict, target_value):
    for key, value in my_dict.items():
        if value == target_value:
            return key
    return None


def run_vae_generator(args, dataset):
    HOI_IDX_TO_OBJ_IDX = [
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14,
        14, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 39,
        39, 39, 39, 39, 39, 39, 39, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 56, 56, 56, 56,
        56, 56, 57, 57, 57, 57, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 60, 60,
        60, 60, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58,
        58, 58, 58, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 6, 6, 6, 6, 6,
        6, 6, 6, 62, 62, 62, 62, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 24, 24,
        24, 24, 24, 24, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 34, 34, 34, 34, 34,
        34, 34, 34, 35, 35, 35, 21, 21, 21, 21, 59, 59, 59, 59, 13, 13, 13, 13, 73,
        73, 73, 73, 73, 45, 45, 45, 45, 45, 50, 50, 50, 50, 50, 50, 50, 55, 55, 55,
        55, 55, 55, 55, 55, 55, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 67, 67, 67,
        67, 67, 67, 67, 74, 74, 74, 74, 74, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
        54, 54, 54, 54, 54, 54, 54, 54, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 10, 10, 10, 10, 10, 42, 42, 42, 42, 42, 42, 29, 29, 29, 29, 29, 29, 23,
        23, 23, 23, 23, 23, 78, 78, 78, 78, 26, 26, 26, 26, 52, 52, 52, 52, 52, 52,
        52, 66, 66, 66, 66, 66, 33, 33, 33, 33, 33, 33, 33, 33, 43, 43, 43, 43, 43,
        43, 43, 63, 63, 63, 63, 63, 63, 68, 68, 68, 68, 64, 64, 64, 64, 49, 49, 49,
        49, 49, 49, 49, 49, 49, 49, 69, 69, 69, 69, 69, 69, 69, 12, 12, 12, 12, 53,
        53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 72, 72, 72, 72, 72, 65, 65, 65, 65,
        48, 48, 48, 48, 48, 48, 48, 76, 76, 76, 76, 71, 71, 71, 71, 36, 36, 36, 36,
        36, 36, 36, 36, 36, 36, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31,
        31, 31, 31, 31, 31, 31, 31, 44, 44, 44, 44, 44, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 11, 11, 11, 11, 28, 28, 28, 28, 28, 28, 28, 28,
        28, 28, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 77, 77, 77, 77, 77,
        38, 38, 38, 38, 38, 27, 27, 27, 27, 27, 27, 27, 27, 70, 70, 70, 70, 61, 61,
        61, 61, 61, 61, 61, 61, 79, 79, 79, 79, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 25, 25, 25, 25, 25, 25, 25, 25, 75, 75, 75, 75, 40, 40, 40, 40, 40,
        40, 40, 22, 22, 22, 22, 22
    ]

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    text_encoder = TextEncoder(clip_model).float().cuda()
    if args.data == 'hoi_data':
        prompt_learner_hoi = PromptLearner_hoi(dataset.classnames, clip_model).float().cuda()  # base_classnames
    elif args.data == 'human_data':
        prompt_learner_h = PromptLearner_h(dataset.classnames, clip_model).float().cuda()
    elif args.data == 'object_data':
        prompt_learner_o = PromptLearner_o(dataset.classnames, clip_model).float().cuda()

    seen_classnames = dataset.classnames
    print('train classnames number:', len(seen_classnames))
    seen_classnames_dict = {index: value for index, value in enumerate(seen_classnames)}
    if args.dataset == 'vcoco_crop':

        from vcoco_list import vcoco_values, vococ_human_name, object_name
        if args.data == 'hoi_data':
            all_classnames = []
            for i in vcoco_values:
                all_classnames.append(i[0] + ' ' + i[1])
        elif args.data == 'human_data':
            all_classnames = human_name
        elif args.data == 'object_data':
            all_classnames = object_name
    elif args.dataset == 'hicodet_crop':

        from hico_label import all_classnames, object_name, human_name, human_for_verb_name, object_seen_name, \
            human_seen_name
        all_classnames = all_classnames
    all_classnames_dict = {index: value for index, value in enumerate(all_classnames)}

    # train VAE.

    netG = Generator().cuda()
    mlp = mlp_net(512, 512, 512).cuda()
    optimizer_mlp = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
    if args.dataset == 'hicodet_crop':
        if args.data == 'hoi_data':
            if args.zs:
                prompt_learner_hoi.load_state_dict(torch.load('ckpt/{}/hoi_prompt_learner_50.pth'.format(args.zs_type)))
                netG.load_state_dict(torch.load('ckpt/{}/hoi_netg_50.pth'.format(args.zs_type)))
            else:
                prompt_learner_hoi.load_state_dict(torch.load('ckpt/no_unseen/hoi_prompt_learner_50.pth'))
                netG.load_state_dict(torch.load('ckpt/no_unseen/hoi_netg_50.pth'))
        elif args.data == 'human_data':
            if args.zs:
                prompt_learner_h.load_state_dict(torch.load('ckpt/{}/hoi_prompt_learner_50.pth'.format(args.zs_type)))
                netG.load_state_dict(torch.load('ckpt/{}/hoi_netg_50.pth'.format(args.zs_type)))
            else:
                prompt_learner_h.load_state_dict(torch.load('ckpt/hico/human_prompt_learner_50.pth'))
                netG.load_state_dict(torch.load('ckpt/hico/human_netg_50.pth'))
        elif args.data == 'object_data':
            if args.zs:
                prompt_learner_o.load_state_dict(
                    torch.load('ckpt/{}/object_prompt_learner_50.pth'.format(args.zs_type)))
                netG.load_state_dict(torch.load('ckpt/{}/object_netg_50.pth'.format(args.zs_type)))
            else:
                prompt_learner_o.load_state_dict(torch.load('ckpt/no_unseen/object_prompt_learner_50.pth'))
                netG.load_state_dict(torch.load('ckpt/no_unseen/object_netg_50.pth'))


    elif args.dataset == 'vcoco_crop':
        if args.data == 'hoi_data':
            prompt_learner_hoi.load_state_dict(torch.load('ckpt/vcoco/hoi_prompt_learner_50.pth'))
            netG.load_state_dict(torch.load('ckpt/vcoco/hoi_netg_50.pth'))
        if args.data == 'human_data':
            prompt_learner_h.load_state_dict(torch.load('ckpt/vcoco/human_prompt_learner_50.pth'))
            netG.load_state_dict(torch.load('ckpt/vcoco/human_netg_50.pth'))
        if args.data == 'object_data':
            prompt_learner_o.load_state_dict(torch.load('ckpt/vcoco/object_prompt_learner_50.pth'))
            netG.load_state_dict(torch.load('ckpt/vcoco/object_netg_50.pth'))

    if args.dataset == 'hicodet_crop':
        if args.data == 'hoi_data':
            cache_pickle = pickle.load(open('./new_gt_features/hoi.pickle', 'rb'))
        elif args.data == 'human_data':
            cache_pickle = pickle.load(open('./new_gt_features/human_for_object.pickle', 'rb'))
        elif args.data == 'object_data':
            cache_pickle = pickle.load(open('./new_gt_features/object.pickle', 'rb'))
    elif args.dataset == 'vcoco_crop':
        if args.data == 'hoi_data':
            cache_pickle = pickle.load(open('./new_gt_features/vcoco/hoi.pickle', 'rb'))
        elif args.data == 'human_data':
            cache_pickle = pickle.load(open('./new_gt_features/vcoco/human_for_object.pickle', 'rb'))
        elif args.data == 'object_data':
            cache_pickle = pickle.load(open('./new_gt_features/vcoco/object.pickle', 'rb'))

    for tensor in cache_pickle.values():
        if tensor != []:
            valid_indices = torch.logical_not(torch.isnan(tensor[0]).any(dim=1))
            tensor[0] = tensor[0][valid_indices]

    for train_idx in range(1, 50 + 1):
        # Train

        # netE.train()
        # netG.train()
        mlp.train()
        loss_list = []
        print('Train  VAE Epoch: {:} / {:}'.format(train_idx, 50))

        for i, (images, target) in enumerate(tqdm(train_loader)):
            images, target = images.cuda(), target.cuda()  # images:(256,3,224,224)  target(256,)
            image_feature = []
            for i in target:
                i = i.cpu().detach().numpy()
                random_number = random.randrange(0, cache_pickle[int(i)][0].shape[0])
                image_feature.append(cache_pickle[int(i)][0][random_number].cuda())
            image_features = torch.stack(image_feature)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            if args.data == 'hoi_data' or args.zs_type == 'unseen_object':
                target_list = []
                for i in range(len(target)):
                    # print(target[i].item())
                    tgt = find_key_by_value(seen_classnames_dict, all_classnames_dict[target[i].item()])
                    target_list.append(tgt)
                target = torch.tensor(target_list).cuda()

            mlp.zero_grad()

            z = torch.randn([target.shape[0], 512]).cuda()
            bias = netG(z)  # bias:(256,512)

            if args.data == 'hoi_data':
                prompt_learner_hoi.get_prefix_suffix_token(seen_classnames, clip_model)
                prompts = prompt_learner_hoi(bias, target)  # (256,77,512)

                tokenized_prompts = prompt_learner_hoi.tokenized_prompts  # (51,77)
            elif args.data == 'human_data':
                prompt_learner_h.get_prefix_suffix_token(seen_classnames, clip_model)
                prompts = prompt_learner_h(bias, target)  # (256,77,512)

                tokenized_prompts = prompt_learner_h.tokenized_prompts  # (51,77)
            elif args.data == 'object_data':
                prompt_learner_o.get_prefix_suffix_token(seen_classnames, clip_model)
                prompts = prompt_learner_o(bias, target)  # (256,77,512)
                tokenized_prompts = prompt_learner_o.tokenized_prompts  # (51,77)

            text_features = text_encoder(prompts, tokenized_prompts[target])  # (256,512)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            recon_features = mlp(text_features)
            loss = mseloss(image_features, recon_features)

            loss.backward()
            loss_list.append(loss.item())
            optimizer_mlp.step()

        print('Loss: {:.4f}'.format(sum(loss_list) / len(loss_list)))

        if train_idx == 50:
            if args.data == 'hoi_data':
                torch.save(mlp.state_dict(), 'ckpt/{}/hoi_mlp_{}.pth'.format(args.dataset[:-5], train_idx))
            elif args.data == 'human_data':
                torch.save(mlp.state_dict(), 'ckpt/{}/human_mlp_{}.pth'.format(args.dataset[:-5], train_idx))
            elif args.data == 'object_data':
                torch.save(mlp.state_dict(), 'ckpt/{}/object_mlp_{}.pth'.format(args.dataset[:-5], train_idx))


def main(args):
    args.subsample_classes = "all"  # all, base or new

    print("\nRunning configs.")

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    global train_loader

    print("Preparing dataset.")
    from datasets.hoi_dataset import HoiDataset
    dataset = HoiDataset(args, args.root_path, args.dataset, args.data)
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True,
                                     shuffle=True)
    run_vae_generator(args, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='hicodet_crop', type=str, choices=('vcoco_crop', 'hicodet_crop'))
    parser.add_argument('--root_path', default='./datasets/', type=str)
    parser.add_argument('--data', default='hoi_data', type=str, choices=('hoi_data', 'human_data', 'object_data'))
    parser.add_argument('--zs', default=True, type=bool)
    parser.add_argument('--zs_type', type=str, default='rare_first',
                        choices=['rare_first', 'non_rare_first', 'unseen_verb', 'unseen_object', 'uc0', 'uc1', 'uc2',
                                 'uc3', 'uc4'])
    parser.add_argument('--backbone', default="ViT-B/16", type=str)

    args = parser.parse_args()
    print(args)
    main(args)