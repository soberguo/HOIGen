import os

import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clipnet as clip
from clipnet.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from torch.autograd import Variable

from collections import Counter

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


def find_key_by_value(my_dict, target_value):

    for key, value in my_dict.items():
        if value == target_value:
            return key

    return None
def  run_vae_generator(args, dataset):

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    
    text_encoder = TextEncoder(clip_model).float().cuda()
    if args.data=='hoi_data':
        prompt_learner_hoi = PromptLearner_hoi(dataset.classnames, clip_model).float().cuda()#base_classnames
    elif args.data=='human_data':
        prompt_learner_h = PromptLearner_h(dataset.classnames, clip_model).float().cuda()
    elif args.data=='object_data':
        prompt_learner_o = PromptLearner_o(dataset.classnames, clip_model).float().cuda()

    HOI_IDX_TO_VERB_IDX =[4, 17, 25, 30, 41, 52, 76, 87, 111, 57, 8, 36, 41, 43, 37, 62, 71, 75, 76, 87,
                           98, 110, 111, 57, 10, 26, 36, 65, 74, 112, 57, 4, 21, 25, 41, 43, 47, 75, 76, 77,
                             79, 87, 93, 105, 111, 57, 8, 20, 36, 41, 48, 58, 69, 57, 4, 17, 21, 25, 41, 52,
                             76, 87, 111, 113, 57, 4, 17, 21, 38, 41, 43, 52, 62, 76, 111, 57, 22, 26, 36,
                             39, 45, 65, 80, 111, 10, 57, 8, 36, 49, 87, 93, 57, 8, 49, 87, 57, 26, 34, 36,
                             39, 45, 46, 55, 65, 76, 110, 57, 12, 24, 86, 57, 8, 22, 26, 33, 36, 38, 39, 41,
                             45, 65, 78, 80, 98, 107, 110, 111, 10, 57, 26, 33, 36, 39, 43, 45, 52, 37, 65,
                             72, 76, 78, 98, 107, 110, 111, 57, 36, 41, 43, 37, 62, 71, 72, 76, 87, 98, 108,
                             110, 111, 57, 8, 31, 36, 39, 45, 92, 100, 102, 48, 57, 8, 36, 38, 57, 8, 26, 34,
                             36, 39, 45, 65, 76, 83, 110, 111, 57, 4, 21, 25, 52, 76, 87, 111, 57, 13, 75, 112,
                               57, 7, 15, 23, 36, 41, 64, 66, 89, 111, 57, 8, 36, 41, 58, 114, 57, 7, 8, 15, 23,
                                 36, 41, 64, 66, 89, 57, 5, 8, 36, 84, 99, 104, 115, 57, 36, 114, 57, 26, 40,
                                 112, 57, 12, 49, 87, 57, 41, 49, 87, 57, 8, 36, 58, 73, 57, 36, 96, 111, 48,
                                57, 15, 23, 36, 89, 96, 111, 57, 3, 8, 15, 23, 36, 51, 54, 67, 57, 8, 14, 15,
                                23, 36, 64, 89, 96, 111, 57, 8, 36, 73, 75, 101, 103, 57, 11, 36, 75, 82,
                                   57, 8, 20, 36, 41, 69, 85, 89, 27, 111, 57, 7, 8, 23, 36, 54, 67, 89, 57, 26, 36, 38, 39,
                                45, 37, 65, 76, 110, 111, 112, 57, 39, 41, 58, 61, 57, 36, 50, 95, 48, 111, 57, 2, 9, 36,
                                90, 104, 57, 26, 45, 65, 76, 112, 57, 36, 59, 75, 57, 8, 36, 41, 57, 8, 14, 15, 23, 36, 54,
                                57, 8, 12, 36, 109, 57, 1, 8, 30, 36, 41, 47, 70, 57, 16, 36, 95, 111, 115, 48, 57, 36, 58,
                                73, 75, 109, 57, 12, 58, 59, 57, 13, 36, 75, 57, 7, 15, 23, 36, 41, 64, 66, 91, 111, 57, 12,
                                36, 41, 58, 75, 59, 57, 11, 63, 75, 57, 7, 8, 14, 15, 23, 36, 54, 67, 88, 89, 57, 12, 36, 56, 58,
                                57, 36, 68, 99, 57, 8, 14, 15, 23, 36, 54, 57, 16, 36, 58, 57, 12, 75, 111, 57, 8, 28, 32, 36,
                                43, 67, 76, 87, 93, 57, 0, 8, 36, 41, 43, 67, 75, 76, 93, 114, 57, 0, 8, 32, 36, 43, 76, 93, 114,
                                57, 36, 48, 111, 85, 57, 2, 8, 9, 19, 35, 36, 41, 44, 67, 81, 84, 90, 104, 57, 36, 94, 97, 57, 8,
                                18, 36, 39, 52, 58, 60, 67, 116, 57, 8, 18, 36, 41, 43, 49, 52, 76, 93, 87, 111, 57, 8, 36, 39, 45,
                                57, 8, 36, 41, 99, 57, 0, 15, 36, 41, 70, 105, 114, 57, 36, 59, 75, 57, 12, 29, 58, 75, 87, 93, 111,
                                  57, 6, 36, 111, 57, 42, 75, 94, 97, 57, 17, 21, 41, 52, 75, 76, 87, 111, 57, 8, 36, 53, 58,
                                  75, 82, 94, 57, 36, 54, 61, 57, 27, 36, 85, 106, 48, 111, 57, 26, 36, 65, 112, 57]

    seen_classnames = dataset.classnames
    print('train classnames number:',len(seen_classnames))
    seen_classnames_dict = {index: value for index, value in enumerate(seen_classnames)}
    if args.dataset=='vcoco_crop':
        
        from vcoco_list import vcoco_values,human_name,object_name
        if args.data=='hoi_data':
            all_classnames=[]
            for i in vcoco_values:
                all_classnames.append(i[0]+' '+i[1])
        elif args.data=='human_data':
            all_classnames=human_name
        elif args.data=='object_data':
            all_classnames=object_name
    elif args.dataset=='hicodet_crop':
        
        from hico_label import all_classnames,object_name,human_name,human_for_verb_name,object_seen_name,human_seen_name
        if args.data=='hoi_data':
            all_classnames=all_classnames
            
        elif args.data=='human_data':
            all_classnames=human_name
        elif args.data=='object_data':
            all_classnames=object_name
    all_classnames_dict = {index: value for index, value in enumerate(all_classnames)}
    
    # train VAE.
    netE = Encoder().cuda()
    netG = Generator().cuda()
    optimizerE = torch.optim.AdamW(netE.parameters(), lr=1e-3)
    optimizerG = torch.optim.AdamW(netG.parameters(), lr=1e-3)
    if args.data=='hoi_data':
        optimizerP_hoi = torch.optim.AdamW(prompt_learner_hoi.parameters(), lr=1e-3)
    elif args.data=='human_data':
        optimizerP_h = torch.optim.AdamW(prompt_learner_h.parameters(), lr=1e-3)
    elif args.data=='object_data':  
        optimizerP_o= torch.optim.AdamW(prompt_learner_o.parameters(), lr=1e-3)
    

    
    for train_idx in range(1, 50 + 1):
        # Train
        netE.train()
        netG.train()

        loss_list = []
        print('Train  VAE Epoch: {:} / {:}'.format(train_idx,50))
     

        for i,(images, target) in enumerate(tqdm(train_loader)):
            images, target = images.cuda(), target.cuda()#images:(256,3,224,224)  target(256,)
            
            

            if args.dataset=='vcoco_crop':
                target_list = []
                for i in range(len(target)):
                    # print(target[i].item())
                    tgt = find_key_by_value(seen_classnames_dict, all_classnames_dict[target[i].item()])
                    target_list.append(tgt)
                target = torch.tensor(target_list).cuda()
            elif args.dataset=='hicodet_crop':
                if args.zs:
                    if args.zs_type=='unseen_object':
                        target_list = []
                        for i in range(len(target)):
                            # print(target[i].item())
                            tgt = find_key_by_value(seen_classnames_dict, all_classnames_dict[target[i].item()])
                            target_list.append(tgt)
                        target = torch.tensor(target_list).cuda()
                    else:
                        if args.data=='hoi':
                            target_list = []
                            for i in range(len(target)):

                                tgt = find_key_by_value(seen_classnames_dict, all_classnames_dict[target[i].item()])
                                target_list.append(tgt)
                            target = torch.tensor(target_list).cuda()

            with torch.no_grad():
                image_features = clip_model.encode_image(images).float()#(256,512)
                image_features /= image_features.norm(dim=-1, keepdim=True)



            netE.zero_grad()
            netG.zero_grad()
            mean, log_var = netE(image_features)#mean:(256,512), log_var:(256,512)
            std = torch.exp(0.5 * log_var)
            z = torch.randn(mean.shape).cuda()
            z = std * z + mean
            bias = netG(z)#bias:(256,512)
            #seen_classnames,object_name
            if args.data=='hoi_data':
                prompt_learner_hoi.get_prefix_suffix_token(seen_classnames, clip_model)
                prompts = prompt_learner_hoi(bias, target)#(256,77,512)

                tokenized_prompts = prompt_learner_hoi.tokenized_prompts#(51,77)
            elif args.data=='human_data':
                prompt_learner_h.get_prefix_suffix_token(seen_classnames, clip_model)
                prompts = prompt_learner_h(bias, target)#(256,77,512)

                tokenized_prompts = prompt_learner_h.tokenized_prompts#(51,77)
            elif args.data=='object_data':
                prompt_learner_o.get_prefix_suffix_token(seen_classnames, clip_model)
                prompts = prompt_learner_o(bias, target)#(256,77,512)

                tokenized_prompts = prompt_learner_o.tokenized_prompts#(51,77)
            text_features = text_encoder(prompts, tokenized_prompts[target])#(256,512)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            recon_features = text_features
            loss = vae_loss(recon_features, image_features, mean, log_var, target)
            if args.data=='hoi_data':
                optimizerP_hoi.zero_grad()
                loss.backward()
                loss_list.append(loss.item())
                optimizerE.step()
                optimizerG.step()
                optimizerP_hoi.step()
            elif args.data=='human_data':
                optimizerP_h.zero_grad()
                loss.backward()
                loss_list.append(loss.item())
                optimizerE.step()
                optimizerG.step()
                optimizerP_h.step()
            elif args.data=='object_data':
                optimizerP_o.zero_grad()
                loss.backward()
                loss_list.append(loss.item())
                optimizerE.step()
                optimizerG.step()
                optimizerP_o.step()
            
        print('Loss: {:.4f}'.format(sum(loss_list)/len(loss_list)))
        if train_idx==50:
            
            
            if args.data=='hoi_data':
                torch.save(prompt_learner_hoi.state_dict(), f'ckpt/{args.dataset[:-5]}/hoi_prompt_learner_{train_idx}.pth')
                torch.save(netE.state_dict(), f'ckpt/{args.dataset[:-5]}/hoi_nete_{train_idx}.pth')
                torch.save(netG.state_dict(), f'ckpt/{args.dataset[:-5]}/hoi_netg_{train_idx}.pth')
            elif args.data=='human_data':
                torch.save(prompt_learner_h.state_dict(), f'ckpt/{args.dataset[:-5]}/human_prompt_learner_{train_idx}.pth')
                torch.save(netE.state_dict(), f'ckpt/{args.dataset[:-5]}/human_nete_{train_idx}.pth')
                torch.save(netG.state_dict(), f'ckpt/{args.dataset[:-5]}/human_netg_{train_idx}.pth')
            elif args.data=='object_data':
                torch.save(prompt_learner_o.state_dict(), f'ckpt/{args.dataset[:-5]}/object_prompt_learner_{train_idx}.pth')
                torch.save(netE.state_dict(), f'ckpt/{args.dataset[:-5]}/object_nete_{train_idx}.pth')
                torch.save(netG.state_dict(), f'ckpt/{args.dataset[:-5]}/object_netg_{train_idx}.pth')

def main(args):

    args.subsample_classes = "all"  # all, base or new
    print("\nRunning configs.")
    random.seed(1)
    torch.manual_seed(1)
    global train_loader
    print("Preparing dataset.")
    from datasets.hoi_dataset import HoiDataset
    dataset = HoiDataset(args,  args.root_path,args.dataset,args.data)
    print(dataset.classnames)
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)
    run_vae_generator(args, dataset)
           

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='vcoco_crop', type=str,choices=('vcoco_crop','hicodet_crop'))
    parser.add_argument('--root_path', default='./datasets/', type=str)
    parser.add_argument('--data', default='hoi_data', type=str,choices=('hoi_data','human_data','object_data'))
    parser.add_argument('--zs', default=False, type=bool)
    parser.add_argument('--zs_type', type=str, default='rare_first', choices=['rare_first', 'non_rare_first', 'unseen_verb', 'unseen_object','uc0', 'uc1', 'uc2', 'uc3', 'uc4'])
    parser.add_argument('--backbone', default="ViT-B/16", type=str)

    

    args=parser.parse_args()
    print(args)
    main(args)