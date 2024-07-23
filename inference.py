import os
import torch
import pocket
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff

from mpl_toolkits.axes_grid1 import make_axes_locatable

# from utils import DataFactory
from utils_tip_cache_and_union_finetune import custom_collate, CustomisedDLE, DataFactory

# from upt import build_detector
from upt_tip_cache_model_free_finetune_distill3 import build_detector
import pdb
import random
from pocket.ops import relocate_to_cpu, relocate_to_cuda




import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models



import clip
from upt_tip_cache_model_free_finetune_distill3 import build_detector
from tqdm import tqdm
from hico_list import hico_verb_object_list,hico_verbs,hico_verbs_sentence,human_name
from hico_label import human_for_verb_name,rare_first_num,human_seen_name,object_seen_name
from torch.utils.data import DataLoader, DistributedSampler
from utils import *
import dino.utils as utils
import dino.vision_transformer as vits
from torchvision import models as torchvision_models
from utils_tip_cache_and_union_finetune import custom_collate, CustomisedDLE, DataFactory
import pdb
from hico_text_label import hico_unseen_index
import vcoco_text_label, hico_text_label
import torch.nn.functional as F
from utils import *
import dino.utils as utils
import dino.vision_transformer as vits
from torchvision import models as torchvision_models

from clipnet.clip import load as clip_load
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clipnet.clip import tokenize as clip_tokenize
import torch.nn as nn
_tokenizer = _Tokenizer()
warnings.filterwarnings("ignore")
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    
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
            prompt = clip_tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].cuda()
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
        prompt_prefix = self.prompt_prefix#'X X X X'
        classnames = [name.replace("_", " ") for name in classnames]
        name_token = [_tokenizer.encode(name) for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip_tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)(51,77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)#(51,77,512)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self, bias, target):

        prefix = self.token_prefix[target]#(256,1,512)
        suffix = self.token_suffix[target]#(256,72,512)
        ctx = self.ctx #(4,512)                    # (n_ctx, ctx_dim)
        bias = bias.unsqueeze(1) #(256,1,512)          # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)#(1,4,512)
        ctx_shifted = ctx + bias   # (256,4,512)        # (batch, n_ctx, ctx_dim)
        prompts = torch.cat([prefix, ctx_shifted, suffix], dim=1)#(256,77,512)
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
            prompt = clip_tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].cuda()
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
        prompt_prefix = self.prompt_prefix#'X X X X'
        classnames = [name.replace("_", " ") for name in classnames]
        name_token = [_tokenizer.encode(name) for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip_tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)(51,77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)#(51,77,512)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self, bias, target):

        prefix = self.token_prefix[target]#(256,1,512)
        suffix = self.token_suffix[target]#(256,72,512)
        ctx = self.ctx #(4,512)                    # (n_ctx, ctx_dim)
        bias = bias.unsqueeze(1) #(256,1,512)          # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)#(1,4,512)
        ctx_shifted = ctx + bias   # (256,4,512)        # (batch, n_ctx, ctx_dim)
        prompts = torch.cat([prefix, ctx_shifted, suffix], dim=1)#(256,77,512)
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
            prompt = clip_tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].cuda()
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
        prompt_prefix = self.prompt_prefix#'X X X X'
        classnames = [name.replace("_", " ") for name in classnames]
        name_token = [_tokenizer.encode(name) for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip_tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)(51,77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)#(51,77,512)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self, bias, target):

        prefix = self.token_prefix[target]#(256,1,512)
        suffix = self.token_suffix[target]#(256,72,512)
        ctx = self.ctx #(4,512)                    # (n_ctx, ctx_dim)
        bias = bias.unsqueeze(1) #(256,1,512)          # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)#(1,4,512)
        ctx_shifted = ctx + bias   # (256,4,512)        # (batch, n_ctx, ctx_dim)
        prompts = torch.cat([prefix, ctx_shifted, suffix], dim=1)#(256,77,512)
        return prompts
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
OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def draw_boxes(ax, boxes):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i+1), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()

def visualise_entire_image(image, output, actions, action=None, thresh=0.2, save_filename=None, failure=False):
    """Visualise bounding box pairs in the whole image by classes"""
    # Rescale the boxes to original image size
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct
    # Find the number of human and object instances
    nh = len(output['pairing'][0].unique()); no = len(boxes)

    scores = output['scores']
    objects = output['objects']
    pred = output['labels']
    

    unique_actions = torch.unique(pred)
    
    if action is not None:
        plt.cla()
        if failure:
            keep = torch.nonzero(torch.logical_and(scores < thresh, pred == action)).squeeze(1)
        else:
            keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
        bx_h, bx_o = boxes[output['pairing']].unbind(0)
        pocket.utils.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
        plt.imshow(image)
        plt.axis('off')
        # pdb.set_trace()
        if len(keep) == 0: return 
        for i in range(len(keep)):
            txt = plt.text(*bx_h[keep[i], :2], f"{scores[keep[i]]:.2f}", fontsize=15, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()
        # plt.show()
        plt.savefig(save_filename, bbox_inches='tight', pad_inches=0.0)
        # plt.savefig(save_filename)
        plt.cla()
        return

    pairing = output['pairing']

    # Print predicted actions and corresponding scores
    unique_actions = torch.unique(pred)
    for verb in unique_actions:
        print(f"\n=> Action: {actions[verb]}")
        sample_idx = torch.nonzero(pred == verb).squeeze(1)
        for idx in sample_idx:
            idxh, idxo = pairing[:, idx] + 1
            print(
                f"({idxh.item():<2}, {idxo.item():<2}),",
                f"score: {scores[idx]:.4f}, object: {OBJECTS[objects[idx]]}."
            )
    
    # Draw the bounding boxes
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    pdb.set_trace()
    ax = plt.gca()
    draw_boxes(ax, boxes)
    # plt.show()
    plt.savefig('visualizations/test.png')

@torch.no_grad()
def main(args):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=0
    )

    random.seed(1234)
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16' 
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'
    args.human_idx = 0 
    dataset = DataFactory(name=args.dataset, partition=args.partition, data_root=args.data_root, clip_model_name=args.clip_model_name)
    conversion = dataset.dataset.object_to_verb if args.dataset == 'hicodet' \
        else list(dataset.dataset.object_to_action.values())
    args.num_classes = 117 if args.dataset == 'hicodet' else 24
    actions = dataset.dataset.verbs if args.dataset == 'hicodet' else \
        dataset.dataset.actions

    # actions = dataset.dataset.interactions
    # object_to_target = dataset.dataset.object_to_verb
    object_n_verb_to_interaction = dataset.dataset.object_n_verb_to_interaction
    #--------------------------------------------
    trainset = DataFactory(name=args.dataset, partition='train2015', data_root=args.data_root, clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type, num_classes=args.num_classes)
    object_name=trainset.dataset.objects
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=0)
    )
    #是否使用dino
    args.dino=True
    #是否使用全局特征
    args.clip_global=True
    #是否加载dino缓存原型特征，true为不加载
    args.dino_load_cache=True
    #是否加载全局缓存原型特征，true为不加载
    args.clip_load_cache=True
    cache_dir = os.path.join('./caches', 'dataset')
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir


    train_loader=None
    if args.dino==True:
        

        dino_model = torchvision_models.__dict__['resnet50'](num_classes=0)
        dino_model.fc = nn.Identity()
        dino_model.cuda()
        utils.load_pretrained_weights(dino_model, "dino/dino_resnet50_pretrain.pth", "teacher", "vit_base", 16)
        dino_model.eval()
        for p in dino_model.parameters():
            p.requires_grad = False
        
        
        print("\nConstructing DINO cache model.")
        
        args.augment_epoch=1
        
        dino_cache_keys, dino_cache_values = build_dino_cache_model(args, dino_model, train_loader)
        print("\nDINO cache model finish.")
    else:
        dino_model =None
        dino_cache_keys, dino_cache_values=None,None

    if args.clip_global:
        
        if args.clip_load_cache==False:
            
            clip_state_dict = torch.load(args.clip_dir_vit, map_location="cpu").state_dict()
            import CLIP_models_adapter_prior2
            clip_model = CLIP_models_adapter_prior2.build_model(state_dict=clip_state_dict, use_adapter=args.use_insadapter, adapter_pos=args.adapter_pos, adapter_num_layers=args.adapter_num_layers)
            from hico_list import hico_verbs_sentence
            from vcoco_list import vcoco_verbs_sentence
            from upt_tip_cache_model_free_finetune_distill3 import CustomCLIP
            if args.num_classes == 117:
                classnames = hico_verbs_sentence#action
            elif args.num_classes == 24:
                classnames = vcoco_verbs_sentence
            model = CustomCLIP(args, classnames=classnames, clip_model=clip_model)
        else:
            model=None
            train_loader_cache=None
        print("\nConstructing CLIP cache model.")
        clip_cache_keys, clip_cache_values = build_clip_cache_model(args, model, train_loader)
        print("\nCLIP cache model finish.")
    else:
        clip_cache_keys, clip_cache_values =None,None
    args.cache_model='gen_feat'
    args.generate_feature=True
    if args.generate_feature:
        print('===>  Generate feature....')
        if args.dataset=='vcoco':
            from vcoco_list import vcoco_keys,vcoco_values,vcoco_human_name,vcoco_object_name,vcoco_seen_values,human_seen_values,object_seen_values
            hoi_seen_classnames=[]
            for i in vcoco_seen_values:
                hoi_seen_classnames.append(i[0]+' '+i[1])
            vcoco_hoi_name=[]
            for j in vcoco_values:
                vcoco_hoi_name.append(j[0]+' '+j[1])
            object_seen_classnames=object_seen_values
            human_seen_classnames=human_seen_values
            hoi_idx_to_obj_idx=[1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 11, 14, 14, 
                                14, 14, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 21, 21, 21, 21, 24, 
                                25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 
                                31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 35, 35, 35, 35, 36, 
                                36, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 
                                42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 47, 
                                48, 48, 48, 48, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52, 53, 53, 53, 53, 54, 
                                54, 54, 54, 55, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 59, 60, 60, 60, 61, 61, 
                                61, 61, 62, 62, 62, 63, 64, 64, 64, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 68, 68, 73, 73, 74, 
                                74, 74, 74, 74, 75, 75, 77, 77, 77, 78, 78, 79, 80]
            hoi_idx_to_obj_idx=[x - 1 for x in hoi_idx_to_obj_idx]
            ori_clip_model, preprocess = clip_load(args.clip_model_name)
            ori_clip_model.eval()
            for p in ori_clip_model.parameters():
                p.requires_grad = False


            
            prompt_learner_hoi = PromptLearner_hoi(hoi_seen_classnames, ori_clip_model).float().cuda()
            prompt_learner_h = PromptLearner_h(human_seen_classnames, ori_clip_model).float().cuda()
            prompt_learner_o = PromptLearner_o(object_seen_classnames, ori_clip_model).float().cuda()
            text_encoder = TextEncoder(ori_clip_model).float().cuda()
            netG_hoi = Generator().cuda()
            netG_h = Generator().cuda()
            netG_o = Generator().cuda()

            netG_hoi.load_state_dict(torch.load(f'./ckpt/vcoco/hoi_netg_50.pth'))

            netG_h.load_state_dict(torch.load(f'./ckpt/vcoco/human_netg_50.pth'))
            
            netG_o.load_state_dict(torch.load(f'./ckpt/vcoco/object_netg_50.pth'))
            
            prompt_learner_o.load_state_dict(torch.load(f'./ckpt/vcoco/object_prompt_learner_50.pth'))
            prompt_learner_hoi.load_state_dict(torch.load(f'./ckpt/vcoco/hoi_prompt_learner_50.pth'))
            prompt_learner_h.load_state_dict(torch.load(f'./ckpt/vcoco/human_prompt_learner_50.pth'))
            netG_hoi.eval()
            netG_h.eval()
            netG_o.eval()
            prompt_learner_h.eval()
            prompt_learner_hoi.eval()
            prompt_learner_o.eval()
            for p in netG_hoi.parameters():
                p.requires_grad = False
            for p in netG_h.parameters():
                p.requires_grad = False
            for p in netG_o.parameters():
                p.requires_grad = False
            for p in prompt_learner_hoi.parameters():
                p.requires_grad = False
            for p in prompt_learner_h.parameters():
                p.requires_grad = False
            for p in prompt_learner_o.parameters():
                p.requires_grad = False

            mlp_hoi=mlp_net(512,512,512).cuda()
            mlp_h=mlp_net(512,512,512).cuda()
            mlp_o=mlp_net(512,512,512).cuda()
            mlp_o.load_state_dict(torch.load(f'./ckpt/vcoco/object_mlp_50.pth'))
            mlp_hoi.load_state_dict(torch.load(f'./ckpt/vcoco/hoi_mlp_50.pth'))
            mlp_h.load_state_dict(torch.load(f'./ckpt/vcoco/human_mlp_50.pth'))
            mlp_hoi.eval()
            mlp_h.eval()
            mlp_o.eval()
            for p in mlp_hoi.parameters():
                p.requires_grad = False
            for p in mlp_h.parameters():
                p.requires_grad = False
            for p in mlp_o.parameters():
                p.requires_grad = False


            with torch.no_grad():
                
                gen_target_hoi_=[]
                gen_target_o_=[]
                gen_target_h_=[]
                gen_feature_hoi_=[]
                gen_feature_h_=[]
                gen_feature_o_=[]
                gen_verb=[]
                
                for i in tqdm(range(100)):
                    # if i%5==0:
                    #     hoi_number=list(range(600))
                    # else:
                        # hoi_number=hico_unseen_index[args.zs_type]
                    hoi_number=list(range(236))
                    # hoi_number=rare_first_num
                    # random.shuffle(hoi_number)
                    # gen_target_hoi2 = torch.tensor(random.choices(hoi_number, k=100)).cuda()
                    gen_target_hoi = torch.tensor(hoi_number).cuda()
                    z_hoi=torch.randn([len(hoi_number),512]).cuda()
                    human_list=[int(i.cpu()) for i in gen_target_hoi]
                    h_number=[]
                    for i in human_list:
                        h_number.append(hoi_idx_to_obj_idx[i])
                    gen_target_h = torch.tensor(h_number).cuda()
                    z_h = torch.randn([len(hoi_number), 512]).cuda()
                    object_list=[int(i.cpu()) for i in gen_target_hoi]
                    o_number=[]
                    for i in object_list:
                        o_number.append(hoi_idx_to_obj_idx[i])
                    gen_target_o = torch.tensor(o_number).cuda()                
                    z_o = torch.randn([len(hoi_number), 512]).cuda()
                    # gen_verb=[]
                    for i in gen_target_hoi:
                        gen_verb.append(vcoco_keys[i][0])
                    bias_hoi = netG_hoi(z_hoi)
                    bias_h = netG_h(z_h)
                    bias_o = netG_o(z_o)
                        # hoi
                    prompt_learner_hoi.get_prefix_suffix_token(vcoco_hoi_name,ori_clip_model)  # update prefix and suffix for new dataset.
                    prompts_hoi = prompt_learner_hoi(bias_hoi,gen_target_hoi)
                    tokenized_prompts_hoi = prompt_learner_hoi.tokenized_prompts
                    text_features_hoi = text_encoder(prompts_hoi, tokenized_prompts_hoi[gen_target_hoi])
                    gen_feature_hoi = text_features_hoi / text_features_hoi.norm(dim=-1, keepdim=True)
                    gen_feature_hoi=mlp_hoi(gen_feature_hoi)


                    
                    # human
                    prompt_learner_h.get_prefix_suffix_token(vcoco_human_name,ori_clip_model)  # update prefix and suffix for new dataset.
                    prompts_h = prompt_learner_h(bias_h,gen_target_h)
                    tokenized_prompts_h = prompt_learner_h.tokenized_prompts
                    text_features_h = text_encoder(prompts_h, tokenized_prompts_h[gen_target_h])
                    gen_feature_h = text_features_h / text_features_h.norm(dim=-1, keepdim=True)
                    gen_feature_h=mlp_h(gen_feature_h)

                    # object
                    prompt_learner_o.get_prefix_suffix_token(vcoco_object_name,ori_clip_model)  # update prefix and suffix for new dataset.
                    prompts_o= prompt_learner_o(bias_o,gen_target_o)
                    tokenized_prompts_o = prompt_learner_o.tokenized_prompts
                    text_features_o = text_encoder(prompts_o, tokenized_prompts_o[gen_target_o])
                    gen_feature_o = text_features_o / text_features_o.norm(dim=-1, keepdim=True)
                    gen_feature_o=mlp_o(gen_feature_o)


                    gen_target_hoi_.append(gen_target_hoi)
                    gen_target_h_.append(gen_target_h)
                    gen_target_o_.append(gen_target_o)
                    gen_feature_hoi_.append(gen_feature_hoi)
                    gen_feature_h_.append(gen_feature_h)
                    gen_feature_o_.append(gen_feature_o)
                gen_target_hoi=torch.cat(gen_target_hoi_,dim=0)
                gen_target_h=torch.cat(gen_target_h_,dim=0)
                gen_target_o=torch.cat(gen_target_o_,dim=0)
                gen_feature_o=torch.cat(gen_feature_o_,dim=0)
                gen_feature_h=torch.cat(gen_feature_h_,dim=0)
                gen_feature_hoi=torch.cat(gen_feature_hoi_,dim=0)
                gen_feature=torch.cat([gen_feature_hoi,gen_feature_h,gen_feature_o],dim=0)
                gen_target=torch.cat([gen_target_hoi,gen_target_h,gen_target_o],dim=0)

        elif args.dataset=='hicodet':
            if args.zs:
                seen_classnames = [trainset.dataset.interactions[i] for i in trainset.remain_hoi_idx]
                seen_classnames_dict={index: value for index, value in enumerate(seen_classnames)}
                unseen_classnames = [trainset.dataset.interactions[i] for i in trainset.filtered_hoi_idx]
                unseen_classnames_dict={index: value for index, value in enumerate(unseen_classnames)}
                all_classnames_dict={index: value for index, value in enumerate(trainset.dataset.interactions)}
            all_classnames=trainset.dataset.interactions
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

            object_to_verb=trainset.dataset.object_to_verb

            ori_clip_model, preprocess = clip_load(args.clip_model_name)
            ori_clip_model.eval()
            for p in ori_clip_model.parameters():
                p.requires_grad = False


            if args.zs==False:
                args.zs_type='no_unseen'
                prompt_learner_hoi = PromptLearner_hoi(all_classnames, ori_clip_model).float().cuda()
                prompt_learner_h = PromptLearner_h(human_name, ori_clip_model).float().cuda()
                prompt_learner_o = PromptLearner_o(object_name, ori_clip_model).float().cuda()
            else:
                if args.zs_type=='unseen_object':
                    prompt_learner_hoi = PromptLearner_hoi(seen_classnames, ori_clip_model).float().cuda()
                    prompt_learner_h = PromptLearner_h(human_seen_name, ori_clip_model).float().cuda()
                    prompt_learner_o = PromptLearner_o(object_seen_name, ori_clip_model).float().cuda()
                else:
                    prompt_learner_hoi = PromptLearner_hoi(seen_classnames, ori_clip_model).float().cuda()
                    prompt_learner_h = PromptLearner_h(human_name, ori_clip_model).float().cuda()
                    prompt_learner_o = PromptLearner_o(object_name, ori_clip_model).float().cuda()
            text_encoder = TextEncoder(ori_clip_model).float().cuda()
            netG_hoi = Generator().cuda()
            netG_h = Generator().cuda()
            netG_o = Generator().cuda()

            netG_hoi.load_state_dict(torch.load(f'./ckpt/{args.zs_type}/hoi_netg_50.pth'))

            netG_h.load_state_dict(torch.load(f'./ckpt/{args.zs_type}/human_netg_50.pth'))
            
            netG_o.load_state_dict(torch.load(f'./ckpt/{args.zs_type}/object_netg_50.pth'))
            
            prompt_learner_o.load_state_dict(torch.load(f'./ckpt/{args.zs_type}/object_prompt_learner_50.pth'))
            prompt_learner_hoi.load_state_dict(torch.load(f'./ckpt/{args.zs_type}/hoi_prompt_learner_50.pth'))
            prompt_learner_h.load_state_dict(torch.load(f'./ckpt/{args.zs_type}/human_prompt_learner_50.pth'))
            netG_hoi.eval()
            netG_h.eval()
            netG_o.eval()
            prompt_learner_h.eval()
            prompt_learner_hoi.eval()
            prompt_learner_o.eval()
            for p in netG_hoi.parameters():
                p.requires_grad = False
            for p in netG_h.parameters():
                p.requires_grad = False
            for p in netG_o.parameters():
                p.requires_grad = False
            for p in prompt_learner_hoi.parameters():
                p.requires_grad = False
            for p in prompt_learner_h.parameters():
                p.requires_grad = False
            for p in prompt_learner_o.parameters():
                p.requires_grad = False

            mlp_hoi=mlp_net(512,512,512).cuda()
            mlp_h=mlp_net(512,512,512).cuda()
            mlp_o=mlp_net(512,512,512).cuda()
            mlp_o.load_state_dict(torch.load(f'./ckpt/{args.zs_type}/object_mlp_50.pth'))
            mlp_hoi.load_state_dict(torch.load(f'./ckpt/{args.zs_type}/hoi_mlp_50.pth'))
            mlp_h.load_state_dict(torch.load(f'./ckpt/{args.zs_type}/human_mlp_50.pth'))
            mlp_hoi.eval()
            mlp_h.eval()
            mlp_o.eval()
            for p in mlp_hoi.parameters():
                p.requires_grad = False
            for p in mlp_h.parameters():
                p.requires_grad = False
            for p in mlp_o.parameters():
                p.requires_grad = False

            with torch.no_grad():
                
                gen_target_hoi_=[]
                gen_target_o_=[]
                gen_target_h_=[]
                gen_feature_hoi_=[]
                gen_feature_h_=[]
                gen_feature_o_=[]
                gen_verb=[]
               
                for _ in tqdm(range(100)):

                    hoi_number=list(range(600))

                    gen_target_hoi = torch.tensor(hoi_number).cuda()
                    z_hoi=torch.randn([len(hoi_number),512]).cuda()
                    human_list=[int(i.cpu()) for i in gen_target_hoi]
                    h_number=[]
                    for i in human_list:
                        h_number.append(HOI_IDX_TO_OBJ_IDX[i])
                    gen_target_h = torch.tensor(h_number).cuda()
                    z_h = torch.randn([len(hoi_number), 512]).cuda()
                    object_list=[int(i.cpu()) for i in gen_target_hoi]
                    o_number=[]
                    for i in object_list:
                        o_number.append(HOI_IDX_TO_OBJ_IDX[i])
                    gen_target_o = torch.tensor(o_number).cuda()                
                    z_o = torch.randn([len(hoi_number), 512]).cuda()
                    # gen_verb=[]
                    for i in gen_target_hoi:
                        gen_verb.append(hico_verbs.index(hico_verb_object_list[i][0]))
                    bias_hoi = netG_hoi(z_hoi)
                    bias_h = netG_h(z_h)
                    bias_o = netG_o(z_o)
                        # hoi
                    prompt_learner_hoi.get_prefix_suffix_token(all_classnames,ori_clip_model)  # update prefix and suffix for new dataset.
                    prompts_hoi = prompt_learner_hoi(bias_hoi,gen_target_hoi)
                    tokenized_prompts_hoi = prompt_learner_hoi.tokenized_prompts
                    text_features_hoi = text_encoder(prompts_hoi, tokenized_prompts_hoi[gen_target_hoi])
                    gen_feature_hoi = text_features_hoi / text_features_hoi.norm(dim=-1, keepdim=True)
                    gen_feature_hoi=mlp_hoi(gen_feature_hoi)


                    
                    # human
                    prompt_learner_h.get_prefix_suffix_token(human_name,ori_clip_model)  # update prefix and suffix for new dataset.
                    prompts_h = prompt_learner_h(bias_h,gen_target_h)
                    tokenized_prompts_h = prompt_learner_h.tokenized_prompts
                    text_features_h = text_encoder(prompts_h, tokenized_prompts_h[gen_target_h])
                    gen_feature_h = text_features_h / text_features_h.norm(dim=-1, keepdim=True)
                    gen_feature_h=mlp_h(gen_feature_h)


                    # object
                    prompt_learner_o.get_prefix_suffix_token(object_name,ori_clip_model)  # update prefix and suffix for new dataset.
                    prompts_o= prompt_learner_o(bias_o,gen_target_o)
                    tokenized_prompts_o = prompt_learner_o.tokenized_prompts
                    text_features_o = text_encoder(prompts_o, tokenized_prompts_o[gen_target_o])
                    gen_feature_o = text_features_o / text_features_o.norm(dim=-1, keepdim=True)
                    gen_feature_o=mlp_o(gen_feature_o)



                    gen_target_hoi_.append(gen_target_hoi)
                    gen_target_h_.append(gen_target_h)
                    gen_target_o_.append(gen_target_o)
                    gen_feature_hoi_.append(gen_feature_hoi)
                    gen_feature_h_.append(gen_feature_h)
                    gen_feature_o_.append(gen_feature_o)
                gen_target_hoi=torch.cat(gen_target_hoi_,dim=0)
                gen_target_h=torch.cat(gen_target_h_,dim=0)
                gen_target_o=torch.cat(gen_target_o_,dim=0)
                gen_feature_o=torch.cat(gen_feature_o_,dim=0)
                gen_feature_h=torch.cat(gen_feature_h_,dim=0)
                gen_feature_hoi=torch.cat(gen_feature_hoi_,dim=0)
                gen_feature=torch.cat([gen_feature_hoi,gen_feature_h,gen_feature_o],dim=0)
                gen_target=torch.cat([gen_target_hoi,gen_target_h,gen_target_o],dim=0)
    else:
        print('No Generate feature.')
        gen_feature=None
        gen_target=None
        gen_verb=None
    num_anno=None
    #---------------------------------------------------------------
    upt = build_detector(args, clip_cache_keys, clip_cache_values,dino_model,dino_cache_keys, dino_cache_values,gen_feature,gen_target, gen_verb,object_to_verb,conversion, object_n_verb_to_interaction=object_n_verb_to_interaction, clip_model_path=args.clip_dir_vit,num_anno=num_anno)
    upt = upt.cuda()
    upt.eval()
    
    if os.path.exists(args.resume):
        print(f"=> Continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Start from a randomly initialised model")
    
    if args.image_path is not None:
        for index in tqdm(range(len(dataset))):
            image, target = dataset[index]
            image = relocate_to_cuda(image)
            output = upt([image])
            output = relocate_to_cpu(output)
            image = dataset.dataset.load_image(
                os.path.join(dataset.dataset._root,
                    dataset.dataset.filename(index)
            ))
            
            filename = target['filename'].split('.')[0] + '_pred.png'
            for action_idx in range(len(actions)):
            # action_idx = args.action
                action_name = actions[action_idx].replace(' ', '_')
                base_path = f'visualization/{args.dataset}/{action_name}'
                if args.zs:
                    base_path = f'visualization/zs/{args.zs_type}/{args.dataset}/{action_name}'
                if args.failure:
                    base_path = f'visualization_fail/{args.dataset}/{action_name}'
                os.makedirs(base_path, exist_ok=True)
                visualise_entire_image(image, output[0], actions, action=action_idx,
                                        thresh=args.action_score_thresh, save_filename=os.path.join(base_path, filename), failure=args.failure)
        return

    else:
        image = dataset.dataset.load_image(args.image_path)
        pdb.set_trace()
        raise NotImplementedError
        image_tensor, _ = dataset.transforms(image, None) 
        image_tensor = relocate_to_cuda(image_tensor)
        output = upt([image_tensor])
        output = relocate_to_cpu(output)

    visualise_entire_image(image, output[0], actions, action=args.action, thresh=args.action_score_thresh, save_filename=f'visualization/{args.dataset}')




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-3, type=float)
    parser.add_argument('--lr-vit', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')
    parser.add_argument('--adapter_pos', type=str, default='all', choices=['all', 'front', 'end', 'random', 'last'])
    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./datasets/')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1233', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='checkpoints/detr-r50-hicodet.pth', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='./checkpoints/non_rare_first/non_rare_first.pt', help='Resume from a model')
    parser.add_argument('--output-dir', default='/checkpoints/out')
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', default=True,action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)
    parser.add_argument('--adapter_num_layers', type=int, default=1)
    parser.add_argument('--visual_mode', default='vit', type=str)
    parser.add_argument('--use_multi_hot',default=True)

    #### add CLIP vision transformer
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)
    parser.add_argument('--clip_visual_layers_vit', default=24, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=768, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=336, type=int)
    parser.add_argument('--clip_visual_width_vit', default=1024, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=14, type=int)


    parser.add_argument('--clip_text_transformer_width_vit', default=768, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=12, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)

    parser.add_argument('--clip_text_context_length_vit', default=77, type=int) # 13 -77
    parser.add_argument('--use_insadapter',default=True, action='store_true')
    parser.add_argument('--use_distill', action='store_true')
    parser.add_argument('--use_consistloss', action='store_true')
    
    parser.add_argument('--use_mean', action='store_true') # 13 -77
    parser.add_argument('--logits_type', default='HO+U+T', type=str) # 13 -77 # text_add_visual, visual
    parser.add_argument('--num_shot', default='2', type=int) # 13 -77 # text_add_visual, visual
    parser.add_argument('--obj_classifier', action='store_true') # 
    parser.add_argument('--classifier_loss_w', default=1.0, type=float)
    parser.add_argument('--file1', default='hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p',type=str)
    parser.add_argument('--interactiveness_prob_thres', default=0.1, type=float)

    parser.add_argument('--prior_type', type=str, default='cbe', choices=['cbe', 'cb', 'ce', 'be', 'c', 'b', 'e'])
    parser.add_argument('--training_set_ratio', type=float, default=1.0)
    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('--zs',default=True, action='store_true') ## zero-shot
    parser.add_argument('--hyper_lambda', type=float, default=2.8)
    parser.add_argument('--use_weight_pred', action='store_true')
    parser.add_argument('--use_mlp_proj', action='store_true')
    parser.add_argument('--zs_type', type=str, default='non_rare_first', choices=['rare_first', 'non_rare_first', 'unseen_verb'])
    parser.add_argument('--domain_transfer', action='store_true') 
    parser.add_argument('--fill_zs_verb_type', type=int, default=0,) # (for init) 0: random; 1: weighted_sum, 
    parser.add_argument('--pseudo_label', action='store_true') 
    parser.add_argument('--tpt', action='store_true') 
    parser.add_argument('--vis_tor', type=float, default=1.0)

    ## prompt learning
    parser.add_argument('--N_CTX', type=int, default=24)  # number of context vectors
    parser.add_argument('--CSC', type=bool, default=False)  # class-specific context
    parser.add_argument('--CTX_INIT', type=str, default='')  # initialization words
    parser.add_argument('--CLASS_TOKEN_POSITION', type=str, default='end')  # # 'middle' or 'end' or 'front'

    parser.add_argument('--prompt_learning', action='store_true') 
    parser.add_argument('--use_templates', action='store_true') 
    parser.add_argument('--LA', action='store_true')  ## Language Aware
    parser.add_argument('--LA_weight', default=0.6, type=float)  ## Language Aware

    parser.add_argument('--feat_mask_type', type=int, default=0,) # 0: dropout(random mask); 1: 
    parser.add_argument('--num_classes', type=int, default=117,) 
    parser.add_argument('--prior_method', type=int, default=0) ## 0: instance-wise, 1: pair-wise, 2: learnable
    parser.add_argument('--box_proj', type=int, default=0,) ## 0: None; 1: f_u = ROI-feat + MLP(uni-box)
    parser.add_argument('--obj_affordance', action='store_true')
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--action', default=None, type=int,
        help="Index of the action class to visualise.")
    parser.add_argument('--action-score-thresh', default=0.2, type=float,
        help="Threshold on action classes.")
    parser.add_argument('--label_choice', default='random', choices=['random', 'single_first', 'multi_first', 'single+multi', 'rare_first', 'non_rare_first', 'rare+non_rare'])

    parser.add_argument('--image_path', default='./HICO_test2015_00000001.jpg', type=str,
        help="Path to an image file.")


    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory to save the images')

    args = parser.parse_args()
    args.failure = False
    main(args)
