from tqdm import tqdm

import torch
import os

def build_clip_cache_model(args, clip_model, train_loader_cache):
    
    if args.clip_load_cache == False:    
        cache_keys = [[] for i in range(args.num_classes)]
        cache_values = [[] for i in range(args.num_classes)]
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(1):
                

                print('Augment Epoch: {:} / {:}'.format(augment_idx, 1))
                for _, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images_clip = [im[1] for im in images]
                    images = torch.stack(images_clip).cuda()
                    clip_model=clip_model.cuda()
                    feat_global,feat_local = clip_model.image_encoder(images)
                    train_features = []
                    for idx_local in range(feat_local.shape[0]):

                        global_features=feat_global[idx_local]

                        global_features=global_features/global_features.norm(dim=-1, keepdim=True)
                        train_features.append(global_features)
                    train_features=torch.stack(train_features)
                    if augment_idx == 0:
                        
                        for ii,verb_i in enumerate(target):
                            values = torch.zeros((args.num_classes,))
                            if args.dataset=='hicodet':
                                verb_for=verb_i['verb']
                            elif args.dataset=='vcoco':
                                verb_for=verb_i['actions']
                            for j in verb_for:
                                values[j.item()]=1

                            for k in torch.nonzero(values):
                                cache_values[k.item()].append(values)
                                cache_keys[k.item()].append(train_features[ii,:])

        new_cache_keys = [[] for i in range(args.num_classes)]
        new_cache_values = [[] for i in range(args.num_classes)]
        for i in range(args.num_classes):
            topk_idx = torch.randperm(len(cache_values[i]))[:args.num_shot]
            for idx in topk_idx:
                new_cache_values[i].append(cache_values[i][idx])
                new_cache_keys[i].append(cache_keys[i][idx]) 
            if new_cache_values[i]==[]:
                for _ in range(args.num_shot):
                    new_cache_keys[i].append(torch.randn(512).cuda())
                    value=torch.zeros(args.num_classes)
                    value[i]=1
                    new_cache_values[i].append(value)
            new_cache_keys[i]=torch.stack(new_cache_keys[i])
            new_cache_values[i]=torch.stack(new_cache_values[i])
        new_cache_keys=torch.cat(new_cache_keys)
        new_cache_values=torch.cat(new_cache_values)
        

        new_cache_keys /= new_cache_keys.norm(dim=-1, keepdim=True)
        new_cache_keys = new_cache_keys.permute(1, 0)

        if not os.path.exists('./caches/dataset'):
            os.makedirs('./caches/dataset')
        if args.dataset=='hicodet':
            if args.zs:
                
                torch.save(new_cache_keys, os.path.join('./caches', 'dataset') + f'/clip_keys_{args.zs_type}_{args.num_shot}.pt')
                torch.save(new_cache_values, os.path.join('./caches', 'dataset') + f'/clip_values_{args.zs_type}_{args.num_shot}.pt')
            else:
                torch.save(new_cache_keys, os.path.join('./caches', 'dataset') + '/clip_keys_2shots.pt')
                torch.save(new_cache_values, os.path.join('./caches', 'dataset') + '/clip_values_2shots.pt')
                
        elif args.dataset=='vcoco':
            torch.save(new_cache_keys, os.path.join('./caches', 'dataset') + '/vcoco_clip_keys_2shots.pt')
            torch.save(new_cache_values, os.path.join('./caches', 'dataset') + '/vcoco_clip_values_2shots.pt')

    else:
        if args.dataset=='hicodet':
            if args.zs:
                new_cache_keys = torch.load(os.path.join('./caches', 'dataset') +  f'/clip_keys_{args.zs_type}_{args.num_shot}.pt')
                new_cache_values = torch.load(os.path.join('./caches', 'dataset') + f'/clip_values_{args.zs_type}_{args.num_shot}.pt')
            else:
                new_cache_keys = torch.load(os.path.join('./caches', 'dataset') +  '/clip_keys_2shots.pt')
                new_cache_values = torch.load(os.path.join('./caches', 'dataset') + '/clip_values_2shots.pt')
        elif args.dataset=='vcoco': 
            new_cache_keys = torch.load(os.path.join('./caches', 'dataset') + '/vcoco_clip_keys_'  + "2shots.pt")
            new_cache_values = torch.load(os.path.join('./caches', 'dataset') + '/vcoco_clip_values_'  + "2shots.pt")

    return new_cache_keys, new_cache_values

def build_dino_cache_model(args, dino_model, train_loader_cache):
    
    if args.dino_load_cache == False:    
        cache_keys = [[] for i in range(args.num_classes)]
        cache_values = [[] for i in range(args.num_classes)]

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(args.augment_epoch):
                # train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, args.augment_epoch))
                for _, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images_clip = [im[1] for im in images]
                    images = torch.stack(images_clip).cuda()
                    image_features = dino_model(images)

                    if augment_idx == 0:

                        for ii,verb_i in enumerate(target):
                            values = torch.zeros((args.num_classes,))
                            if args.dataset=='hicodet':
                                verb_for=verb_i['verb']
                            elif args.dataset=='vcoco':
                                verb_for=verb_i['actions']
                            for j in verb_for:
                                values[j.item()]=1

                            for k in torch.nonzero(values):
                                cache_values[k.item()].append(values)
                                cache_keys[k.item()].append(image_features[ii,:])

        new_cache_keys = [[] for i in range(args.num_classes)]
        new_cache_values = [[] for i in range(args.num_classes)]
        for i in range(args.num_classes):
            topk_idx = torch.randperm(len(cache_values[i]))[:args.num_shot]
            for idx in topk_idx:
                new_cache_values[i].append(cache_values[i][idx])
                new_cache_keys[i].append(cache_keys[i][idx]) 
            if new_cache_values[i]==[]:
                for _ in range(args.num_shot):
                    new_cache_keys[i].append(torch.randn(2048).cuda())
                    value=torch.zeros(args.num_classes)
                    value[i]=1
                    new_cache_values[i].append(value)
            new_cache_keys[i]=torch.stack(new_cache_keys[i])
            new_cache_values[i]=torch.stack(new_cache_values[i])
        new_cache_keys=torch.cat(new_cache_keys)
        new_cache_values=torch.cat(new_cache_values)
        

        new_cache_keys /= new_cache_keys.norm(dim=-1, keepdim=True)
        new_cache_keys = new_cache_keys.permute(1, 0)

        
        
        if args.dataset=='hicodet':
            if args.zs:
                torch.save(new_cache_keys, args.cache_dir + f'/dino_keys_{args.zs_type}_{args.num_shot}.pt')
                torch.save(new_cache_values, args.cache_dir + f'/dino_values_{args.zs_type}_{args.num_shot}.pt')
            else:
                torch.save(new_cache_keys, args.cache_dir + '/dino_keys_2shots.pt')
                torch.save(new_cache_values, args.cache_dir + '/dino_values_2shots.pt')
        elif args.dataset=='vcoco':
            torch.save(new_cache_keys, args.cache_dir + '/vcoco_dino_keys_2shots.pt')
            torch.save(new_cache_values, args.cache_dir + '/vcoco_dino_values_2shots.pt')

    else:
        if args.dataset=='hicodet':
            if args.zs:
                new_cache_keys = torch.load(args.cache_dir + f'/dino_keys_{args.zs_type}_{args.num_shot}.pt')
                new_cache_values = torch.load(args.cache_dir + f'/dino_values_{args.zs_type}_{args.num_shot}.pt')
            else:
                new_cache_keys = torch.load(args.cache_dir + '/dino_keys_2shots.pt')
                new_cache_values = torch.load(args.cache_dir + '/dino_values_2shots.pt')
        elif args.dataset=='vcoco':
            new_cache_keys = torch.load(args.cache_dir + '/vcoco_dino_keys_2shots.pt')
            new_cache_values = torch.load(args.cache_dir + '/vcoco_dino_values_2shots.pt')
    

    return new_cache_keys, new_cache_values



