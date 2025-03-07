import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import faiss
import json
import shutil
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
import torchaudio
import traceback
torchaudio.set_audio_backend("soundfile")



from util import \
create_fp_dir, load_config, \
query_len_from_seconds, seconds_from_query_len, \
load_augmentation_index
from modules.data import Sample100Dataset
# from encoder.graph_encoder import GraphEncoder
from encoder.dgl.graph_encoder import GraphEncoderDGL
from encoder.resnet_ibn import ResNetIBN
from simclr.simclr import SimCLR   
from modules.transformations import GPUTransformSampleID
from eval import get_index, load_memmap_data, eval_faiss
from eval_map import eval_faiss_with_map


# Directories
root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")

parser = argparse.ArgumentParser(description='Neuralfp Testing')
parser.add_argument('--config', default='config/grafp.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--test_config', default='config/test_config.yaml', type=str)
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing testing. ')
parser.add_argument('--test_dir', default='data/fma_medium.json', type=str,
                    help='path to test data')
parser.add_argument('--noise_idx', default=None, type=str)
parser.add_argument('--noise_split', default='all', type=str,
                    help='Noise index file split to use for testing (all, test)')
parser.add_argument('--fp_dir', default='fingerprints', type=str)
parser.add_argument('--query_lens', default=None, type=str)
parser.add_argument('--encoder', default='grafp', type=str)
parser.add_argument('--n_dummy_db', default=None, type=int)
# parser.add_argument('--n_query_db', default=350, type=int)
parser.add_argument('--small_test', action='store_true', default=False)
parser.add_argument('--text', default='test', type=str)
# parser.add_argument('--test_snr', default=None, type=int)
parser.add_argument('--recompute', action='store_true', default=False)
parser.add_argument('--map', action='store_true', default=False)
# parser.add_argument('--hit_rate', action='store_true', default=True)
parser.add_argument('--k', default=3, type=int)
parser.add_argument('--test_ids', default='1000', type=str)

parser.add_argument('--stem_eval', action='store_true', default=False, help='Perform stem-wise evaluation')

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')



def create_table(hit_rates, overlap, dur, test_seq_len=[1,3,5,9,11,19], text="test"):
    table = f'''<table>
    <tr>
    <th colspan="5"{text}</th>
    <th>Query Length</th>
    <th>Top-1 Exact</th>
    <th>Top-3 Exact</th>
    <th>Top-10 Exact</th>
    </tr>
    '''
    for idx, q_len in enumerate(test_seq_len):
        table += f'''
        <tr>
        <td>{seconds_from_query_len(q_len, overlap, dur)}</td>
        <td>{hit_rates[0][idx]}</td>
        <td>{hit_rates[1][idx]}</td>
        <td>{hit_rates[2][idx]}</td>
        </tr>
        '''
    table += '</table>'
    return table

def create_query_db(dataloader, augment, model, output_root_dir, fname='query_db', verbose=False, max_size=128):
    fp = []
    lookup_table = []  # Initialize lookup table
    print("=> Creating query fingerprints...")
    for idx, (nm,audio) in enumerate(dataloader):
        nm = nm[0] # Extract filename from list
        audio = audio.to(device)
        x_i, _ = augment(audio, None)
        x_list = torch.split(x_i, max_size, dim=0)
        fp_size = 0
        for x in x_list:
            with torch.no_grad():
                _, _, z_i, _= model(x.to(device),x.to(device))  

            fp.append(z_i.detach().cpu().numpy())
            fp_size += z_i.shape[0]

        # Append song number to lookup table for each segment in the batch
        if fname == 'query_db':
            lookup_table.extend([nm + "_" + str(idx)] * x_i.shape[0])
            log_mod = 100
        else:
            lookup_table.extend([nm] * x_i.shape[0])
            log_mod = 20

        if verbose and idx % log_mod == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: [{fp_size,z_i.shape[1]}]")

    fp = np.concatenate(fp)
    arr_shape = (len(fp), z_i.shape[-1])

    arr = np.memmap(f'{output_root_dir}/{fname}.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr[:] = fp[:]
    arr.flush(); del(arr)   #Close memmap

    np.save(f'{output_root_dir}/{fname}_shape.npy', arr_shape)

    # Save lookup table
    json.dump(lookup_table, open(f'{output_root_dir}/{fname}_lookup.json', 'w'))

def create_ref_db(dataloader, augment, model, output_root_dir, fname='ref_db', verbose=False, max_size=128):
    fp = []
    lookup_table = []  # Initialize lookup table
    print("=> Creating reference fingerprints...")
    for idx, (nm,audio) in enumerate(dataloader):
        nm = nm[0] # Extract filename from list
        audio = audio.to(device)
        x_i, _ = augment(audio, None)
        x_list = torch.split(x_i, max_size, dim=0)
        fp_size = 0
        for x in x_list:
            with torch.no_grad():
                _, _, z_i, _= model(x.to(device),x.to(device))  

            fp.append(z_i.detach().cpu().numpy())
            fp_size += z_i.shape[0]

        # Append song number to lookup table for each segment in the batch
        lookup_table.extend([nm] * x_i.shape[0])

        if verbose and idx % 20 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {fp_size}")

    fp = np.concatenate(fp)
    arr_shape = (len(fp), z_i.shape[-1])

    arr = np.memmap(f'{output_root_dir}/{fname}.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr[:] = fp[:]
    arr.flush(); del(arr)   #Close memmap

    np.save(f'{output_root_dir}/{fname}_shape.npy', arr_shape)

    # Save lookup table
    json.dump(lookup_table, open(f'{output_root_dir}/{fname}_lookup.json', 'w'))


def create_dummy_db(dataloader, augment, model, output_root_dir, fname='dummy_db', verbose=False, max_size=128):
    fp = []
    print("=> Creating dummy fingerprints...")
    for idx, (nm,audio) in enumerate(dataloader):
        audio = audio.to(device)
        x_i, _ = augment(audio, None)
        # print(f"Shape of x_i (dummy): {x_i.shape}")
        x_list = torch.split(x_i, max_size, dim=0)
        # print(f"Number of splits: {len(x_list)}")
        fp_size = 0
        for x in x_list:
            try:
                with torch.no_grad():
                    _, _, z_i, _= model(x.to(device),x.to(device)) 

            except Exception as e:
                print(f"Error in model forward pass in file {nm}")
                print(f"Shape of x_i (dummy): {x.shape}")
                print(f"x_i mean: {x.mean()}, x_i std: {x.std()}")
                print(f"All x shapes in list: {[x_.shape for x_ in x_list]}")
                print(f"Index of data {idx}")
                # print(traceback.format_exc())
                # print(f"Shape of z_i (dummy): {z_i.shape}")
                continue 

            fp.append(z_i.detach().cpu().numpy())
            fp_size += z_i.shape[0]
        
        if verbose and idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {fp_size}")
        # fp = torch.cat(fp)
    
    fp = np.concatenate(fp)
    arr_shape = (len(fp), z_i.shape[-1])

    arr = np.memmap(f'{output_root_dir}/{fname}.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr[:] = fp[:]
    arr.flush(); del(arr)   #Close memmap

    np.save(f'{output_root_dir}/{fname}_shape.npy', arr_shape)


def evaluate_stems(cfg, model, test_augment, output_root_dir, annot_path, test_dir, index_type='ivfpq'):
    """
    Evaluate each stem type separately and combine results
    """
    stem_types = ['bass', 'drums', 'vocals', 'other', 'mix']
    stem_results = {}
    
    
    for stem in stem_types:
        print(f"\n=== Evaluating {stem.upper()} stems ===")
        
        # Create datasets for this stem
        query_dataset = Sample100Dataset(cfg, path=test_dir, annot_path=annot_path, 
                                         mode="query", stem=stem)
        ref_dataset = Sample100Dataset(cfg, path=test_dir, annot_path=annot_path, 
                                       mode="ref", stem=stem)
        
        # Create fingerprint directory for this stem
        stem_dir = os.path.join(output_root_dir, f"stem_{stem}")
        os.makedirs(stem_dir, exist_ok=True)
        
        # Process reference audio files individually to avoid batch size issues
        print("Creating reference fingerprints...")
        ref_lookup = []
        ref_fingerprints = []
        
        for idx in range(len(ref_dataset)):
            try:
                fname, audio = ref_dataset[idx]
                audio = audio.unsqueeze(0).to(device)  # Add batch dimension
                
                x_i, _ = test_augment(audio, None)
                
                # Process in small chunks to avoid memory issues
                max_size = 32  # Smaller batch size
                x_list = torch.split(x_i, max_size, dim=0)
                
                for x in x_list:
                    with torch.no_grad():
                        try:
                            _, _, z_i, _ = model(x.to(device), x.to(device))
                            ref_fingerprints.append(z_i.detach().cpu().numpy())
                            ref_lookup.extend([fname] * z_i.shape[0])
                            
                            if idx % 20 == 0:
                                print(f"Processed ref {idx}/{len(ref_dataset)}, shape: {z_i.shape}")
                        except Exception as e:
                            print(f"Error processing reference {fname}: {e}")
                            continue
            except Exception as e:
                print(f"Error loading reference at index {idx}: {e}")
                continue
        
        # Process query audio files
        print("Creating query fingerprints...")
        query_lookup = []
        query_fingerprints = []
        
        for idx in range(len(query_dataset)):
            try:
                fname, audio = query_dataset[idx]
                audio = audio.unsqueeze(0).to(device)  # Add batch dimension
                
                x_i, _ = test_augment(audio, None)
                
                # Process in small chunks
                max_size = 32
                x_list = torch.split(x_i, max_size, dim=0)
                
                for x in x_list:
                    with torch.no_grad():
                        try:
                            _, _, z_i, _ = model(x.to(device), x.to(device))
                            query_fingerprints.append(z_i.detach().cpu().numpy())
                            query_lookup.extend([fname + "_" + str(idx)] * z_i.shape[0])
                            
                            if idx % 20 == 0:
                                print(f"Processed query {idx}/{len(query_dataset)}, shape: {z_i.shape}")
                        except Exception as e:
                            print(f"Error processing query {fname}: {e}")
                            continue
            except Exception as e:
                print(f"Error loading query at index {idx}: {e}")
                continue
        
        # Save fingerprints and lookup tables
        if ref_fingerprints and query_fingerprints:
            # Concatenate fingerprints
            ref_fp = np.concatenate(ref_fingerprints)
            query_fp = np.concatenate(query_fingerprints)
            
            # Save reference fingerprints
            ref_shape = (len(ref_fp), ref_fp.shape[1])
            ref_arr = np.memmap(f'{stem_dir}/ref_db.mm',
                              dtype='float32',
                              mode='w+',
                              shape=ref_shape)
            ref_arr[:] = ref_fp[:]
            ref_arr.flush()
            del ref_arr
            
            np.save(f'{stem_dir}/ref_db_shape.npy', ref_shape)
            json.dump(ref_lookup, open(f'{stem_dir}/ref_db_lookup.json', 'w'))
            
            # Save query fingerprints
            query_shape = (len(query_fp), query_fp.shape[1])
            query_arr = np.memmap(f'{stem_dir}/query_db.mm',
                                dtype='float32',
                                mode='w+',
                                shape=query_shape)
            query_arr[:] = query_fp[:]
            query_arr.flush()
            del query_arr
            
            np.save(f'{stem_dir}/query_db_shape.npy', query_shape)
            json.dump(query_lookup, open(f'{stem_dir}/query_db_lookup.json', 'w'))
            
            # Run evaluation
            try:
                hit_rates, histograms = eval_faiss(emb_dir=stem_dir, emb_dummy_dir=output_root_dir, index_type=index_type, 
                                                 nogpu=True, return_histograms=True)
                
                stem_results[stem] = {
                    'hit_rates': hit_rates,
                    'histograms': histograms
                }
                
                print(f"--- {stem.upper()} Results ---")
                print(f'Top-1 exact hit rate = {hit_rates[0]}')
                print(f'Top-3 exact hit rate = {hit_rates[1]}')
                print(f'Top-10 exact hit rate = {hit_rates[2]}')
            except Exception as e:
                print(f"Error evaluating {stem} stems: {e}")
        else:
            print(f"No valid fingerprints for {stem} stems")
    
    return stem_results


# def evaluate_stems(cfg, model, test_augment, output_root_dir, annot_path, test_dir, index_type='ivfpq'):
    # """
    # Evaluate each stem type separately and combine results
    # """
    # stem_types = ['bass', 'drums', 'vocals', 'other', 'mix']
    # stem_results = {}
    
    # for stem in stem_types:
    #     print(f"\n=== Evaluating {stem.upper()} stems ===")
        
    #     # Create datasets for this stem
    #     query_dataset = Sample100Dataset(cfg, path=test_dir, annot_path=annot_path, 
    #                                      mode="query", stem=stem)
    #     ref_dataset = Sample100Dataset(cfg, path=test_dir, annot_path=annot_path, 
    #                                     mode="ref", stem=stem)
        
    #     # Create DataLoader instances
    #     query_loader = DataLoader(query_dataset, batch_size=1, 
    #                              shuffle=False, num_workers=4, 
    #                              pin_memory=True, drop_last=False)
        
    #     ref_loader = DataLoader(ref_dataset, batch_size=1, 
    #                            shuffle=False, num_workers=4, 
    #                            pin_memory=True, drop_last=False)
        
    #     # Create fingerprints directory for this stem
    #     stem_dir = os.path.join(output_root_dir, f"stem_{stem}")
    #     os.makedirs(stem_dir, exist_ok=True)
        
    #     # Create reference and query fingerprints
    #     create_ref_db(ref_loader, augment=test_augment,
    #                  model=model, output_root_dir=stem_dir, verbose=True)
        
    #     create_query_db(query_loader, augment=test_augment,
    #                    model=model, output_root_dir=stem_dir, verbose=True)
        
    #     # Evaluate
    #     hit_rates = eval_faiss(emb_dir=stem_dir, index_type=index_type, nogpu=True)
        
    #     stem_results[stem] = hit_rates
        
    #     print(f"--- {stem.upper()} Results ---")
    #     print(f'Top-1 exact hit rate = {hit_rates[0]}')
    #     print(f'Top-3 exact hit rate = {hit_rates[1]}')
    #     print(f'Top-10 exact hit rate = {hit_rates[2]}')
    
    # return stem_results

def evaluate_rankings(rankings, annotations):
    """Evaluate combined rankings against ground truth"""
    top1_exact = 0
    top3_exact = 0
    top10_exact = 0
    
    # Create ground truth mapping
    gt_dict = {}
    for ann in annotations:
        query = ann['query_file']
        ref = ann['ref_file']
        if query not in gt_dict:
            gt_dict[query] = []
        gt_dict[query].append(ref)
    
    # Evaluate rankings
    for query_id, refs in rankings.items():
        # Get ground truth for this query
        gt_refs = gt_dict.get(query_id, [])
        
        # Check for matches
        if refs and any(r in gt_refs for r in refs[:1]):
            top1_exact += 1
        if refs and any(r in gt_refs for r in refs[:3]):
            top3_exact += 1
        if refs and any(r in gt_refs for r in refs[:10]):
            top10_exact += 1
    
    # Calculate hit rates
    total = len(rankings)
    hit_rates = [
        100.0 * top1_exact / total,
        100.0 * top3_exact / total,
        100.0 * top10_exact / total
    ]
    
    return hit_rates

def combine_stem_results(stem_results, annotations):
    """
    Combine results from all stems to generate a final ranking
    Weighting can be adjusted based on your preference
    """
    # Create a combined scoring system
    combined_scores = defaultdict(float)
    
    # Define weights for each stem (can be adjusted)
    weights = {
        'bass': 0.2, 
        'drums': 0.2, 
        'vocals': 0.2, 
        'other': 0.2, 
        'mix': 0.2
    }
    # weights = {
    #     'bass': 0.25, 
    #     'drums': 0.25, 
    #     'vocals': 0.25, 
    #     'other': 0.15, 
    #     'mix': 0.1
    # }
    
    for stem, results in stem_results.items():
        # Get the lookup tables for query and reference
        query_lookup = json.load(open(f'{output_root_dir}/stem_{stem}/query_db_lookup.json', 'r'))
        ref_lookup = json.load(open(f'{output_root_dir}/stem_{stem}/ref_db_lookup.json', 'r'))
        
        # Process each query
        for q_idx, q_id in enumerate(query_lookup):
            # Get the histogram for this query
            query_hist = results['histograms'][q_idx]
            
            # Apply weight to this stem's scores
            for ref_id, score in query_hist.items():
                combined_scores[(q_id, ref_id)] += score * weights[stem]
    
    # Create final rankings
    final_rankings = {}
    for (q_id, _), _ in combined_scores.items():
        if q_id not in final_rankings:
            final_rankings[q_id] = []
            
        # Get all refs for this query
        refs = [(ref_id, combined_scores[(q_id, ref_id)]) for q_id, ref_id in combined_scores.keys() if q_id == q_id]
        
        # Sort by score
        refs.sort(key=lambda x: x[1], reverse=True)
        
        # Store rankings
        final_rankings[q_id] = [ref_id for ref_id, _ in refs]
    
    # Evaluate final rankings
    hit_rates = evaluate_rankings(final_rankings, annotations)
    
    return hit_rates, final_rankings



def main():

    args = parser.parse_args()
    cfg = load_config(args.config)
    test_cfg = load_config(args.test_config)
    ir_dir = cfg['ir_dir']
    noise_dir = cfg['noise_dir']
    annot_path = cfg['annot_path']
    # args.recompute = False
    # assert args.recompute is False
    assert args.small_test is False
    # Hyperparameters
    random_seed = 42
    shuffle_dataset =True


    print("Creating new model...")
    if args.encoder == 'grafp':
        model = SimCLR(cfg, encoder=GraphEncoderDGL(cfg=cfg, in_channels=cfg['n_filters'], k=args.k))
    elif args.encoder == 'resnet-ibn':
        model = SimCLR(cfg, encoder=ResNetIBN())
    else:
        raise ValueError(f"Invalid encoder: {args.encoder}")
    
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # model = DataParallel(model).to(device)
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(device)

    print("Creating dataloaders ...")

    # Augmentation for testing with specific noise subsets
    if args.noise_idx is not None:
        noise_test_idx = load_augmentation_index(noise_dir, json_path=args.noise_idx, splits=0.8)[args.noise_split]
    else:
        noise_test_idx = load_augmentation_index(noise_dir, splits=0.8)["test"]
    ir_test_idx = load_augmentation_index(ir_dir, splits=0.8)["test"]
    test_augment = GPUTransformSampleID(cfg=cfg, ir_dir=ir_test_idx, 
                                        noise_dir=noise_test_idx, 
                                        train=False).to(device)

    query_dataset = Sample100Dataset(cfg, path=args.test_dir, annot_path=annot_path, mode="query")
    query_full_dataset = Sample100Dataset(cfg, path=args.test_dir, annot_path=annot_path, mode="query_full")
    ref_dataset = Sample100Dataset(cfg, path=args.test_dir, annot_path=annot_path, mode="ref")
    dummy_path = 'data/sample_100.json'     # Required for dummy db
    dummy_dataset = Sample100Dataset(cfg, path=dummy_path, annot_path=annot_path, mode="dummy")

    # Create DataLoader instances for each dataset
    dummy_db_loader = DataLoader(dummy_dataset, batch_size=1, 
                                shuffle=False, num_workers=4, 
                                pin_memory=True, drop_last=False)

    query_db_loader = DataLoader(query_dataset, batch_size=1, 
                                shuffle=False, num_workers=4, 
                                pin_memory=True, drop_last=False)
    
    query_full_db_loader = DataLoader(query_full_dataset, batch_size=1, 
                                shuffle=False, num_workers=4, 
                                pin_memory=True, drop_last=False)

    ref_db_loader = DataLoader(ref_dataset, batch_size=1, 
                            shuffle=False, num_workers=4, 
                            pin_memory=True, drop_last=False)


    if args.small_test:
        index_type = 'l2'
    else:
        index_type = 'ivfpq'

    if args.query_lens is not None:
        args.query_lens = [float(q) for q in args.query_lens.split(',')]
        test_seq_len = [query_len_from_seconds(q, cfg['overlap'], dur=cfg['dur'])
                        for q in args.query_lens]
        

    for ckp_name, epochs in test_cfg.items():
        if not type(epochs) == list:
            epochs = [epochs]   # Hack to handle case where only best ckp is to be tested
        writer = SummaryWriter(f'runs/{ckp_name}')

        for epoch in epochs:
            ckp = os.path.join(model_folder, f'model_{ckp_name}_{str(epoch)}.pth')
            if os.path.isfile(ckp):
                print("=> loading checkpoint '{}'".format(ckp))
                checkpoint = torch.load(ckp)
                # Check for DataParallel
                if 'module' in list(checkpoint['state_dict'].keys())[0] and torch.cuda.device_count() == 1:
                    checkpoint['state_dict'] = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(ckp))
                continue
            
            fp_dir = create_fp_dir(resume=ckp, train=False)
            if args.recompute or os.path.isfile(f'{fp_dir}/dummy_db.mm') is False:
                print("=> Computing dummy fingerprints...")
                create_dummy_db(dummy_db_loader, augment=test_augment,
                                model=model, output_root_dir=fp_dir, verbose=True)
            else:
                print("=> Skipping dummy db creation...")

            create_ref_db(ref_db_loader, augment=test_augment,
                            model=model, output_root_dir=fp_dir, verbose=True)
            
            create_query_db(query_db_loader, augment=test_augment,
                            model=model, output_root_dir=fp_dir, verbose=True)
            
            if args.map:
                create_query_db(query_full_db_loader, augment=test_augment,
                                model=model, output_root_dir=fp_dir, fname='query_full_db', verbose=True)
            
            
            text = f'{args.text}_{str(epoch)}'
            label = epoch if type(epoch) == int else 0


            if args.query_lens is not None:
                hit_rates = eval_faiss(emb_dir=fp_dir,
                                    test_seq_len=test_seq_len, 
                                    index_type=index_type,
                                    nogpu=True) 


                writer.add_text("table", 
                                create_table(hit_rates, 
                                            cfg['overlap'], cfg['dur'],
                                            test_seq_len, text=text), 
                                label)
  
            else:
                hit_rates = eval_faiss(emb_dir=fp_dir, 
                                    index_type=index_type,
                                    nogpu=True)
                
                writer.add_text("table", 
                                create_table(hit_rates, 
                                            cfg['overlap'], cfg['dur'], text=text), 
                                label)
                
            print("-------Test hit-rates-------")
            # Create table
            print(f'Top-1 exact hit rate = {hit_rates[0]}')
            print(f'Top-3 exact hit rate = {hit_rates[1]}')
            print(f'Top-10 exact hit rate = {hit_rates[2]}')
            
            if args.map:
                map_score, k_map = eval_faiss_with_map(emb_dir=fp_dir, 
                                        index_type='ivfpq',
                                        test_seq_len=test_seq_len,
                                        nogpu=True)




                print("-------Test MAP-------")
                print(f'Mean Average Precision (MAP@{k_map}): {map_score:.4f}')

            if args.stem_eval:
                print("\n=== Performing Stem-Based Evaluation ===")
                stem_results = evaluate_stems(cfg, model, test_augment, fp_dir, annot_path, args.test_dir)
                
                # Load annotations for combined evaluation
                with open(annot_path, 'r') as fp:
                    annotations = json.load(fp)
                
                hit_rates, final_rankings = combine_stem_results(stem_results, annotations)
                
                print("\n=== Combined Stem Results ===")
                print(f'Top-1 exact hit rate = {hit_rates[0]}')
                print(f'Top-3 exact hit rate = {hit_rates[1]}')
                print(f'Top-10 exact hit rate = {hit_rates[2]}')
                
                # Save combined results
                np.save(f'{fp_dir}/combined_hit_rates.npy', hit_rates)
                with open(f'{fp_dir}/final_rankings.json', 'w') as f:
                    json.dump(final_rankings, f)


if __name__ == '__main__':
    main()