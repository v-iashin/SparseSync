# VGGSound imports csv, whereas LRS imports pickle
import logging
import os
import random
import sys
from glob import glob
from pathlib import Path

from tqdm import tqdm
import torch

sys.path.insert(0, '.')
# was dataset.data_utils
from dataset.dataset_utils import (get_fixed_offsets, get_video_and_audio) 

logger = logging.getLogger(f'main.{__name__}')

class SportsAndNews(torch.utils.data.Dataset):
    
    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 splits_path='./data',
                 seed=1337,
                 load_fixed_offsets_on_test=True,
                 vis_load_backend=None, # This doesn't appear to be used anywhere, so can be empty
                 size_ratio=None,
                 channel='CNN',
                 distribution_type = 'normal'):
        super().__init__()
        self.max_clip_len_sec = 5 # VGGSound has None, LRS has 11
        logger.info(f'During IO, the length of clips is limited to {self.max_clip_len_sec} sec')
        self.split = split
        self.vids_dir = vids_dir
        self.transforms = transforms
        self.splits_path = splits_path
        self.seed = seed
        self.load_fixed_offsets_on_test = load_fixed_offsets_on_test
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio # used to do curriculum learning
        self.distribution_type = distribution_type

        if split == 'test': # TODO: Existing code only supports evaluation on the "test" split; using our val split for that, but will need to refactor if we want to do both using their code
            data_csv = open(f'data/sports_and_news_{distribution_type}.evaluation.csv').readlines()
            offset_path = f'data/sports_and_news_{distribution_type}.evaluation.json'
            # skip_ids = [line.strip() for line in open(f'data/sports_and_news_{distribution_type}.{split}.skip_id_list.txt')]
            # skip_ids = [line.strip() for line in open('data/sports_and_news_normal.evaluation.skip_id_list.txt')]
        elif split == 'train':
            data_csv = open(f'data/sports_and_news_{distribution_type}.train.csv').readlines()
            offset_path = f'data/sports_and_news_{distribution_type}.train.json'
            # skip_ids = [line.strip() for line in open(f'data/sports_and_news_{distribution_type}.{split}.skip_id_list.txt')]
        elif split == 'valid':
            data_csv = open(f'data/sports_and_news_{distribution_type}.test.csv').readlines()
            offset_path = f'data/sports_and_news_{distribution_type}.test.json'
            # skip_ids = [line.strip() for line in open(f'data/sports_and_news_{distribution_type}.{split}.skip_id_list.txt')]
        else:
            self.dataset = [0]
            return # Not set up yet!
        clip_paths = []

        # broken_vids = skip_ids #'bcdWbE64hDE_900_1200', 'alY7_M_ibR4_900_1200']
        broken_vids = [line.strip() for line in open(f"data/sports_and_news_{distribution_type}.skip_id_list.txt", "r")]

        for line in data_csv:
            skip = 'broken' in line
            if channel != None:
                if channel not in line:
                    skip = True
            for bv in broken_vids:
                # Parsing ids
                bv = bv[:-2] # Cut the _0 at the end
                if bv[-3] == '_': # 10-90
                    bv_vid = bv[:-3]
                    bv_offset = int(bv[-2:])
                elif bv[-4] == '_': # 90-990
                    bv_vid = bv[:-4]
                    bv_offset = int(bv[-3:])
                else:
                    print('PROBLEM: NOT SURE HOW TO PARSE VIDEO ID', bv)
                if bv_vid in line:
                    if int(float(line.split(',')[1])) == bv_offset:
                        skip = True
            if not skip:
                file_name_chunks = line.split(',')[0].split('_')
                assert(len(file_name_chunks) >= 5)
                file_stem = '_'.join(file_name_chunks[:-2])
                video_folder = '_'.join(file_name_chunks[:-4])
                full_path = '/saltpool0/data/datasets/avsync/data/v5/videos_at_25fps-encode_script/' + video_folder + '/' + file_stem + '.mkv'
                # full_path = '/saltpool0/data/datasets/avsync/data/v5/videos_at_25fps-encode_script/rOn7uGVVf1I/rOn7uGVVf1I_3000_3300.mkv'
                video_id = line.split(',')[0]
                tup = (video_id, full_path, float(line.split(',')[1]))
                clip_paths.append(tup)

        # We load fixed offsets for all splits
        logger.info(f'Using fixed offset for {split}')
        self.vid2offset_params = get_fixed_offsets(transforms, split, splits_path, 'sports_and_news', sports_and_news_path=offset_path) # VGG has 'vggsound', LRS has 'lrs3'

        self.dataset = clip_paths
        if size_ratio is not None and 0.0 < size_ratio < 1.0:
            cut_off = int(len(self.dataset) * size_ratio)
            self.dataset = self.dataset[:cut_off]

        logger.info(f'{split} has {len(self.dataset)} items')

        # self.check_lengths()

    def check_lengths(self):
        failed_vids = set()
        print(f"Checking {self.split} dataset:")
        for i in tqdm(range(len(self.dataset))):
            video_id, path, start = self.dataset[i]
            if video_id in failed_vids: continue
            rgb, _, _ = get_video_and_audio(path, get_meta=True, max_clip_len_sec=self.max_clip_len_sec, start_sec=start)
            if rgb == None:
                failed_vids.add(video_id)
        print(len(failed_vids), 'total videos failed')

        with open(f"sports_and_news_normal.{self.split}.skip_id_list.txt", "w") as fd:
            fd.write('\n'.join(list(failed_vids)))


    def __getitem__(self, index):
        video_id, path, start = self.dataset[index] 

        rgb, audio, meta = get_video_and_audio(path, get_meta=True, max_clip_len_sec=self.max_clip_len_sec, start_sec=start)
        

        # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
        item = {
                'video': rgb,
                'audio': audio,
                'meta': meta,
                'path': str(path), 
                'targets': {}, 
                'split': self.split,
                'start':start,
        }

        # loading the fixed offsets. COMMENT THIS IF YOU DON'T HAVE A FILE YET
        if self.load_fixed_offsets_on_test and self.split in ['valid', 'test']:
            item['targets']['offset_sec'] = self.vid2offset_params[video_id]['offset_sec']
            item['targets']['v_start_i_sec'] = self.vid2offset_params[video_id]['v_start_i_sec']
            if self.transforms is not None:
                try:
                    item = self.transforms(item) # , skip_start_offset=True)
                except:
                    print(f"Failed id: {video_id}\nOffset: {item['targets']['offset_sec']}, Start: {item['targets']['v_start_i_sec']}")
                    with open(f"sports_and_news_{self.distribution_type}.{self.split}.skip_id_list.txt", "a+") as fd:
                        if video_id not in set([line.strip for line in fd]):
                            fd.write(f"{video_id}\n")
                    return self[index-1]
                
        elif self.split == 'train':
            item['targets']['offset_sec'] = (random.random()*4)-2 # Random offset every time -> +/- 2 seconds
            item['targets']['v_start_i_sec'] = (random.random()*296) + 2 # random start time (2,298)
            if self.transforms is not None:
                try:
                    item = self.transforms(item)
                except:
                    print(f"Failed id: {video_id}\nOffset: {item['targets']['offset_sec']}, Start: {item['targets']['v_start_i_sec']}")
                    return self[index-1] # Just retrain on previous data

        return item

    def __len__(self):
        return len(self.dataset)


