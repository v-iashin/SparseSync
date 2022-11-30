# VGGSound imports csv, whereas LRS imports pickle
import logging
import os
import random
import sys
from glob import glob
from pathlib import Path

import torch

sys.path.insert(0, '.')
from dataset.data_utils import (get_fixed_offsets, get_video_and_audio) 

logger = logging.getLogger(f'main.{__name__}')

class SportsAndNews(torch.utils.data.Dataset):
    
    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 splits_path='./data',
                 # VGGSound has meta_path='./data/vggsound.csv', LRS doesn't
                 seed=1337,
                 load_fixed_offsets_on_text=True,
                 vis_load_backend=None, # This doesn't appear to be used anywhere, so can be empty
                 size_ratio=None):
        super().__init__()
        self.max_clip_len_sec = 10 # VGGSound has None, LRS has 11
        logger.info(f'During IO, the length of clips is limited to {self.max_clip_len_sec} sec')
        self.split = split
        self.vids_dir = vids_dir
        self.transforms = transforms
        self.splits_path = splits_path
        self.meta_path = meta_path
        self.seed = seed
        self.load_fixed_offsets_on_test = load_fixed_offsets_on_test
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio # used to do curriculum learning

        """
        TODO: Signficant divergence here, goal is to create a "clip_paths" object which contains sorted paths to clip .mp4 files
        For our purposes, we want to load not just the path but also the start time!

        For LRS:
            vid_folder = Path(vids_dir) / 'pretrain'

            split_clip_ids_path = os.path.join(splits_path, f'lrs_{split}.txt')
            if not os.path.exists(split_clip_ids_path): # NOTE: We should never be in this condition, since our splits were already computed by Puyuan+James!
                clip_paths = sorted(vid_folder.rglob('*/*.mp4'))
                # filter "bad" examples
                clip_paths = self.filter_bad_examples(clip_paths)
                self.make_split_files(clip_paths) # NOTE: We therefore probably don't need to implement this function
                
            # read the ids from a split
            split_clip_ids = sorted(open(split_clip_ids_path).read().splitlines()) 
            # NOTE: the file at {splits_path}/lrs_{split}.txt must contain one clip ID in the split per line and nothing else

            # make paths from the ids
            clip_paths = [os.path.join(vids_dir, v + '.mp4') for v in split_clip_ids]
            # NOTE: the videos must be stored as {vids_dir}/{clip ID}.mp4
        
        For VGGSound:
            vggsound_meta = list(csv.reader(open(meta_path), quotechar='"')) # NOTE: " is already the default quotechar, so probably unnecessary to specify
            # NOTE: metadata csv format inferred from code (no header line): video_ID, start_time_in_s, label

            # filter "bad" examples
            if to_filter_bad_examples: # NOTE: why do we only want to do this sometimes?
                vggsound_meta = self.filter_bad_examples(vggsound_meta)

            unique_classes = sorted(list(set(row[2] for row in vggsound_meta))) # NOTE: 3rd col in csv is class
            self.label2target = {label: target for target, label in enumerate(unique_classes)} # NOTE: assigns an index to each class
            self.target2label = {target: label for label, target in self.label2target.items()} # NOTE: l2t and t2l are inverses (l2t[t2l[x]] = x and t2l[l2t[x]] = x)
            self.video2target = {row[0]: self.label2target[row[2]] for row in vggsound_meta} # NOTE: 1st col in csv is video ID

            split_clip_ids_path = os.path.join(splits_path, f'vggsound_{split}.txt')
            if not os.path.exists(split_clip_ids_path): # NOTE: We should never be in this condition, since our splits were already computed by Puyuan+James!
                self.make_split_files() # We therefore probably don't need to implement this function
            
            # the ugly string converts ['AdfsGsfII2yQ', '1'] into `AdfsGsfII2yQ_1000_11000`
            meta_available = set([f'{r[0]}_{int(r[1]*1000}_{(int(r[1])+10)*1000}' for r in vggsound_meta]) # NOTE: 2nd col in csv is 10s clip's start time (in s)
            within_split = set(open(split_clip_ids_path).read().splitlines())
            # NOTE: the file at {splits_path}/vggsound_{split}.txt must contain one clip ID in the split per line and nothing else, in format ClipID_Start_Stop
            clip_paths = [os.path.join(vids_dir, v + '.mp4') for v in meta_available.intersection(within_split)]
            clip_paths = sorted(clip_paths)

        NOTE: At least for VGGSound, this dataloader will expect input clips to already be 10s long! LRS version may trim
        """

        # We load fixed offsets for all splits
        logger.info(f'Using fixed offset for {split}')
        self.vid2offset_params = get_fixed_offets(transforms, split, splits_path, 'sports+news') # VGG has 'vggsound', LRS has 'lrs3'

        self.dataset = clip_paths
        if size_ratio is not None and 0.0 < size_ratio < 1.0:
            cut_off = int(len(self.dataset) * size_ratio)
            self.dataset = self.dataset[:cut_off]

        logger.info(f'{split} has {len(self.datasest)} items')

    def __getitem__(self, index):
        path, start = self.dataset[index] 

        rgb, audio, meta = get_video_and_audio(path, get_meta=True, max_clip_len_sec=self.max_clip_len_sec, start_sec=start)
        
        # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
        item = {
                'video': rgb,
                'audio': audio,
                'meta': meta,
                'path': str(path), 
                'targets': {}, 
                'split': self.split,
        }

        # loading the fixed offsets. COMMENT THIS IF YOU DON'T HAVE A FILE YET
        if self.load_fixed_offsets_on_test and self.split in ['valid', 'test']:
            # TODO: LRS only has unique_id = path.replace(f'{self.vids_dir}/','').replace(self.vids_dir, '').replace('.mp4','')
            item['targets']['offset_sec'] = self.vid2offset_params[Path(path).stem]['offset_sec'] # VGG has Path(path).stem, LRS has unique_id
            item['targets']['v_start_i_sec'] = self.vid2offset_params[Path(path).stem]['v_start_i_sec'] # VGG has Path(path).stem, LRS has unique_id

            if self.transforms is not None:
                item = self.transforms(item)
        
        return item

    def __len__(self):
        return len(self.dataset)


