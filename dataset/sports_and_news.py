# VGGSound imports csv, whereas LRS imports pickle
import logging
import os
import random
import sys
from glob import glob
from pathlib import Path

from tqdm import tqdm
import torch
import json
import time

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
                 channel=None, # None indicates use all channels
                 distribution_type = 'uniform'):
        super().__init__()
        self.max_clip_len_sec = 5 # VGGSound has None, LRS has 11
        logger.info(f'During IO, the length of clips is limited to {self.max_clip_len_sec} sec')
        self.split = split
        self.use_random_offset = False
        if split == 'valid-random':
            self.use_random_offset = True
            self.split = 'valid'
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
            self.lengths_dict = json.load(open('/saltpool0/data/datasets/avsync/data/v5/evaluation_set_track_lengths.json'))
        elif split == 'train':
            data_csv = open(f'data/sports_and_news_{distribution_type}.train.csv').readlines() # TODO: switch back to train after running tests
            offset_path = f'data/sports_and_news_{distribution_type}.train.json' # TODO: switch back to train after running tests
            # skip_ids = [line.strip() for line in open(f'data/sports_and_news_{distribution_type}.{split}.skip_id_list.txt')]
            self.lengths_dict = json.load(open('/saltpool0/data/datasets/avsync/data/v5/train_set_track_lengths.json'))
        elif split in ('valid', 'valid-random'):
            data_csv = open(f'data/sports_and_news_{distribution_type}.test.csv').readlines()
            offset_path = f'data/sports_and_news_{distribution_type}.test.json'
            # skip_ids = [line.strip() for line in open(f'data/sports_and_news_{distribution_type}.{split}.skip_id_list.txt')]
            self.lengths_dict = json.load(open('/saltpool0/data/datasets/avsync/data/v5/test_set_track_lengths.json'))
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
                full_path = '/data3/scratch/videos_at_25fps-encode_script/' + video_folder + '/' + file_stem + '.mkv'
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

        self.vids_seen = 0
        self.load_times = []
        self.transforms_times = []
        self.too_long_paths = []
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
            # try:
            video_id, path, start = self.dataset[index] 
            if path in self.too_long_paths:
                return self[index - 1]

            video_lengths = self.lengths_dict['./'+path[15:]]
            if (start + self.max_clip_len_sec) > min(video_lengths):
                # print('Val set item', path, 'with fixed start time', start, 'runs over the end of the tracks at', min(video_lengths))
                # print('Reseting start time to max possible', int(min(video_lengths) - self.max_clip_len_sec - 2))
                start = int(min(video_lengths) - self.max_clip_len_sec - 2)
            elif start <= 10:
                # Nearly all failures are from the video having only 7-9s of frames when the start time is 10s for some reason
                start += 5
            start_load = time.time()
            rgb, audio, meta = get_video_and_audio(path, get_meta=True, max_clip_len_sec=self.max_clip_len_sec, start_sec=start)
            end_load = time.time()
            load_time = end_load - start_load
            if load_time > 5:
                print(path, 'took', load_time, 'seconds to load')
                self.too_long_paths.append(path)
            if rgb == None:
                # print('get_video_and_audio failed for path', path, 'max_clip_len_sec', self.max_clip_len_sec, 'and start_sec', start, 'so retrieving prior example')
                return self[index-1]
    
            # print('got response with shapes', rgb.shape, audio.shape, 'for path', path)

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

            start_transforms = time.time()
            # loading the fixed offsets. COMMENT THIS IF YOU DON'T HAVE A FILE YET
            if self.load_fixed_offsets_on_test and self.split in ['valid', 'test'] and not self.use_random_offset:
                item['targets']['offset_sec'] = self.vid2offset_params[video_id]['offset_sec']
                item['targets']['v_start_i_sec'] = self.vid2offset_params[video_id]['v_start_i_sec'] # This doesn't get used I think?
                if self.transforms is not None:
                    try:
                        item = self.transforms(item) # , skip_start_offset=True)
                    except:
                        print(f"Failed id: {video_id}\nOffset: {item['targets']['offset_sec']}, Start: {item['targets']['v_start_i_sec']}")
                        print('item is', item)
                        print('for path', path)
                        print('video shape is', item['video'].shape)
                        print('audio shape is', item['audio'].shape)
                        print('Applying transforms to get an error message, then exiting')
                        item = self.transforms(item)
                        exit(0)
                        # with open(f"sports_and_news_{self.distribution_type}.{self.split}.skip_id_list.txt", "a+") as fd:
                        #     if video_id not in set([line.strip for line in fd]):
                        #         fd.write(f"{video_id}\n")
                        return self[index-1]
                
            elif self.split in ('train', 'valid'):
                item['targets']['offset_sec'] = (random.random()*4)-2 # Random offset every time -> +/- 2 seconds
                # Old start time was (random.random()*296) + 2 to get range(2,298)
                # New restricts to (2,actual_min_track_length-2)
                # Also wait a second -- shouldn't we be loading the clip after this? Since we only load 5s? And this selects within the full 300s?
                item['targets']['v_start_i_sec'] = start # (random.random()*296) + 2 # random start time (2,298)
                if self.transforms is not None:
                    try:
                        item = self.transforms(item)
                    except:
                        print(f"Failed id: {video_id}\nOffset: {item['targets']['offset_sec']}, Start: {item['targets']['v_start_i_sec']}")
                        print('item is', item)
                        print('video shape is', item['video'].shape)
                        print('audio shape is', item['audio'].shape)
                        print('Applying transforms to get an error message, then exiting')
                        item = self.transforms(item)
                        exit(0)
                        return self[index-1] # Just retrain on previous data
            end_transforms = time.time()
            transforms_time = end_transforms - start_transforms

            # Decrease sizes
            item['video'] = item['video'].half()
            item['audio'] = item['audio'].half()

            # Dump times
            self.load_times.append(load_time)
            self.transforms_times.append(transforms_time)
            self.vids_seen += 1
            if self.vids_seen % 500 == 0:
                json.dump(self.load_times, open('load_times.json', 'w'))
                json.dump(self.transforms_times, open('transforms_times.json', 'w'))

            return item
            # except:
            # print('Failed on video', self.dataset[index][0], 'for a reason other than the transforms')
            # return self[index-1]

    def __len__(self):
        return len(self.dataset)


