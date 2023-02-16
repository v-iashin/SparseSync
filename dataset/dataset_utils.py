import csv
import os
from pathlib import Path
from time import sleep
import torchaudio
import torchvision
from glob import glob
import shutil
import sys
sys.path.append('..')
from utils.utils import get_fixed_off_fname
import json


def get_fixed_offsets(transforms, split, splits_path, dataset_name, sports_and_news_path=None):
    '''dataset_name: `vggsound` or `lrs3` or `sports_and_news`'''
    # TODO: Add sports and news dataset, load the vid2offset_params dict directly from json, instead of doing this building thing
    if dataset_name == "sports_and_news":
        with open(sports_and_news_path, 'r') as file_data:
            return json.load(file_data)
    
    vid2offset_params = {}
    fixed_offset_fname = get_fixed_off_fname(transforms, split)
    fixed_offset_path = os.path.join(splits_path, f'fixed_offsets_{dataset_name}', fixed_offset_fname)
    fixed_offset_paths = sorted(glob(fixed_offset_path.replace(split, '*')))
    assert len(fixed_offset_paths) > 0, f'Perhaps: {fixed_offset_path} does not exist. Make fixed offsets'

    for fix_off_path in fixed_offset_paths:
        reader = csv.reader(open(fix_off_path))
        # skipping the header
        next(reader)
        for v, o, s in reader:
            assert v not in vid2offset_params, 'otherwise, offsets from other splits will override each other'
            vid2offset_params[v] = {'offset_sec': float(o), 'v_start_i_sec': float(s)}
    return vid2offset_params


def maybe_cache_file(path: os.PathLike):
    '''Motivation: if every job reads from a shared disk it`ll get very slow, consider an image can
    be 2MB, then with batch size 32, 16 workers in dataloader you`re already requesting 1GB!! -
    imagine this for all users and all jobs simultaneously.'''
    # checking if we are on cluster, not on a local machine
    if 'LOCAL_SCRATCH' in os.environ:
        cache_dir = os.environ.get('LOCAL_SCRATCH')
        # a bit ugly but we need not just fname to be appended to `cache_dir` but parent folders,
        # otherwise the same fnames in multiple folders will create a bug (the same input for multiple paths)
        cache_path = os.path.join(cache_dir, Path(path).relative_to('/'))
        if not os.path.exists(cache_path):
            os.makedirs(Path(cache_path).parent, exist_ok=True)
            shutil.copyfile(path, cache_path)
        return cache_path
    else:
        return path


def get_video_and_audio(path, get_meta=False, max_clip_len_sec=None, start_sec=None):
    path = maybe_cache_file(path)
    # try-except was meant to solve issue when `maybe_cache_file` copies a file but another worker tries to
    # load it because it thinks that the file exists. However, I am not sure if it works :/.
    # Feel free to refactor it.
    try:
        if start_sec != None:
            end_sec = start_sec+max_clip_len_sec+2
            rgb, audio, meta = torchvision.io.read_video(path, pts_unit='sec', start_pts=start_sec-2, end_pts=end_sec)
        else:
            rgb, audio, meta = torchvision.io.read_video(path, pts_unit='sec', end_pts=max_clip_len_sec)
    except KeyError:
        print(f'Problem at {path}. Trying to wait and load again...')
        sleep(5)
        if start_sec != None:
            end_sec = start_sec+max_clip_len_sec+2
        rgb, audio, meta = torchvision.io.read_video(path, pts_unit='sec', start_pts=start_sec-2, end_pts=end_sec)
        meta['video_fps']
    # (T, 3, H, W) [0, 255, uint8] <- (T, H, W, 3)
    rgb = rgb.permute(0, 3, 1, 2)
    # (Ta) <- (Ca, Ta)
    audio = audio.mean(dim=0)
    if audio.shape[0] < 144000:
        # print('Audio shape only', audio.shape[0], 'for clip at', start_sec, 'in video', path)
        return None, None, None # To flag for debugging
    # FIXME: this is legacy format of `meta` as it used to be loaded by VideoReader.
    if meta == {}:
        print('video path of failure', path)
        print('extracted rgb and audio shapes', rgb.shape, audio.shape)
        exit(0)
    meta = {
        'video': {'fps': [meta['video_fps']]},
        'audio': {'framerate': [meta['audio_fps']]},
    }
    return rgb, audio, meta


def get_audio_stream(path, get_meta=False):
    '''Used only in feature extractor training'''
    path = str(Path(path).with_suffix('.wav'))
    path = maybe_cache_file(path)
    waveform, _ = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if get_meta:
        info = torchaudio.info(path)
        duration = info.num_frames / info.sample_rate
        meta = {'audio': {'duration': [duration], 'framerate': [info.sample_rate]}}
        return waveform, meta
    else:
        return waveform
