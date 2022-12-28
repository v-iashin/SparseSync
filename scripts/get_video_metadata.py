#!/usr/bin/python3

import argparse
import cv2
import enzyme
import json
from os import listdir
import os.path
import sys

def main(root: str = "/saltpool0/data/datasets/avsync/data/v5/", filename: str = "all_video_metadata.json"):
    categories = {}

    # with open(os.path.join(root, "vid_links.txt")) as links_file:
    #     for line in links_file:
    #         link, category, _ = line.split(", ")
    #         video_id = link.replace("https://www.youtube.com/watch?v=", '')
    #         categories[video_id] = category[4:-4]

    # with open(os.path.join(root, "vid_links_v2.txt")) as links_file:
    #     for line in links_file:
    #         link, category, _ = line.split(", ")
    #         video_id = link.replace("https://www.youtube.com/watch?v=", '')
    #         categories[video_id] = category[4:-4]

    with open(os.path.join(root, "vid_links_v3.txt")) as links_file:
        for line in links_file:
            link, category, _ = line.split(", ")
            video_id = link.replace("https://www.youtube.com/watch?v=", '')
            categories[video_id] = category[4:-4]

    data_path = os.path.join(root, "videos")
    all_data = []
    for n, video_id in enumerate(categories):
        sys.stdout.write(f"\rProgress: {n / len(categories) * 100}%{' '*100}")
        video_data = {"video_id":video_id, "clips":[]}
        if not os.path.isdir(os.path.join(data_path, video_id)): continue
        for clip_file in listdir(os.path.join(data_path, video_id)):
            clip_split = clip_file.split('.')
            if len(clip_split) != 2 and clip_split[1] != '.mkv': continue
            full_path = os.path.join(data_path, video_id, clip_file)
            category = categories[video_id]
            
            # Read metadata from video
            try:
                with open(full_path, "rb") as video:
                    mkv = enzyme.MKV(video)
                    clip_data = {
                        "path":full_path, 
                        "duration":mkv.info.duration.total_seconds(), 
                        "framecount":int(cv2.VideoCapture(full_path).get(cv2.CAP_PROP_FRAME_COUNT)),
                        "samplerate":mkv.audio_tracks[0].sampling_frequency, 
                        "category":category
                    }
                    video_data["clips"].append(clip_data)
            except PermissionError:
                print(f"Could not open (permissions):  {full_path}")
                continue
            except enzyme.exceptions.MalformedMKVError:
                print(f"Could not open (mkv metadata): {full_path}")
                continue
        all_data.append(video_data)
    print(f"\nDone! Saving to {filename}")
    with open(filename, "w") as outfile:
        json.dump(all_data, outfile, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse through directories and save metadata for videos and clips as a json file.')
    parser.add_argument('data_path', metavar='data_path',
                        help='directory for data to parse')
    parser.add_argument('filename', metavar='output_name.json', type=str,
                        help='output filename for video metadata',
                        default='all_video_metadata.json', nargs='?')
    args = parser.parse_args()
    main(args.data_path, args.filename)