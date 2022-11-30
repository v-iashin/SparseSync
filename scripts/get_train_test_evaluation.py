#!/usr/bin/python3

import argparse
import json

def main(input_json, test_percent, evaluation_percent, target_channels):
    if target_channels:
        print("Targeting channel distrubution...")
        target_channel_distribution(input_json, test_percent, evaluation_percent)
    else:
        print("Targeting dataset size distrubution...")
        target_size_distribution(input_json, test_percent, evaluation_percent)

def target_channel_distribution(input_json, test_percent, evaluation_percent):
    if (test_percent + evaluation_percent > 1):
        print(f"ERROR: cannot allocate {(test_percent + evaluation_percent)*100}% to testing and evaluation databases")
        exit(-1)
    
    test = []
    evaluation = []
    train = []
    print(f"Opening file {input_json}:")
    with open(input_json, "r") as data:
        raw_json = json.load(data)
        
        channels = {}
        for video in raw_json:
            if len(video["clips"]) == 0:
                continue
            this_channel = video["clips"][0]["category"]
            if this_channel not in channels:
                channels[this_channel] = {}
                
            channels[this_channel][video["video_id"]] = video["clips"]
        
       
        channel_lengths = {channel:(sum(sum(clip["duration"] for clip in channels[channel][id]) for id in channels[channel])) for channel in channels}
        video_orders = {channel:(sorted(channels[channel].keys(), key=lambda id: sum(clip["duration"] for clip in channels[channel][id]))) for channel in channels}
        
        for channel in channels:
            print(f"Distributing {channel_lengths[channel]} seconds from {len(video_orders[channel])} videos for channel {channel}...")
            video_index = 0
            
            test_target = test_percent * channel_lengths[channel]
            test_length = 0
            print(f"|\tBuilding test dataset, targeting {test_target} seconds of video")
            while test_length < test_target:
                video_id = video_orders[channel][video_index]
                clips = channels[channel][video_id]
                test.append({"video_id":video_id, "clips":clips})
                test_length += sum(clip["duration"] for clip in clips)
                video_index += 1
            print(f"|\t-\tFinished building test dataset with {test_length} seconds from {len(test)} videos")
            
            evaluation_target = evaluation_percent * channel_lengths[channel]
            evaluation_length = 0
            print(f"|\tBuilding evaluation dataset, targeting {test_target} seconds of video")
            while evaluation_length < evaluation_target:
                video_id = video_orders[channel][video_index]
                clips = channels[channel][video_id]
                evaluation.append({"video_id":video_id, "clips":clips})
                evaluation_length += sum(clip["duration"] for clip in clips)
                video_index += 1
            print(f"|\t-\tFinished building evaluation dataset with {evaluation_length} seconds from {len(evaluation)} videos")
            
            train_length = 0
            print(f"|\tBuilding train dataset with remaining {channel_lengths[channel]-(test_length+evaluation_length)} seconds from {len(channels[channel])-(len(test)+len(evaluation))} videos")
            for video_id in video_orders[channel][video_index:]:
                clips = channels[channel][video_id]
                train.append({"video_id":video_id, "clips":clips})
                train_length += sum(clip["duration"] for clip in clips)
            
    test_length = sum(sum(clip["duration"] for clip in video["clips"]) for video in test)
    evaluation_length = sum(sum(clip["duration"] for clip in video["clips"]) for video in evaluation)
    train_length = sum(sum(clip["duration"] for clip in video["clips"]) for video in train)
    total_length = test_length+evaluation_length+train_length

    print("Saving clips as new json files...")
    test_filename = f"{'.'.join(input_json.split('.')[:-1])}.target_channels.test.json"
    print(f"Saving {(test_length/total_length)*100}% of data to {test_filename}")
    with open(test_filename, "w") as file:
        json.dump(test, file, indent=4)
        
    evaluation_filename = f"{'.'.join(input_json.split('.')[:-1])}.target_channels.evaluation.json"
    print(f"Saving {(evaluation_length/total_length)*100}% of data to {evaluation_filename}")
    with open(evaluation_filename, "w") as file:
        json.dump(evaluation, file, indent=4)
        
    train_filename = f"{'.'.join(input_json.split('.')[:-1])}.target_channels.train.json"
    print(f"Saving {(train_length/total_length)*100}% of data to {train_filename}")
    with open(train_filename, "w") as file:
        json.dump(train, file, indent=4)

def target_size_distribution(input_json, test_percent, evaluation_percent):
    if (test_percent + evaluation_percent > 1):
        print(f"ERROR: cannot allocate {(test_percent + evaluation_percent)*100}% to testing and evaluation databases")
        exit(-1)
        
    test = []
    evaluation = []
    train = []
    print(f"Opening file {input_json}:")
    with open(input_json, "r") as data:
        raw_json = json.load(data)
        video_clips = {}
        for video in raw_json:
            video_clips[video["video_id"]] = video["clips"]
            
        total_length = sum(sum(clip["duration"] for clip in video_clips[id]) for id in video_clips)
        video_order = sorted(video_clips.keys(), key=lambda id: sum(clip["duration"] for clip in video_clips[id]))
        print(f"Read in {total_length} seconds from {len(video_clips)} videos...")
        
        video_index = 0
        
        test_target = test_percent * total_length
        test_length = 0
        print(f"|\tBuilding test dataset, targeting {test_target} seconds of video")
        while test_length < test_target:
            video_id = video_order[video_index]
            clips = video_clips[video_id]
            test.append({"video_id":video_id, "clips":clips})
            test_length += sum(clip["duration"] for clip in clips)
            video_index += 1
        print(f"|\t-\tFinished building test dataset with {test_length} seconds from {len(test)} videos")
        
        evaluation_target = evaluation_percent * total_length
        evaluation_length = 0
        print(f"|\tBuilding evaluation dataset, targeting {test_target} seconds of video")
        while evaluation_length < evaluation_target:
            video_id = video_order[video_index]
            clips = video_clips[video_id]
            evaluation.append({"video_id":video_id, "clips":clips})
            evaluation_length += sum(clip["duration"] for clip in clips)
            video_index += 1
        print(f"|\t-\tFinished building evaluation dataset with {evaluation_length} seconds from {len(evaluation)} videos")
        
        train_length = 0
        print(f"|\tBuilding train dataset with remaining {total_length-(test_length+evaluation_length)} seconds from {len(video_clips)-(len(test)+len(evaluation))} videos")
        for video_id in video_order[video_index:]:
            clips = video_clips[video_id]
            train.append({"video_id":video_id, "clips":clips})
            train_length += sum(clip["duration"] for clip in clips)
        
    print("Saving clips as new json files...")
    
    test_filename = f"{'.'.join(input_json.split('.')[:-1])}.test.json"
    print(f"Saving {(test_length/total_length)*100}% of data to {test_filename}")
    with open(test_filename, "w") as file:
        json.dump(test, file, indent=4)
        
    evaluation_filename = f"{'.'.join(input_json.split('.')[:-1])}.evaluation.json"
    print(f"Saving {(evaluation_length/total_length)*100}% of data to {evaluation_filename}")
    with open(evaluation_filename, "w") as file:
        json.dump(evaluation, file, indent=4)
        
    train_filename = f"{'.'.join(input_json.split('.')[:-1])}.train.json"
    print(f"Saving {(train_length/total_length)*100}% of data to {train_filename}")
    with open(train_filename, "w") as file:
        json.dump(train, file, indent=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use video json file to splt data into train, test, and evaluation groups.')
    parser.add_argument('json', metavar='all_video_metadata.json',
                        help='json of videos to split')
    parser.add_argument('-c', '--channels', action='store_true',
                        help='Target dataset distribution (default targets dataset size)')
    parser.add_argument('-t', '--test', metavar='N', type=float,
                        help='number of test videos',
                        default=0.05, nargs='?')
    parser.add_argument('-e', '--evaluation', metavar='N', type=float,
                        help='number of evaluation videos',
                        default=0.05, nargs='?')
    args = parser.parse_args()
    main(args.json, args.test, args.evaluation, args.channels)
