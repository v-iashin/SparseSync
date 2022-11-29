from argparse import ArgumentParser
import json

'''
e.g.: 
    python3 scripts/make_sports_and_news.py 
        --dataset data/sports_and_news/all_video_metadata_v5_label_uniform.target_channels.evaluation.json 
        --csv data/sports_and_news_uniform.evaluation.csv 
        --json data/sports_and_news_uniform.evaluation.json
'''
def main(json_path, csv_out_path, json_out_path):
    with open(json_path, 'r') as file_data:
        videos = json.load(file_data)
        
        vid2offset_params = {}
        with open(csv_out_path, 'w') as csv_file:
            for video in videos:
                for clip in video["clips"]:
                    for i in range(len(clip["video_start_end"])):
                        id = f'{".".join(clip["path"].split("/")[-1].split(".")[:-1])}_{clip["video_start_end"][i][0]}'.replace(".","_")
                        csv_file.write(f'{id},{clip["video_start_end"][i][0]},{clip["category"]}\n')
                        vid2offset_params[id] = {
                            'offset_sec': float(clip["audio_offset"][i][1]),
                            'v_start_i_sec': float(clip["video_start_end"][i][0])
                        }
                        
        with open(json_out_path, 'w') as json_out:
            json.dump(vid2offset_params, json_out)
            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help="input json file to read sports and news dataset (e.g. data/sports_and_news/all_video_metadata_v5_label_uniform.target_channels.evaluation.json)")
    parser.add_argument('--csv', help="ouput filename for csv (e.g. data/sports_and_news_uniform.evaluation.csv)")
    parser.add_argument('--json', help="output filename for json (e.g. data/sports_and_news_uniform.evaluation.json)")
    args = parser.parse_args()
    
    main(args.dataset, args.csv, args.json)
