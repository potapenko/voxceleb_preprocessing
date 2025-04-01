import numpy as np
import pandas as pd
import imageio
import os
import warnings
import glob
import time
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
warnings.filterwarnings("ignore")
import cv2
import yt_dlp
import subprocess

from libs.utilities import make_path
from preprocess_voxCeleb import extract_frames_opencv, preprocess_frames
from libs.landmarks_estimation import LandmarksEstimation

"""
1. Download videos from YouTube for VoxCeleb1 dataset
2. Generate chunk videos using the metadata provided by VoxCeleb1 dataset

Optionally:
    3. Extract frames from chunk videos
    4. Preprocess extracted frames by cropping them around the detected faces

Arguments:
    output_path:            Path to save the videos
    metadata_path:          Path to metadata
    dataset:                Dataset name: vox1 or vox2
    fail_video_ids:         Txt file to save video IDs that fail to download
    --cookies:              (Optional) Path to cookies file (Netscape format) for YouTube authentication
    --cookies-from-browser: (Optional) Browser name (e.g., chrome, firefox) to load cookies automatically
    --extract_frames:       Select to extract frames from videos
    --preprocessing:        Select to preprocess extracted frames
    --delete_mp4:           Delete the original video downloaded from YouTube
    --delete_or_frames:     Delete the original extracted frames

Example:
python download_voxCeleb.py --output_path ./VoxCeleb1_test --metadata_path ./txt_test --dataset vox1 \
      --fail_video_ids ./fail_video_ids_test.txt --delete_mp4 --extract_frames --preprocessing \
      --cookies ./cookies.txt --cookies-from-browser chrome
"""

DEVNULL = open(os.devnull, 'wb')
REF_FPS = 25

parser = ArgumentParser()
parser.add_argument("--output_path", required=True, help='Path to save the videos')
parser.add_argument("--metadata_path", required=True, help='Path to metadata')
parser.add_argument("--dataset", required=True, type=str, choices=('vox1', 'vox2'), help="Download vox1 or vox2 dataset")
parser.add_argument("--fail_video_ids", default=None, help='Txt file to save videos that fail to download')
parser.add_argument("--cookies", default=None, help='(Optional) Path to cookies file (Netscape format) for YouTube authentication')
parser.add_argument("--cookies-from-browser", default=None, help='(Optional) Browser name to load cookies from (e.g., chrome, firefox)')
parser.add_argument("--extract_frames", action='store_true', help='Extract frames from videos')
parser.set_defaults(extract_frames=False)
parser.add_argument("--preprocessing", action='store_true', help='Preprocess extracted frames')
parser.set_defaults(preprocessing=False)
parser.add_argument("--delete_mp4", action='store_true', help='Delete original video downloaded from YouTube')
parser.set_defaults(delete_mp4=False)
parser.add_argument("--delete_or_frames", dest='delete_or_frames', action='store_true', help="Delete original frames and keep only the cropped frames")
parser.set_defaults(delete_or_frames=False)

def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')

def download_video(video_id, video_path, id_path, fail_video_ids=None, cookies=None, browser_cookies=None):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': video_path,
        'progress_hooks': [my_hook],
    }
    # If a cookies file is provided, use it.
    if cookies:
        ydl_opts['cookiefile'] = cookies
    # If cookies-from-browser is provided, pass it along.
    if browser_cookies:
        ydl_opts['cookies_from_browser'] = browser_cookies

    success = True
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(['https://www.youtube.com/watch?v=' + video_id])
    except KeyboardInterrupt:
        print('Stopped')
        exit()
    except Exception as e:
        print('Error downloading video {}: {}'.format(video_id, e))
        success = False
        if fail_video_ids is not None:
            with open(fail_video_ids, "a") as f:
                f.write(id_path + '/' + video_id + '\n')
    return success

def split_in_utterances(video_id, video_path, utterance_files, chunk_folder):
    chunk_videos = []
    utterances = [pd.read_csv(f, sep='\t', skiprows=6) for f in utterance_files]
    for i, utterance in enumerate(utterances):
        first_frame, last_frame = utterance['FRAME '].iloc[0], utterance['FRAME '].iloc[-1]
        st = first_frame
        en = last_frame    
        first_frame = round(first_frame / float(REF_FPS), 3)
        last_frame = round(last_frame / float(REF_FPS), 3)
        head, tail = os.path.split(utterance_files[i])
        tail = tail.split('.tx')[0]
        chunk_name = os.path.join(chunk_folder, video_id + '#' + tail + '#' + str(st) + '-' + str(en) + '.mp4')
        
        command_fps = 'ffmpeg -y -i {} -qscale:v 5 -r 25 -threads 1 -ss {} -to {} -strict -2 {} -loglevel quiet'.format(video_path, first_frame, last_frame, chunk_name)
        os.system(command_fps)
        chunk_videos.append(chunk_name)

    return chunk_videos

if __name__ == "__main__":
    args = parser.parse_args()
    extract_frames = args.extract_frames
    preprocessing = args.preprocessing
    fail_video_ids = args.fail_video_ids
    output_path = args.output_path
    make_path(output_path)
    delete_mp4 = args.delete_mp4
    delete_or_frames = args.delete_or_frames
    metadata_path = args.metadata_path
    dataset = args.dataset

    # Determine cookies file if provided, or default to "cookies.txt" if exists.
    cookies = None
    if args.cookies:
        cookies = args.cookies
    elif os.path.exists("cookies.txt"):
        cookies = "cookies.txt"

    # Determine browser cookies option if provided.
    browser_cookies = args.cookies_from_browser

    if not cookies and not browser_cookies:
        print("Warning: No cookies file or browser cookies specified. Downloads may fail for videos requiring authentication.")

    if not os.path.exists(metadata_path):
        print('Please download the metadata for {} dataset'.format(dataset))
        exit()

    ids_path = glob.glob(os.path.join(metadata_path, '*/'))
    ids_path.sort()
    print('{} dataset has {} identities'.format(dataset, len(ids_path)))

    print('--Delete original mp4 videos: \t\t{}'.format(delete_mp4))
    print('--Delete original frames: \t\t{}'.format(delete_or_frames))
    print('--Extract frames from chunk videos: \t{}'.format(extract_frames))
    print('--Preprocess original frames: \t\t{}'.format(preprocessing))

    if preprocessing:
        landmark_est = LandmarksEstimation(type='2D')

    for i, id_path in enumerate(ids_path):
        id_index = id_path.split('/')[-2]
        videos_path = glob.glob(os.path.join(id_path, '*/'))
        videos_path.sort()
        print('*********************************************************')
        print('Identity {}/{}: {} videos for {} identity'.format(i, len(ids_path), len(videos_path), id_index))
        
        for j, video_path in enumerate(videos_path):
            print('{}/{} videos'.format(j, len(videos_path)))
            video_id = video_path.split('/')[-2]
            output_path_video = os.path.join(output_path, id_index, video_id)
            make_path(output_path_video)
        
            print('Download video id {}. Save to {}'.format(video_id, output_path_video))
            
            txt_metadata = glob.glob(os.path.join(video_path, '*.txt'))
            txt_metadata.sort()

            mp4_path = os.path.join(output_path_video, '{}.mp4'.format(video_id))
            if not os.path.exists(mp4_path):
                success = download_video(video_id, mp4_path, id_index, fail_video_ids=fail_video_ids,
                                         cookies=cookies, browser_cookies=browser_cookies)
            else:
                # Video already exists
                success = True

            if success:
                # Split into small videos using the metadata
                output_path_chunk_videos = os.path.join(output_path, id_index, video_id, 'chunk_videos')
                make_path(output_path_chunk_videos)
                chunk_videos = split_in_utterances(video_id, mp4_path, txt_metadata, output_path_chunk_videos)
                if delete_mp4:  # Delete original video downloaded from YouTube
                    command_delete = 'rm -rf {}'.format(mp4_path)
                    os.system(command_delete)

                extracted_frames_path = os.path.join(output_path_video, 'frames')
                if extract_frames:
                    # Run frame extraction
                    extract_frames_opencv(chunk_videos, REF_FPS, extracted_frames_path)
                if preprocessing:
                    # Run preprocessing
                    image_files = glob.glob(os.path.join(extracted_frames_path, '*.png'))
                    image_files.sort()
                    if len(image_files) > 0:
                        save_dir = os.path.join(output_path_video, 'frames_cropped')
                        make_path(save_dir)
                        preprocess_frames(dataset, output_path_video, extracted_frames_path, image_files, save_dir, txt_metadata, landmark_est)
                    else:
                        print('There are no extracted frames on path: {}'.format(extracted_frames_path))
                
                if delete_or_frames and os.path.exists(extracted_frames_path):  # Delete original frames
                    command_delete = 'rm -rf {}'.format(extracted_frames_path)
                    os.system(command_delete)
            else:
                print('Error downloading video {}/{}. Deleting folder {}'.format(id_index, video_id, output_path_video))
                command_delete = 'rm -rf {}'.format(output_path_video)
                os.system(command_delete)
