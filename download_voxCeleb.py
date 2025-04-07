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
from concurrent.futures import ThreadPoolExecutor

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
    --cookies-from-browser: (Optional) Browser name to load cookies from (if provided without a value, defaults to "chrome")
    --extract_frames:       Select to extract frames from videos
    --preprocessing:        Select to preprocess extracted frames
    --delete_mp4:           Delete the original video downloaded from YouTube
    --delete_or_frames:     Delete the original extracted frames

Example:
python download_voxCeleb.py --output_path ./VoxCeleb1_test --metadata_path ./txt_test --dataset vox1 \
      --fail_video_ids ./fail_video_ids_test.txt --delete_mp4 --extract_frames --preprocessing \
      --cookies ./cookies.txt --cookies-from-browser firefox
"""

DEVNULL = open(os.devnull, 'wb')
REF_FPS = 25

parser = ArgumentParser()
parser.add_argument("--output_path", required=True, help='Path to save the videos')
parser.add_argument("--metadata_path", required=True, help='Path to metadata')
parser.add_argument("--dataset", required=True, type=str, choices=('vox1', 'vox2'), help="Download vox1 or vox2 dataset")
parser.add_argument("--fail_video_ids", default=None, help='Txt file to save videos that fail to download')
parser.add_argument("--cookies", default=None, help='(Optional) Path to cookies file (Netscape format) for YouTube authentication')
# Using nargs='?' and const='chrome' so the flag is optional and defaults to "chrome" when not provided a value.
parser.add_argument("--cookies-from-browser", nargs='?', const='chrome', default=None,
                    help='(Optional) Browser name to load cookies from (e.g., chrome, firefox). Defaults to "chrome" if flag is present without a value.')
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
    if cookies:
        ydl_opts['cookiefile'] = cookies
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
        first_frame_sec = round(first_frame / float(REF_FPS), 3)
        last_frame_sec = round(last_frame / float(REF_FPS), 3)
        head, tail = os.path.split(utterance_files[i])
        tail = tail.split('.tx')[0]
        chunk_name = os.path.join(chunk_folder, video_id + '#' + tail + '#' + str(st) + '-' + str(en) + '.mp4')
        
        command_fps = 'ffmpeg -y -i {} -qscale:v 5 -r 25 -threads 1 -ss {} -to {} -strict -2 {} -loglevel quiet'.format(
            video_path, first_frame_sec, last_frame_sec, chunk_name)
        os.system(command_fps)
        chunk_videos.append(chunk_name)

    return chunk_videos

def process_video(id_index, video_path, output_path, cookies, browser_cookies, fail_video_ids,
                  extract_frames, preprocessing, delete_mp4, delete_or_frames, landmark_est):
    video_id = video_path.split(os.sep)[-2]
    output_path_video = os.path.join(output_path, id_index, video_id)
    make_path(output_path_video)

    # Check if chunk videos already exist.
    output_path_chunk_videos = os.path.join(output_path_video, 'chunk_videos')
    if os.path.exists(output_path_chunk_videos) and glob.glob(os.path.join(output_path_chunk_videos, '*.mp4')):
        print('URL https://www.youtube.com/watch?v=' + video_id + ' is already loaded')
        return

    print('Download video id {}. Save to {}'.format(video_id, output_path_video))
    txt_metadata = sorted(glob.glob(os.path.join(video_path, '*.txt')))
    mp4_path = os.path.join(output_path_video, '{}.mp4'.format(video_id))
    if not os.path.exists(mp4_path):
        success = download_video(video_id, mp4_path, id_index, fail_video_ids=fail_video_ids,
                                 cookies=cookies, browser_cookies=browser_cookies)
    else:
        success = True

    if success:
        make_path(output_path_chunk_videos)
        chunk_videos = split_in_utterances(video_id, mp4_path, txt_metadata, output_path_chunk_videos)
        if delete_mp4:
            os.system('rm -rf {}'.format(mp4_path))

        extracted_frames_path = os.path.join(output_path_video, 'frames')
        if extract_frames:
            extract_frames_opencv(chunk_videos, REF_FPS, extracted_frames_path)
        if preprocessing:
            image_files = sorted(glob.glob(os.path.join(extracted_frames_path, '*.png')))
            if len(image_files) > 0:
                save_dir = os.path.join(output_path_video, 'frames_cropped')
                make_path(save_dir)
                preprocess_frames(args.dataset, output_path_video, extracted_frames_path,
                                  image_files, save_dir, txt_metadata, landmark_est)
            else:
                print('There are no extracted frames on path: {}'.format(extracted_frames_path))
        if delete_or_frames and os.path.exists(extracted_frames_path):
            os.system('rm -rf {}'.format(extracted_frames_path))
    else:
        print('Error downloading video {}/{}. Deleting folder {}'.format(id_index, video_id, output_path_video))
        os.system('rm -rf {}'.format(output_path_video))

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

    cookies = args.cookies if args.cookies else ("cookies.txt" if os.path.exists("cookies.txt") else None)
    browser_cookies = args.cookies_from_browser

    if not cookies and not browser_cookies:
        print("Warning: No cookies file or browser cookies specified. Downloads may fail for videos requiring authentication.")

    if not os.path.exists(metadata_path):
        print('Please download the metadata for {} dataset'.format(dataset))
        exit()

    ids_path = sorted(glob.glob(os.path.join(metadata_path, '*/')))
    print('{} dataset has {} identities'.format(dataset, len(ids_path)))

    print('--Delete original mp4 videos: \t\t{}'.format(delete_mp4))
    print('--Delete original frames: \t\t{}'.format(delete_or_frames))
    print('--Extract frames from chunk videos: \t{}'.format(extract_frames))
    print('--Preprocess original frames: \t\t{}'.format(preprocessing))

    landmark_est = None
    if preprocessing:
        landmark_est = LandmarksEstimation(type='2D')

    # Create a thread pool executor with maximum 10 threads.
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for id_path in ids_path:
            id_index = id_path.split(os.sep)[-2]
            videos_path = sorted(glob.glob(os.path.join(id_path, '*/')))
            print('*********************************************************')
            print('Identity: {} with {} videos'.format(id_index, len(videos_path)))
            for video_path in videos_path:
                futures.append(
                    executor.submit(process_video, id_index, video_path, output_path, cookies,
                                    browser_cookies, fail_video_ids, extract_frames,
                                    preprocessing, delete_mp4, delete_or_frames, landmark_est)
                )
        # Wait for all tasks to finish.
        for future in futures:
            future.result()
