import numpy as np
from tqdm import tqdm
import os
import glob
from argparse import ArgumentParser
import cv2
import torch
from skimage.transform import resize
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from libs.utilities import make_path, _parse_metadata_file, crop_box, read_image_opencv
from libs.ffhq_cropping import crop_using_landmarks
from libs.landmarks_estimation import LandmarksEstimation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

REF_FPS = 25        # fps to extract frames
REF_SIZE = 360      # Height
LOW_RES_SIZE = 400  

parser = ArgumentParser()
parser.add_argument("--root_path", default='videos', required=True, help='Path to youtube videos')
parser.add_argument("--metadata_path", default='metadata', required=True, help='Path to metadata')
parser.add_argument("--dataset", required=True, type=str, choices=('vox1', 'vox2'), help="Download vox1 or vox2 dataset")
parser.add_argument("--delete_videos", action='store_true', help='Delete chunk videos')
parser.set_defaults(delete_videos=False)
parser.add_argument("--delete_or_frames", dest='delete_or_frames', action='store_true',
                    help="Delete original frames and keep only the cropped frames")
parser.set_defaults(delete_or_frames=False)


def get_frames(video_path, frames_path, video_index, fps):
    cap = cv2.VideoCapture(video_path)
    counter = 0
    frame_skip = fps
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if counter % frame_skip == 0:
            filename = os.path.join(frames_path, '{:02d}_{:06d}.png'.format(video_index, counter))
            cv2.imwrite(filename, frame)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()


def extract_frames_opencv(videos_tmp, fps, frames_path):
    logging.info('Extracting frames...')
    make_path(frames_path)
    for i in tqdm(range(len(videos_tmp))):
        get_frames(videos_tmp[i], frames_path, i, fps)


def preprocess_frames(dataset,
                      output_path_video,
                      frames_path,
                      image_files,
                      save_dir,
                      txt_metadata,
                      landmark_est=None):

    if dataset == 'vox2':
        image_ref = read_image_opencv(image_files[0])
        mult = image_ref.shape[0] / REF_SIZE
        image_ref = resize(image_ref, (REF_SIZE, int(image_ref.shape[1] / mult)), preserve_range=True)
    else:
        image_ref = None

    info_metadata = _parse_metadata_file(txt_metadata, dataset=dataset, frame=image_ref)

    logging.info('Preprocessing frames...')
    errors = []
    chunk_id = 0
    frame_i = 0
    for i in tqdm(range(len(image_files))):
        image_file = image_files[i]
        image_name = os.path.basename(image_file)
        image_chunk_id = int(image_name.split('.')[0].split('_')[0])

        # Update chunk index if necessary
        if chunk_id != image_chunk_id:
            chunk_id = image_chunk_id
            frame_i = 0

        bbox = None
        if chunk_id < len(info_metadata):
            frames = info_metadata[chunk_id]['frames']
            bboxes_metadata = info_metadata[chunk_id]['bboxes']
            index = frame_i + 1 + frame_i * (REF_FPS - 1)
            if index < len(bboxes_metadata):
                bbox = bboxes_metadata[index]
                frame = frames[index]

        if bbox is not None:
            image = read_image_opencv(image_file)
            frame_copy = image.copy()
            (h, w) = image.shape[:2]

            scale_res = REF_SIZE / float(h)
            bbox_new = bbox.copy()
            bbox_new = [coord / scale_res for coord in bbox_new]

            cropped_image, bbox_scaled = crop_box(frame_copy, bbox_new, scale_crop=2.0)
            filename = os.path.join(save_dir, image_name)
            cv2.imwrite(filename, cv2.cvtColor(cropped_image.copy(), cv2.COLOR_RGB2BGR))
            h_c, w_c, _ = cropped_image.shape

            image_tensor = torch.tensor(
                np.transpose(cropped_image, (2, 0, 1))
            ).float().cuda()

            if landmark_est is not None:
                with torch.no_grad():
                    landmarks = landmark_est.detect_landmarks(image_tensor.unsqueeze(0))
                    landmarks = landmarks[0].detach().cpu().numpy()
                    landmarks = np.asarray(landmarks)
                    if not (np.any(landmarks > w_c) or np.any(landmarks < 0)):
                        img = crop_using_landmarks(cropped_image, landmarks)
                        if img is not None:
                            filename = os.path.join(save_dir, image_name)
                            cv2.imwrite(filename, cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))
        frame_i += 1


def transform(pt, center, scale, resolution, invert=False):
    """
    Transform function used internally by the FAN code to compute an affine transform matrix.
    """
    center = [float(center[0]), float(center[1])]
    scale = float(scale)
    resolution = float(resolution)

    h = scale * 200.0
    t = torch.zeros((3, 3))
    t[0, 0] = float(resolution / h)
    t[1, 1] = float(resolution / h)
    t[2, 2] = 1.0

    if invert:
        t = torch.inverse(t)

    return t


def process_video(id_index, video_path, root_path, metadata_path, dataset, delete_videos, delete_or_frames, landmark_est):
    video_id = os.path.basename(os.path.normpath(video_path))
    output_path_video = os.path.join(root_path, id_index, video_id)
    complete_file = os.path.join(output_path_video, 'processing_complete.txt')
    
    # Check if processing already completed
    if os.path.exists(complete_file):
        logging.info(f"Skipping {output_path_video}: already processed.")
        return

    logging.info(f"Processing video: {output_path_video}")
    output_path_chunk_videos = os.path.join(output_path_video, 'chunk_videos')
    if not os.path.exists(output_path_chunk_videos):
        logging.warning(f'Chunk videos folder {output_path_chunk_videos} does not exist.')
        return

    txt_metadata = glob.glob(os.path.join(metadata_path, id_index, video_id, '*.txt'))
    txt_metadata.sort()

    # Extract frames
    videos_tmp = glob.glob(os.path.join(output_path_chunk_videos, '*.mp4'))
    videos_tmp.sort()
    extracted_frames_path = os.path.join(output_path_video, 'frames')
    if len(videos_tmp) > 0:
        extract_frames_opencv(videos_tmp, REF_FPS, extracted_frames_path)
    else:
        logging.warning(f'No videos found in {output_path_video}')
        return

    # Preprocess frames
    image_files = glob.glob(os.path.join(extracted_frames_path, '*.png'))
    image_files.sort()
    if len(image_files) > 0:
        save_dir = os.path.join(output_path_video, 'frames_cropped')
        make_path(save_dir)
        preprocess_frames(dataset,
                          output_path_video,
                          extracted_frames_path,
                          image_files,
                          save_dir,
                          txt_metadata,
                          landmark_est)
    else:
        logging.warning(f'No frames found in {extracted_frames_path}')

    # Optionally delete chunk videos
    if delete_videos:
        command_delete = 'rm -rf {}'.format(os.path.join(output_path_video, '*.mp4'))
        os.system(command_delete)
        logging.info(f"Deleted chunk videos in {output_path_video}")

    # Optionally delete original frames
    if delete_or_frames:
        frames_folder_name = 'frames'
        command_delete = 'rm -rf {}'.format(os.path.join(output_path_video, frames_folder_name))
        os.system(command_delete)
        logging.info(f"Deleted original frames in {output_path_video}")

    # Mark processing complete
    with open(complete_file, 'w') as f:
        f.write("Processing complete")
    logging.info(f"Completed processing for {output_path_video}")


if __name__ == "__main__":
    args = parser.parse_args()
    root_path = args.root_path
    metadata_path = args.metadata_path
    dataset = args.dataset
    delete_videos = args.delete_videos
    delete_or_frames = args.delete_or_frames

    if not os.path.exists(root_path):
        logging.error(f'Videos path {root_path} does not exist')
        exit(1)
    if not os.path.exists(metadata_path):
        logging.error(f'Please download the metadata for {dataset} dataset')
        exit(1)

    landmark_est = LandmarksEstimation(type='2D')

    logging.info(f"--Delete chunk videos: {delete_videos}")
    logging.info(f"--Delete original frames: {delete_or_frames}")

    ids_path = glob.glob(os.path.join(root_path, '*/'))
    ids_path.sort()
    logging.info(f"Dataset has {len(ids_path)} identities")

    # Use ThreadPoolExecutor to process videos in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for id_dir in ids_path:
            id_index = os.path.basename(os.path.normpath(id_dir))
            videos_path = glob.glob(os.path.join(id_dir, '*/'))
            videos_path.sort()
            logging.info(f"*********************************************************")
            logging.info(f"Identity: {id_index} has {len(videos_path)} videos")
            for video_path in videos_path:
                futures.append(executor.submit(process_video, id_index, video_path, root_path,
                                                   metadata_path, dataset, delete_videos, delete_or_frames, landmark_est))
        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error during processing: {e}")

    logging.info("All video processing tasks completed.")
