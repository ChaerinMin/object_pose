import cv2
import os
import ffmpeg
import numpy as np
from glob import glob
from natsort import natsorted
from typing import Dict, Iterable, Generator, Tuple
import json
import ipdb
import datetime

def parse_timestamp(filename):
    # Strip the extension and parse the remaining part as a datetime
    timestamp_str = filename.split('_')[-1].split('.')[0]
    return datetime.datetime.fromtimestamp(int(timestamp_str) / 1e6)

def find_closest_video(folder, anchor_timestamp):
    min_diff = datetime.timedelta(seconds=1)  # Max allowed difference
    closest_video = None

    for filename in os.listdir(folder):
        if filename.endswith('.mp4'):
            video_timestamp = parse_timestamp(filename)
            time_diff = abs(video_timestamp - anchor_timestamp)
            
            if time_diff < min_diff:
                min_diff = time_diff
                closest_video = filename
    
    return closest_video, min_diff

class Reader():
    iterator = []

    def __init__(
            self, inp_type: str, path: str, undistort: bool=False, cams_to_remove=[], ith: int=0, start_frame_path=None, anchor_camera=None
        ):
        """ith: the ith video in each folder will be processed."""
        self.type = inp_type
        self.ith = ith
        self.frame_count = int(1e9)
        self.start_frames = None
        self.path = path
        self.cams_to_remove = cams_to_remove
        self.to_delete = []
        
        if self.type == "video":
            self.streams = {}
            self.vids = []
            if anchor_camera:
                mp4_list = natsorted(glob(f"{path}/{anchor_camera}/*.mp4"))
                if len(mp4_list) > self.ith:
                    self.vids.append(natsorted(glob(f"{path}/{anchor_camera}/*.mp4"))[self.ith])
            else:
                for cam in os.listdir(path):
                    if 'imu' not in cam and 'mic' not in cam and len(glob(f"{path}/{cam}/*.mp4")) > self.ith:
                        if cam not in cams_to_remove:
                            self.vids.append(natsorted(glob(f"{path}/{cam}/*.mp4"))[self.ith])
                    if len(self.vids) > 0:
                        break
            self.anchor_timestamp = parse_timestamp(self.vids[0])
            self.check_timestamp()
            self.init_videos()
            if start_frame_path:
                with open(start_frame_path, 'r') as file:
                    start_frames = json.load(file)
                self.start_frames = start_frames
        else:
            pass

        # Sanity checks
        assert(self.frame_count < int(1e9)), "No frames found"
        if self.frame_count <= 0:
            print("No frames found")
            
        self.cur_frame = 0
    
    def _get_next_frame(self, frame_idx) -> Dict[str, np.ndarray]:
        """ Get next frame (stride 1) from each camera"""
        self.cur_frame = frame_idx
        
        if self.cur_frame == self.frame_count:
            return {}

        frames = {}
        for cam_name, cam_cap in self.streams.items():
            if self.start_frames:
                start_frame = self.start_frames.get(cam_name, [0, 0])[0]
            else:
                start_frame = 0
            cam_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + start_frame)
            suc, frame = cam_cap.read()
            if not suc:
                raise RuntimeError(f"Couldn't retrieve frame from {cam_name}")
            frames[cam_name] = frame
        
        return frames

    def check_timestamp(self):
        
        for cam in os.listdir(self.path):
            if 'imu' not in cam and 'mic' not in cam and cam not in self.cams_to_remove and cam not in self.vids[0]:
                closest_file, time_diff = find_closest_video(f"{self.path}/{cam}", self.anchor_timestamp)
                if closest_file:
                    self.vids.append(f"{self.path}/{cam}/{closest_file}")
                else:
                    self.to_delete.append(cam.split('/')[-1].rsplit('_', 0)[0])
            
    def reinit(self):
        """ Reinitialize the reader """
        if self.type == "video":
            self.release_videos()
            self.init_videos()

        self.cur_frame = 0

    def init_videos(self):
        """ Create video captures for each video
                ith: the ith video in each folder will be processed."""
        for vid in self.vids:
            cap = cv2.VideoCapture(vid)
            frame_count = int(ffmpeg.probe(vid, cmd="ffprobe")["streams"][0]["nb_frames"])
            self.frame_count = min(self.frame_count, frame_count)
            cam_name = os.path.basename(vid).split(".")[0]
            self.streams[cam_name] = cap
            
        self.frame_count -= 5 # To account for the last few frames that are corrupted

    def release_videos(self):
        for cap in self.streams.values():
            cap.release()
    
    def __call__(self, frames: Iterable[int]=[]):
        # Sort the frames so that we access them in order
        frames = sorted(frames)
        self.iterator = frames
        
        for frame_idx in frames:
            frame = self._get_next_frame(frame_idx)
            if not frame:
                break
                
            yield frame, self.cur_frame

        # Reinitialize the videos
        self.reinit()

if __name__ == "__main__":
    reader = Reader("video", "/hdd_data/common/BRICS/hands/peisen/actions/abduction_adduction/", 5, 16, 3)
    for i in range(len(reader)):
        frames, frame_num = reader.get_frames()
        print(frame_num)
