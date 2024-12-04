import os
import argparse
from tqdm import tqdm
import random
import cv2
import math 
import numpy as np
import torch
from ultralytics import YOLO, checks
from PIL import Image, ImageDraw, ImageFont
import logging
from skimage import img_as_ubyte
import imageio

import sys
sys.path.append(".")
from src.utils.reader_v2 import Reader
from src.utils.video_handler import frame_preprocess
from src.utils.cameras import removed_cameras, map_camera_names, get_projections, get_ngp_cameras
import src.utils.params as param_utils
from src.utils.renderer import render_image
from src.utils.video_handler import create_video_writer, convert_video_ffmpeg

sys.path.append("./Grounded-SAM-2")
import supervision as sv
from src.sam2_video_predictor import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from src.utils.sam_track_utils import sample_points_from_masks

sys.path.append("../camera_calibration/3rdparty/instant-ngp/build_object")
import pyngp as ngp
from src.utils.instantngp_helper import *

sys.path.append("./thirdparty/dinov2")
import src.utils.dinov2_utils as dinov2_utils
from src.utils.template_util import crop_image, images_to_template_reps, match_image_with_template_reps

import src.utils.colmap_utils as colmap_utils
from src.utils.pytorch3d_utils import setup_renderer, batch_render_loader, batch_render_ref_loader, DRModel, check_for_nan_params, visualize_image_list, alpha_blend, apply_transformation_to_mesh, save_transformed_mesh, random_quaternions, uniform_quaternions
import pytorch3d
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply, save_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRendererWithFragments, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesAtlas,
    TexturesVertex,
    BlendParams
)

from pytorch3d.transforms import Rotate, Translate, matrix_to_quaternion, quaternion_to_matrix, euler_angles_to_matrix, axis_angle_to_matrix, quaternion_to_axis_angle, matrix_to_axis_angle

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

def track_and_save_video(args, grounding_processor, grounding_model, video_predictor, text_prompt, text_features, text_num, images_cv2, save_folder='test.png', view_name='', save_video=True, save_frame=True):
    # prompt grounding dino to get the box coordinates on specific frame
    gap = args.gap
    total_length = len(images_cv2)
    interval = max(total_length // 2 * gap, 1)
    sampled_frame_indices = list(range(0, total_length, interval))
    
    all_results = []
    for ann_frame_idx in sampled_frame_indices:
        image = Image.fromarray(cv2.cvtColor(images_cv2[ann_frame_idx], cv2.COLOR_BGR2RGB)).convert("RGB")

        # run Grounding DINO on the image
        inputs = grounding_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
        all_results.append(results[0])

    # TODO: we assume we only have one candidate.
    all_best_ann_frame_idx = 0
    all_best_score = 0
    all_best_box = None
    all_best_label = ''
    for ann_frame_idx, results in zip(sampled_frame_indices, all_results):
        boxes = results["boxes"].cpu().numpy() 
        labels = results["labels"]
        scores = results["scores"].cpu().numpy()

        valid_boxes = []
        valid_scores = []
        valid_labels = []
        
        for box, label, score in zip(boxes, labels, scores):
            if label.strip():
                valid_boxes.append(box)
                valid_scores.append(score)
                valid_labels.append(label)
        
        valid_boxes = np.array(valid_boxes)
        valid_scores = np.array(valid_scores)
        valid_labels = np.array(valid_labels)
        
        if len(valid_scores) > 0:
            max_score_idx = np.argmax(valid_scores)
            best_box = valid_boxes[max_score_idx]
            best_label = valid_labels[max_score_idx]
            best_score = valid_scores[max_score_idx]
            
            if best_score > all_best_score:
                all_best_ann_frame_idx, all_best_score, all_best_box, all_best_label = ann_frame_idx, best_score, best_box, best_label

    if all_best_box is None:
        return None, [None] * total_length

    all_best_ann_frame_idx_gap = all_best_ann_frame_idx // gap
    input_boxes = [all_best_box]
    OBJECTS = [all_best_label]
    scores = [all_best_score]

    # Using box prompt
    # init video predictor state
    images_cv2_gap = images_cv2[0::gap]
    inference_state = video_predictor.init_state(images_cv=images_cv2_gap)
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=all_best_ann_frame_idx_gap,
            obj_id=object_id,
            box=box,
        )

        if args.to_filter_views:
            bgr_image = images_cv2_gap[all_best_ann_frame_idx_gap].copy()
            rgb_image = bgr_image[:, :, ::-1]
            masks_dict = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            masks = list(masks_dict.values())
            masks = np.concatenate(masks, axis=0)
            combined_mask = np.any(masks, axis=0)
            mask_expanded = combined_mask[:, :, np.newaxis]
            rgb_image[~mask_expanded.repeat(3, axis=2)] = 0
            preprocess_image = open_clip_preprocess(Image.fromarray(rgb_image)).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = open_clip_model.encode_image(preprocess_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                max_prob, max_id = torch.max(text_probs, dim=-1)
                if max_id.item() >= text_num:
                    return None, [None] * total_length

    # Propagate the video predictor to get the segmentation results for each frame
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    video_segments = dict(sorted(video_segments.items(), key=lambda x: x[0]))

    all_masks = [None] * total_length
    for gap_frame_idx, segments in video_segments.items():
        # Get the object IDs and masks
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        all_masks[gap_frame_idx * gap] = masks

    # Assuming OBJECTS and video_segments are defined
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

    if save_video or save_frame:
        # Initialize the VideoWriter object
        frame_size = (image.size[0], image.size[1]) # Set the frame size (adjust as needed based on your frames)
        fps = 30  # Set frames per second

        # Use the 'mp4v' codec for MP4 files
        os.makedirs(save_folder, exist_ok = True)
        video_path = os.path.join(save_folder, view_name + '.mp4')
        if save_frame:
            frame_folder = os.path.join(save_folder, view_name)
            os.makedirs(frame_folder, exist_ok = True)
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        # Iterate through each frame
        for gap_frame_idx, segments in video_segments.items():
            # Read the image/frame
            img = images_cv2[gap_frame_idx * gap]
            
            # Resize image if the size doesn't match the video size (optional, for consistency)
            img = cv2.resize(img, frame_size)
            
            # Get the object IDs and masks
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)

            if save_frame and gap_frame_idx % 1 == 0:
                combined_mask = np.any(masks, axis=0)
                alpha_mask = (combined_mask * 255).astype(np.uint8)
                img_rgba = np.dstack((img, alpha_mask))

                output_filename = os.path.join(frame_folder, str(gap_frame_idx * gap).zfill(6) + '.png')
                cv2.imwrite(output_filename, img_rgba)

            # Create detections using supervision library
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks,  # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            
            # Annotate the image with bounding boxes, labels, and masks
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            if save_video:
                video_writer.write(annotated_frame)
        if save_video:
            video_writer.release()

    return all_best_ann_frame_idx * gap, all_masks

def ngp_train(args, params, cam_names, cam_mapper, all_view_images, all_view_masks, intrs, extrs, dists, frame_id, mesh_dir):
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.reload_network_from_file(args.network)
    
    ## Remove camera from params
    params = [param for param in params if (param["cam_name"] in cam_names) and (param["cam_name"] in cam_mapper)][::1]
    params = np.asarray(params)
    
    testbed.create_empty_nerf_dataset(
        n_images=len(params), aabb_scale=args.aabb_scale
    )
    print(f"Training on {len(params)} views.")
    id_ = 0
    
    imgs_full = []
    imgs = []
    img_names = []

    for idx, param in enumerate(params):
        img_cv2 = all_view_images[param['cam_name']][frame_id]
        mask = all_view_masks[param['cam_name']][frame_id]

        if mask is not None:
            combined_mask = np.any(mask, axis=0)  # Shape: [h, w]
            alpha_channel = np.where(combined_mask, 255, 0).astype(np.uint8)  # Opaque (255) or Transparent (0)
            rgba_img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGBA)
            rgba_img[:, :, 3] = alpha_channel

            img = rgba_img.astype(np.float32)
            img /= 255

            imgs.append(img)
            img_full = img.copy()
            img_full[:, :, 3] = np.ones_like(np.array(img_full[:, :, 3]))
            imgs_full.append(img_full)
            img_names.append(param['cam_name'])
            depth_img = np.zeros((img.shape[0], img.shape[1]))
            img = srgb_to_linear(img)
            # # premultiply
            img[..., :3] *= img[..., 3:4]

            extr = extrs[idx]
            intr = intrs[idx]
            dist = dists[idx]

            testbed.nerf.training.set_image(id_, img, depth_img)
            testbed.nerf.training.set_camera_extrinsics(id_, extr[:3], convert_to_ngp=False)
            testbed.nerf.training.set_camera_intrinsics(
                id_,
                fx=param["fx"],
                fy=param["fy"],
                cx=param["cx"],
                cy=param["cy"],
                k1=param["k1"],
                k2=param["k2"],
                p1=param["p1"],
                p2=param["p2"],
            )
            id_ += 1

    # Taken from i-ngp:scripts/run.py
    # testbed.color_space = ngp.ColorSpace.SRGB
    testbed.nerf.visualize_cameras = True
    testbed.background_color = [0.0, 0.0, 0.0, 0.0]
    testbed.nerf.training.random_bg_color = True
    testbed.training_batch_size = args.batch_size

    testbed.nerf.training.n_images_for_training = id_

    testbed.shall_train = True
    testbed.nerf.training.optimize_extrinsics = False
    testbed.nerf.training.optimize_focal_length = args.optimize_focal_length
    testbed.nerf.training.optimize_distortion = args.optimize_distortion
    testbed.nerf.cone_angle_constant = 0.000

    n_steps = args.n_steps
    old_training_step = 0
    tqdm_last_update = 0

    start = time.time()
    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while testbed.frame():
                # What will happen when training is done?
                if testbed.training_step >= n_steps:
                    break

                # if testbed.training_step == n_steps // 2:
                #     for idx in range(id_):
                #         img_full = imgs_full[idx]
                #         img_full = srgb_to_linear(img_full)
                #         # # premultiply
                #         img_full[..., :3] *= img_full[..., 3:4]
                #         testbed.nerf.training.set_image(idx, img_full, depth_img)

                # Update progress bar
                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

    testbed.shall_train = False
    testbed.nerf.cone_angle_constant = 0.0
    end = time.time()
    
    success, vertices = save_raw_density(testbed, 128, mesh_dir, frame_id)
    return success, vertices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Keypoint Detection')
    parser.add_argument("--ith", type=int, default=14)
    parser.add_argument("--gap", type=int, default=3)
    parser.add_argument("--session", type=str, default='2024-07-01-02-action-boxing')
    parser.add_argument("--scan_path", type=str, default='boxing/boxing-bag-stand-base/boxing-bag-stand-base-simplified.obj')
    parser.add_argument("--to_filter_views", action="store_true")
    parser.add_argument("--use_optim_params", action="store_true")
    parser.add_argument("--save_seg_video", action="store_true")
    parser.add_argument("--save_seg_frame", action="store_true")
    parser.add_argument("--undistort", action="store_true")
    parser.add_argument("--step1_only", action="store_true", default=False)
    parser.add_argument("--step2_only", action="store_true", default=False)
    parser.add_argument("--step3_only", action="store_true", default=False)
    parser.add_argument("--remove_bottom", action="store_true", default=False)
    parser.add_argument("--remove_side", action="store_true", default=True)
    parser.add_argument("--input_type", "-t", default="video", choices=["video", "image"], help="Whether the input is a video or set of images")
    parser.add_argument('--down', default=4, type=float, help='downsample image size')
    parser.add_argument('--camera_type', default='perspective', help='')
    parser.add_argument('--texture_type', default='ply', help='single|multi|ply')
    parser.add_argument('--shader', default='phong', help='phong|mask')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--normalize_verts', default=False, action='store_true')
    add_npg_parser(parser)
    args = parser.parse_args()

    try:
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except RuntimeError as e:
        if "No CUDA GPUs are available" in str(e):
            print("No CUDA GPUs available. Resubmitting job.")
            # Construct the sbatch command
            sbatch_command = f"sbatch ABATCH/{args.session}_{args.ith}.sh"
            
            # Submit the job
            os.system(sbatch_command)
            
            # Exit the current script to prevent further execution
            sys.exit(1)

    args.to_filter_views = True
    args.use_optim_params = True
    args.undistort = True
    
    args.step1 = True # segmentation
    args.step2 = True # reconstruction for T
    args.step3 = True # motion tracking

    if args.step1_only:
        args.step2 = args.step3 = False
    if args.step2_only:
        args.step1 = args.step3 = False
    if args.step3_only:
        args.step1 = args.step2 = False

    ith = args.ith
    gap = args.gap

    scene_name, object_name, _ = args.scan_path.split('/')
    if 'gopro' in object_name:
        args.remove_bottom = True
    if 'plant' in object_name:
        args.remove_bottom = True
    if 'salad' in object_name:
        args.remove_bottom = True
    object_main_path = '/users/rfu7/ssrinath/datasets/Action/brics-mini-objects'
    scanned_mesh_path = f"{object_main_path}/scans/{args.scan_path}"
    input_path = f'/users/rfu7/ssrinath/brics/non-pii/brics-mini/{args.session}'
    out_dir = f'/users/rfu7/ssrinath/datasets/Action/brics-mini/{args.session}'
    positive_prompt_path = f"{object_main_path}/scans/{scene_name}/{object_name}/positive_prompts.txt"
    with open(positive_prompt_path, "r") as file:
        positive_text_prompts = file.read().splitlines()
    negative_prompt_path = f"{object_main_path}/scans/{scene_name}/{object_name}/negative_prompts.txt"
    with open(negative_prompt_path, "r") as file:
        negative_text_prompts = file.read().splitlines()
    default_translation_path = f"{object_main_path}/scans/{scene_name}/{object_name}/default_translation.txt"
    with open(default_translation_path, 'r') as file:
        float_values = [float(line.strip()) for line in file]
    default_translation = np.asarray([float_values])
    save_main_path = os.path.join(object_main_path, object_name.replace(' ', '_').replace('-', '_'), args.session + '_' + str(ith).zfill(4))
    os.makedirs(save_main_path, exist_ok = True)
    save_segmentation_video_path = os.path.join(save_main_path, 'segmentation_' + positive_text_prompts[0].replace(' ', '_'))
    os.makedirs(save_segmentation_video_path, exist_ok = True)
    mesh_dir = os.path.join(save_main_path, "mesh", "ngp_mesh")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.step1:
        # Load DINO and SAM2
        # init sam image predictor and video predictor model
        sam2_checkpoint = "Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

        # init grounding dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-base"
        grounding_processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    if args.use_optim_params:
        params_txt = "optim_params.txt"
    else:
        params_txt = "params.txt"

    # Prepare for cameras and video readers
    params_path = os.path.join(out_dir, params_txt)
    params = param_utils.read_params(params_path)
    cam_names = list(params[:]["cam_name"])
    removed_camera_path = os.path.join(out_dir, 'ignore_camera.txt')
    if os.path.isfile(removed_camera_path):
        with open(removed_camera_path) as file:
            ignored_cameras = [line.rstrip() for line in file]
    else:
        ignored_cameras = None
    cams_to_remove = removed_cameras(remove_side=args.remove_side, remove_bottom=args.remove_bottom, ignored_cameras=ignored_cameras)

    for cam in cams_to_remove:
        if cam in cam_names:
            cam_names.remove(cam)
    cam_mapper = map_camera_names(input_path, cam_names)

    total_video_idxs = 0
    max_folder_id = 0
    for fid, folder in enumerate(os.listdir(input_path)):
        if 'cam' in folder and folder not in cams_to_remove:
            length = len([file for file in os.listdir(os.path.join(input_path, folder)) if file.endswith('.mp4')])
            if length > total_video_idxs:
                total_video_idxs = length
                max_folder_id = fid
                anchor_camera_by_length = os.listdir(input_path)[fid]

    intrs, projs, dist_intrs, dists, cameras = get_projections(args, params, cam_names, cam_mapper, easymocap_format=True)
    reader = Reader(args.input_type, input_path, cams_to_remove=cams_to_remove, ith=ith, anchor_camera=anchor_camera_by_length)
    
    extra_cams_to_remove = reader.to_delete
    for cam in extra_cams_to_remove:
        if cam in cam_names:
            cam_names.remove(cam)

    if args.to_filter_views:
        import open_clip
        open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
        open_clip_model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
        tokenized_text = tokenizer(positive_text_prompts + negative_text_prompts)
        text_features = open_clip_model.encode_text(tokenized_text)
    else:
        text_features = None

    cam_no_masks = []
    all_view_masks = {}
    all_view_images = {}
    if args.step1:
        # # Step 1: Track object in videos across views -> 51 x 1000 frames. 
        sam_text_prompt = '. '.join(positive_text_prompts)
        all_best_frame_ids = []
        for v_idx, input_video_path in tqdm(enumerate(reader.vids), total=len(reader.vids), desc="Segmenting across views"):
            camera_name = input_video_path.split('/')[-1].rpartition('_')[0]
            im_names, orig_imgs, im_h, im_w = frame_preprocess(input_video_path, args.undistort, intrs[v_idx], dist_intrs[v_idx], dists[v_idx])
            all_view_images[camera_name] = [img[:, :, ::-1] for img in orig_imgs]
            best_frame_id, all_masks = track_and_save_video(args, grounding_processor, grounding_model, video_predictor, f'{sam_text_prompt}.', text_features, len(positive_text_prompts), orig_imgs, save_folder=save_segmentation_video_path, view_name=camera_name, save_video=args.save_seg_video, save_frame=args.save_seg_frame)
            if best_frame_id is None:
                cam_no_masks.append(camera_name)
                print(f'No valid mask at view {camera_name}')
            else:
                all_best_frame_ids.append(best_frame_id)
            all_view_masks[camera_name] = all_masks
        the_best_frame_id = np.asarray(all_best_frame_ids).mean().astype(np.uint8)
        unique_best_frame_ids = np.unique(all_best_frame_ids)
        
        video_predictor.to('cpu')
        del video_predictor
        grounding_model = grounding_model.to('cpu')
        del grounding_model
        if args.to_filter_views:
            open_clip_model.to('cpu')
            del open_clip_model
        torch.cuda.empty_cache()
    else:
        # Read Template Mesh
        # Onboarding Multiview Templates
        # multi-view images
        for v_idx, input_video_path in tqdm(enumerate(reader.vids), total=len(reader.vids), desc="Loading segmentations across views"):
            camera_name = input_video_path.split('/')[-1].rpartition('_')[0]
            if os.path.exists(os.path.join(save_segmentation_video_path, camera_name)):
                orig_imgs = []
                all_masks = []
                for f_idx in range(reader.frame_count)[0::gap*10]:
                    img_path = os.path.join(save_segmentation_video_path, camera_name, str(f_idx).zfill(6) + '.png')
                    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    image = cv2.undistort(image, intrs[v_idx], dists[v_idx], None, dist_intrs[v_idx])
                    if args.to_filter_views and f_idx == 0:
                        is_to_filter = False
                        bgr_image = image[:, :, :3].copy()
                        rgb_image = bgr_image[:, :, ::-1]
                        mask = image[:, :, 3] > 0
                        rgb_image[~mask] = 0
                        preprocess_image = open_clip_preprocess(Image.fromarray(rgb_image)).unsqueeze(0)
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            image_features = open_clip_model.encode_image(preprocess_image)
                            text_features = open_clip_model.encode_text(tokenized_text)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                            max_prob, max_id = torch.max(text_probs, dim=-1)
                            if max_id.item() == 0:
                                orig_imgs.append(image[:, :, :3][:, :, ::-1])
                                all_masks.append(image[:, :, 3][np.newaxis, :, :] > 0)
                            else:
                                is_to_filter = True
                    else:
                        orig_imgs.append(image[:, :, :3][:, :, ::-1])
                        all_masks.append(image[:, :, 3][np.newaxis, :, :] > 0)                        
                if args.to_filter_views and is_to_filter:
                    print(f"Filtered {camera_name}")
                    cam_no_masks.append(camera_name)
                else:
                    all_view_images[camera_name] = orig_imgs
                    all_view_masks[camera_name] = all_masks
            else:
                cam_no_masks.append(camera_name)

    for cam in cam_no_masks:
        if cam in cam_names:
            cam_names.remove(cam)

    print(f'Total {len(cam_names)} available masks')
    if len(cam_names) == 0:
        print('NO VALID MASKS among all views.')
        args.step2 = False
        args.step3 = False

    # Step 2: Multi-view reconstruct the object
    with open(args.cam_faces_path, "r") as f:
        faces = json.load(f)

    intrs, extrs, dists = get_ngp_cameras(args, params, cam_names, cam_mapper, faces)

    # TODO: use best frame or max mask frame instead of the first frame.
    random_ranges = torch.tensor([0.1, 0.0, 0.1])
    if args.step2:
        translation_success = False
        sorted_frame_id_list = sort_frames_by_mask_area(all_view_masks, num_frames = reader.frame_count)

        for abs_idx, anchor_frame_id in enumerate(sorted_frame_id_list):
            print(f'The best frame id is {anchor_frame_id}')
            if True:
                translation_success, vertices = ngp_train(args, params, cam_names, cam_mapper, all_view_images, all_view_masks, intrs, extrs, dists, anchor_frame_id, mesh_dir=mesh_dir)
                if translation_success:
                    lower_bound = np.percentile(vertices, 5, axis=0)
                    upper_bound = np.percentile(vertices, 95, axis=0)
                    center_of_boundary = (lower_bound + upper_bound) / 2
                    vertices = center_of_boundary[np.newaxis, :].astype(np.float32)
                    break
            else:
                pass
        if not translation_success:
            vertices = default_translation
            anchor_frame_id = 0
    elif args.step3:
        anchor_frame_id = 0
        translation_success = False
        mesh_path= os.path.join(mesh_dir, 'volume_raw', f"{str(0).zfill(6)}_filtered_bounded_clutered_samples.ply")
        vertices, faces = load_ply(mesh_path)
        vertices = vertices.cpu().numpy()
        lower_bound = np.percentile(vertices, 5, axis=0)
        upper_bound = np.percentile(vertices, 95, axis=0)
        center_of_boundary = (lower_bound + upper_bound) / 2
        vertices = center_of_boundary[np.newaxis, :].astype(np.float32)

    # TODO: if less than 10% pixels contains the object. Then change T initialization.
    if args.step3:
        save_render_folder = os.path.join(save_main_path, 'render')
        os.makedirs(save_render_folder, exist_ok = True)
        save_pose_folder = os.path.join(save_main_path, 'pose')
        os.makedirs(save_pose_folder, exist_ok = True)
        save_transform_mesh_path = os.path.join(save_pose_folder, 'transform_mesh.obj')

        # Step 3: Multi-view optimize object pose
        extractor = dinov2_utils.DinoFeatureExtractor(model_name="dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1").to(device)
        
        """ initialize T from reconstruction"""
        print(vertices)
        mesh_translation = torch.Tensor(np.mean(vertices, axis=0)[np.newaxis, :].astype(np.float32))
        print(mesh_translation)
        """ load camera """
        render_cameras = colmap_utils.read_cameras_from_txt(params_path, cam_names, cam_mapper)

        """ load scanned mesh """
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            template_mesh = load_objs_as_meshes([scanned_mesh_path], device=device)



        num_retrieval_initializations = 100
        retrieval_rotation_matrices = uniform_quaternions(num_retrieval_initializations)
        # retrieval_rotation_matrices = quaternion_to_axis_angle(random_quats)

        """ Round0: initialize with NGP mean; Round1: initalize with center. """
        for init_round in range(2):
            """ load renders """
            renderer_list = batch_render_loader(args, render_cameras, device)
            
            if init_round == 1:
                translation_success = False
                mesh_translation = torch.Tensor(np.mean(default_translation, axis=0)[np.newaxis, :].astype(np.float32))
            
            templates_list = []
            """ initialize random views """
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                for mid, mesh_rotation in tqdm(enumerate(retrieval_rotation_matrices)):
                    # Render the image using the updated camera position. Based on the new position of the 
                    # camera we calculate the rotation and translation matrices
                    R = Rotate(quaternion_to_matrix(mesh_rotation.to(device)), orthogonal_tol=1e-3)
                    T = Translate(torch.clamp(mesh_translation.to(device), min=-0.2, max=1.2))   # (1, 3)
                    transform  = R.compose(T)
                    tverts = transform.transform_points(template_mesh.verts_list()[0])
                    faces = template_mesh.faces_list()[0]
                    tmesh = Meshes(
                        verts=[tverts],   
                        faces=[faces],
                        textures = template_mesh.textures,
                    )
                    
                    template_info = {'mesh_rotation': mesh_rotation.cpu().numpy().tolist()}
                    for rid, render_info in enumerate(renderer_list):
                        cam_name =render_info['cam_name']
                        image_tensor, fragments = render_info['renderer'](meshes_world=tmesh)
                        image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                        template_info[cam_name] = image_np
                    templates_list.append(template_info)

            """ Retrieval Comparison Across Views """
            all_init_scores = [0] * num_retrieval_initializations
            all_init_freq = [0] * num_retrieval_initializations
            all_init_dist = [0] * num_retrieval_initializations
            best_view_score = []
            for camera in render_cameras:
                anchor_cam_name = camera['cam_name']
                template_imgs = []
                for template_info in templates_list:
                    template_imgs.append(template_info[anchor_cam_name])

                valid_template_ids, feat_raw_projectors, feat_cluster_centroids, template_descs, feat_cluster_idfs = images_to_template_reps(template_imgs, extractor, device)
                if valid_template_ids is None:
                    continue

                instance_image_rgb = all_view_images[anchor_cam_name][anchor_frame_id]
                instance_image_mask = all_view_masks[anchor_cam_name][anchor_frame_id]
                if instance_image_rgb is None or instance_image_mask is None:
                    continue
                combined_mask = np.any(instance_image_mask, axis=0)  # Shape: [h, w]
                alpha_channel = np.where(combined_mask, 255, 0).astype(np.uint8)  # Opaque (255) or Transparent (0)
                instance_image = cv2.cvtColor(instance_image_rgb, cv2.COLOR_RGB2RGBA)
                instance_image[:, :, 3] = alpha_channel
                
                template_scores, sorted_template_ids, valid_template_ids = match_image_with_template_reps(instance_image, extractor, valid_template_ids, feat_raw_projectors, feat_cluster_centroids, template_descs, feat_cluster_idfs, device)
                for s, sid in zip(template_scores, sorted_template_ids):
                    if s > 0.2:
                        tid = valid_template_ids[sid]
                        all_init_scores[tid] += s
                        all_init_freq[tid] += 1


            all_init_scores = torch.tensor(all_init_scores)
            sorted_indices = torch.argsort(all_init_scores, descending=True)
            sorted_scores = all_init_scores[sorted_indices]


            # Number of initializations
            num_initializations = 15 if init_round == 1 else 10
            init_optimization_its = 1000 if init_round == 1 else 600
            final_optimization_its = 1000 if init_round == 1 else 600
            refine_optimization_its = 400
            init_lr = 2e-3
            final_lr = 1e-3
            refine_lr = 1e-3
            lambda_rgb = 0.0

            print("Sorted scores:", sorted_scores[:num_initializations])
            rotation_matrices = retrieval_rotation_matrices[sorted_indices[:num_initializations]]
            """ Optimize First Frame  """
            # Load the first frame
            image_ref_list, renderer_list, n_mask_pixel = batch_render_ref_loader(args, anchor_frame_id, render_cameras, all_view_images, all_view_masks, device)

            """ Find best R"""
            # Store best loss and best model state
            best_loss = float('inf')
            best_init_R = None
            best_model_state_dict = None
            best_it = 0
            best_iou = 0
            converge_loss = n_mask_pixel // 5000
            all_iou = np.zeros((num_initializations,))

            # Loop over all num_initializations
            for init_idx, init_R in enumerate(rotation_matrices):
                print(f"Optimizing initialization {init_idx} / {num_initializations}")
                opt_filename_output = f"{save_render_folder}/optimization_demo_r{init_idx}.mp4"

                # Initialize the model with the current anchor_T
                random_offset = (torch.rand_like(mesh_translation) * 2 - 1) * random_ranges
                model = DRModel(meshes=template_mesh, renderer_list=renderer_list, image_ref_list=image_ref_list, anchor_T=mesh_translation if translation_success else mesh_translation + random_offset, init_R=init_R, lambda_rgb=lambda_rgb).to(device)
                optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=init_lr)

                # Training loop for 100 epochs
                loop = tqdm(range(init_optimization_its))
                for i in loop:
                    with torch.autocast(device_type="cuda", dtype=torch.float32):
                        optimizer.zero_grad()
                        loss_pixels_list, image_list, _ = model(i)

                        loss = torch.sum(torch.stack(loss_pixels_list)) / len(loss_pixels_list)
                        loss.backward()
                        # Check gradients before clipping
                        has_grad_nan = False
                        for name, param in model.named_parameters():
                            if param.grad is not None:  # Ensure the parameter has a gradient
                                grad_norm = param.grad.norm().item()  # Calculate the norm of the gradient
                                if torch.isnan(param.grad).any():
                                    has_grad_nan = True
                                    print(f"NaN detected in gradient for parameter: {name}")
                        if has_grad_nan:
                            break
                        # torch.nn.utils.clip_grad_norm_([model.mesh_translation, model.mesh_rotation], max_norm=1.0)
                        optimizer.step()
                        model.mesh_rotation.data = torch.nn.functional.normalize(model.mesh_rotation.data, dim=0)

                        loop.set_description(f'Init {init_idx}: Optimizing (loss {loss.item():.4f})')

                        # Save outputs to create a GIF. 
                        if i % 10 == 0:
                            render_image = visualize_image_list(image_list, save_alpha=True)
                            ref_image = visualize_image_list(image_ref_list, save_alpha=True)
                            image = alpha_blend(render_image, ref_image)[..., :3]
                        
                        if i == 0:
                            writer = create_video_writer(opt_filename_output, (image.shape[1], image.shape[0]), fps=init_optimization_its//4)
                        writer.write(image)

                        if loss.item() < converge_loss:
                            break

                writer.release()
                convert_video_ffmpeg(opt_filename_output)

                """ Calculate IOU"""
                all_intersection = 0
                all_union = 0
                for img, ref in zip(image_list, image_ref_list):
                    img_mask = (img[0, :, :, 3] > 0).detach().squeeze().cpu().numpy()
                    ref_mask = ref[:, :, 3]
                    all_intersection += np.logical_and(img_mask, ref_mask).sum()
                    all_union += np.logical_or(img_mask, ref_mask).sum()

                iou = all_intersection / all_union if all_union != 0 else 0
                print(f"Init {init_idx}: IOU {iou:.4f}")
                all_iou[init_idx] = iou

                # Check if the current initialization yields a lower loss
                if loss.item() < best_loss and not check_for_nan_params(model):
                    best_it = init_idx
                    best_loss = loss.item()
                    best_init_R = init_R
                    best_model_state_dict = model.state_dict()  # Save the model's state dict for later use
                    best_iou = iou


            # After the initial optimization for all initializations, we continue with the best one
            print(f"Best initialization found with loss {best_loss:.4f} at iteration {best_it}")
            
            if np.any(all_iou > 0.45):
                break
            if np.all(all_iou > 0.10):
                break

        """ Optimize the anchor frame"""
        opt_filename_output = f"{save_render_folder}/optimization_demo_best.mp4"

        model = DRModel(meshes=template_mesh, renderer_list=renderer_list, image_ref_list=image_ref_list, anchor_T=mesh_translation, init_R=best_init_R, lambda_rgb=lambda_rgb).to(device)
        model.load_state_dict(best_model_state_dict)
        optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=final_lr)

        # Store best loss and best model state
        best_loss = float('inf')
        best_model_state_dict = None
        best_it = 0
        converge_loss = n_mask_pixel // 500

        loop = tqdm(range(final_optimization_its))
        for i in loop:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                optimizer.zero_grad()
                loss_pixels_list, image_list, tmesh = model(i)

                loss = torch.sum(torch.stack(loss_pixels_list)) / len(loss_pixels_list)
                loss.backward()
                # Check gradients before clipping
                has_grad_nan = False
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Ensure the parameter has a gradient
                        grad_norm = param.grad.norm().item()  # Calculate the norm of the gradient
                        if torch.isnan(param.grad).any():
                            has_grad_nan = True
                            print(f"NaN detected in gradient for parameter: {name}")
                if has_grad_nan:
                    break
                # torch.nn.utils.clip_grad_norm_([model.mesh_translation, model.mesh_rotation], max_norm=1.0)
                optimizer.step()
                model.mesh_rotation.data = torch.nn.functional.normalize(model.mesh_rotation.data, dim=0)
                
                loop.set_description(f'Refine Initialization Optimizing ({loss.data:.4f})' )

                # Check if the current initialization yields a lower loss
                if loss.item() < best_loss:
                    best_it = i
                    best_loss = loss.item()
                    best_model_state_dict = model.state_dict()  # Save the model's state dict for later use

                # Save outputs to create a GIF. 
                if i % 10 == 0:
                    render_image = visualize_image_list(image_list, save_alpha=True)
                    ref_image = visualize_image_list(image_ref_list, save_alpha=True)
                    image = alpha_blend(render_image, ref_image)[..., :3]

                if i == 0:
                    writer = create_video_writer(opt_filename_output, (image.shape[1], image.shape[0]), fps=final_optimization_its//4)
                writer.write(image)

                if loss.item() < converge_loss:
                    break

        writer.release()
        convert_video_ffmpeg(opt_filename_output)
        save_transformed_mesh(tmesh, template_mesh, save_transform_mesh_path)

        """ Optimize all frames"""
        """ find total frames"""

        opt_filename_output = f"{save_render_folder}/optimization_demo_all.mp4"
        video_writer_dict = {}
        pose_writer = f"{save_pose_folder}/optimized_pose.json"
        pose_dict = {}
        all_init_state_dict = best_model_state_dict.copy()

        ## >>Forward pose optimization
        for frame_id in range(anchor_frame_id, reader.frame_count)[0::gap]:
                
            image_ref_list, renderer_list, n_mask_pixel = batch_render_ref_loader(args, frame_id, render_cameras, all_view_images, all_view_masks, device)

            model = DRModel(meshes=template_mesh, renderer_list=renderer_list, image_ref_list=image_ref_list, anchor_T=best_model_state_dict['mesh_translation'], init_R=best_model_state_dict['mesh_rotation'], lambda_rgb=lambda_rgb).to(device)
            # model.load_state_dict(best_model_state_dict)
            model_init_state = model.state_dict()
            optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=refine_lr)
            
            # Store best loss and best model state
            best_loss = float('inf')
            # best_model_state_dict = None
            best_it = 0
            
            converge_loss = n_mask_pixel // 500
            loop = tqdm(range(refine_optimization_its))
            for i in loop:
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    optimizer.zero_grad()
                    loss_pixels_list, image_list, tmesh = model(i)

                    loss = torch.sum(torch.stack(loss_pixels_list)) / len(loss_pixels_list)
                    loss.backward()
                    # Check gradients before clipping
                    has_grad_nan = False
                    for name, param in model.named_parameters():
                        if param.grad is not None:  # Ensure the parameter has a gradient
                            grad_norm = param.grad.norm().item()  # Calculate the norm of the gradient
                            if torch.isnan(param.grad).any():
                                has_grad_nan = True
                                print(f"NaN detected in gradient for parameter: {name}")
                    if has_grad_nan:
                        break
                    # torch.nn.utils.clip_grad_norm_([model.mesh_translation, model.mesh_rotation], max_norm=1.0)
                    optimizer.step()
                    model.mesh_rotation.data = torch.nn.functional.normalize(model.mesh_rotation.data, dim=0)
                    
                    loop.set_description(f'Frame {frame_id}/{reader.frame_count} Optimizing {loss.item():.4f}')

                    has_nan = check_for_nan_params(model)
                    if has_nan:
                        model.load_state_dict(all_init_state_dict)
                        optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=refine_lr)

                    # Check if the current initialization yields a lower loss
                    if loss.item() < best_loss and not has_nan:
                        best_it = i
                        best_loss = loss.item()
                        best_model_state_dict = model.state_dict()  # Save the model's state dict for later use

                    # Save outputs to create a GIF. 
                    if i == refine_optimization_its - 1 or loss.item() < converge_loss and not has_nan:
                        render_image = visualize_image_list(image_list, save_alpha=True)
                        ref_image = visualize_image_list(image_ref_list, save_alpha=True)
                        image = alpha_blend(render_image, ref_image)[..., :3]
                        
                        
                        video_writer_dict[str(frame_id).zfill(6)] = image
                        mesh_translation_list = best_model_state_dict['mesh_translation'].cpu().numpy().tolist()  # Convert to NumPy, then to list
                        mesh_rotation_list = best_model_state_dict['mesh_rotation'].cpu().numpy().tolist()        # Convert to NumPy, then to list

                        pose_dict[str(frame_id).zfill(6)] = {
                            'mesh_translation': mesh_translation_list,
                            'mesh_rotation': mesh_rotation_list
                        }

                    if loss.item() < converge_loss and not has_nan:
                        break

        best_model_state_dict = all_init_state_dict.copy()
        ## >>backward pose optimization
        for frame_id in range(0, anchor_frame_id)[0::gap][::-1]:
                
            image_ref_list, renderer_list, n_mask_pixel = batch_render_ref_loader(args, frame_id, render_cameras, all_view_images, all_view_masks, device)

            model = DRModel(meshes=template_mesh, renderer_list=renderer_list, image_ref_list=image_ref_list, anchor_T=best_model_state_dict['mesh_translation'], init_R=best_model_state_dict['mesh_rotation'], lambda_rgb=lambda_rgb).to(device)
            # model.load_state_dict(best_model_state_dict)
            model_init_state = model.state_dict()
            optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=refine_lr)
            
            # Store best loss and best model state
            best_loss = float('inf')
            # best_model_state_dict = None
            best_it = 0
            
            converge_loss = n_mask_pixel // 500
            loop = tqdm(range(refine_optimization_its))
            for i in loop:
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    optimizer.zero_grad()
                    loss_pixels_list, image_list, tmesh = model(i)

                    loss = torch.sum(torch.stack(loss_pixels_list)) / len(loss_pixels_list)
                    loss.backward()
                    # Check gradients before clipping
                    has_grad_nan = False
                    for name, param in model.named_parameters():
                        if param.grad is not None:  # Ensure the parameter has a gradient
                            grad_norm = param.grad.norm().item()  # Calculate the norm of the gradient
                            if torch.isnan(param.grad).any():
                                has_grad_nan = True
                                print(f"NaN detected in gradient for parameter: {name}")
                    if has_grad_nan:
                        break
                    # torch.nn.utils.clip_grad_norm_([model.mesh_translation, model.mesh_rotation], max_norm=1.0)
                    optimizer.step()
                    model.mesh_rotation.data = torch.nn.functional.normalize(model.mesh_rotation.data, dim=0)
                    
                    loop.set_description(f'Frame {frame_id}/{reader.frame_count} Optimizing {loss.item():.4f}')

                    has_nan = check_for_nan_params(model)
                    if has_nan:
                        model.load_state_dict(model_init_state)
                        optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=refine_lr)

                    # Check if the current initialization yields a lower loss
                    if loss.item() < best_loss and not has_nan:
                        best_it = i
                        best_loss = loss.item()
                        best_model_state_dict = model.state_dict()  # Save the model's state dict for later use

                    # Save outputs to create a GIF. 
                    if i == refine_optimization_its - 1 or loss.item() < converge_loss and not has_nan:
                        render_image = visualize_image_list(image_list, save_alpha=True)
                        ref_image = visualize_image_list(image_ref_list, save_alpha=True)
                        image = alpha_blend(render_image, ref_image)[..., :3]
                        
                        
                        video_writer_dict[str(frame_id).zfill(6)] = image
                        mesh_translation_list = best_model_state_dict['mesh_translation'].cpu().numpy().tolist()  # Convert to NumPy, then to list
                        mesh_rotation_list = best_model_state_dict['mesh_rotation'].cpu().numpy().tolist()        # Convert to NumPy, then to list

                        pose_dict[str(frame_id).zfill(6)] = {
                            'mesh_translation': mesh_translation_list,
                            'mesh_rotation': mesh_rotation_list
                        }

                    if loss.item() < converge_loss and not has_nan:
                        break

        with open(pose_writer, 'w') as f:
            json.dump(pose_dict, f, indent=4)  # indent=4 makes the JSON file more readable

        sorted_video_writer_dict = dict(sorted(video_writer_dict.items(), key=lambda x: int(x[0])))
        writer = create_video_writer(opt_filename_output, (image.shape[1], image.shape[0]), fps=10)
        for fid, image in sorted_video_writer_dict.items():
            writer.write(image)
        writer.release()
        convert_video_ffmpeg(opt_filename_output)

        print(f"Data saved to {pose_writer}")

