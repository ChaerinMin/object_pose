#!/usr/bin/env python3

from typing import Optional, Tuple
import numpy as np
import torch

from src.utils import knn_util#, repre_util
from src.utils.feature_util import generate_grid_points, filter_points_by_mask, sample_feature_map_at_points
from src.utils.projector_util import PCAProjector
from src.utils.cluster_util import kmeans
from src.utils.knn_util import KNN

import cv2
import kornia
# logger: logging.Logger = logging.get_logger()


def crop_image(image_arr, mask_image_arr, pad_rate=0.2, crop_size = (420, 420)):
    # Step 1: Find the min and max coordinates of the non-zero mask
    coords = np.column_stack(np.where(mask_image_arr > 0))
    if coords.size == 0:
        return None, None
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)

    # Step 2: Calculate the center of the bounding box
    center_y = (ymin + ymax) // 2
    center_x = (xmin + xmax) // 2

    # Step 3: Find the new bounding box size (square bounding box)
    box_size = max(ymax - ymin, xmax - xmin)

    # Step 4: Calculate padding with pad_rate
    padded_box_size = int(box_size * (1 + pad_rate * 2))

    # Step 5: Determine the new bounding box coordinates (adjusted to image bounds)
    img_height, img_width = image_arr.shape[:2]

    new_ymin = max(0, center_y - padded_box_size // 2)
    new_ymax = min(img_height, center_y + padded_box_size // 2)
    new_xmin = max(0, center_x - padded_box_size // 2)
    new_xmax = min(img_width, center_x + padded_box_size // 2)

    # Step 6: Create a new square image with padding (black for transparent parts)
    padded_image = np.zeros((padded_box_size, padded_box_size, 3), dtype=np.uint8)
    padded_mask = np.zeros((padded_box_size, padded_box_size), dtype=np.uint8)
    if padded_image.size == 0:
        return None, None
    # Step 7: Determine the crop's actual dimensions
    crop_ymin = max(0, new_ymin)
    crop_ymax = min(img_height, new_ymax)
    crop_xmin = max(0, new_xmin)
    crop_xmax = min(img_width, new_xmax)

    # Step 8: Calculate the offset where to place the cropped image in the padded box
    y_offset = (padded_box_size - (crop_ymax - crop_ymin)) // 2
    x_offset = (padded_box_size - (crop_xmax - crop_xmin)) // 2

    # Step 9: Crop the image within the valid bounds
    cropped_image = image_arr[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    cropped_mask = mask_image_arr[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # Step 10: Insert the cropped image into the padded image
    padded_image[y_offset:y_offset + (crop_ymax - crop_ymin),
                 x_offset:x_offset + (crop_xmax - crop_xmin)] = cropped_image
    padded_mask[y_offset:y_offset + (crop_ymax - crop_ymin),
                 x_offset:x_offset + (crop_xmax - crop_xmin)] = cropped_mask
    
    # Step 11: Resize the image to crop_size
    final_image = cv2.resize(padded_image, crop_size, interpolation=cv2.INTER_AREA)
    final_mask = cv2.resize(padded_mask, crop_size, interpolation=cv2.INTER_AREA)
    return final_image, final_mask

def images_to_template_reps(template_img_paths, extractor, device):
    feat_vectors_list = []
    feat_to_template_ids_list = []
    valid_template_ids = []
    for tid, img_info in enumerate(template_img_paths):
        # Center Crop Image
        if isinstance(img_info, str):
            image_rgba = np.array(cv2.imread(img_info, cv2.IMREAD_UNCHANGED))
        elif isinstance(img_info, np.ndarray):
            image_rgba = img_info
        image_arr = image_rgba[:, :, :3]
        mask_image_arr = image_rgba[:, :, 3]
        if np.column_stack(np.where(mask_image_arr > 0)).size == 0:
            continue
        image_arr, mask_image_arr = crop_image(image_arr, mask_image_arr)
        if image_arr is None:
            continue
        image_chw = torch.Tensor(image_arr).to(torch.float32).permute(2,0,1).to(device) / 255.0
        object_mask = torch.Tensor(mask_image_arr).to(torch.float32).to(device) / 255.0

        device = image_chw.device

        # Generate grid points at which to sample feature vectors.
        grid_points = generate_grid_points(
            grid_size=(image_chw.shape[2], image_chw.shape[1]),
            cell_size=14.0,
        ).to(device)

        kernel = torch.ones(5, 5).to(device)
        object_mask_eroded = (
            kornia.morphology.erosion(
                object_mask.reshape(1, 1, *object_mask.shape).to(torch.float32), kernel
            )
            .squeeze([0, 1])
            .to(object_mask.dtype)
        )
        query_points = filter_points_by_mask(grid_points, object_mask_eroded)

        # Extract image feature
        image_bchw = image_chw.unsqueeze(0)
        extractor_output = extractor(image_bchw)
        feature_map_chw = extractor_output["feature_maps"][0]
        feature_map_chw = feature_map_chw.to(device)
        feat_vectors = sample_feature_map_at_points(
            feature_map_chw=feature_map_chw,
            points=query_points,
            image_size=(image_chw.shape[-1], image_chw.shape[-2]),
        ).detach()
        feat_to_template_ids = len(valid_template_ids) * torch.ones(
            feat_vectors.shape[0], dtype=torch.int32, device=device
        )
        feat_vectors_list.append(feat_vectors)
        feat_to_template_ids_list.append(feat_to_template_ids)
        valid_template_ids.append(tid)
    if len(valid_template_ids) == 0:
        return None, None, None, None, None
    feat_vectors = torch.cat(feat_vectors_list)
    feat_to_template_ids = torch.cat(feat_to_template_ids_list)

    # Transform the selected feature vectors to the PCA space.
    pca_projector = PCAProjector(
        n_components=256 if feat_vectors.shape[0] > 256 else feat_vectors.shape[0], whiten=False
    )
    pca_projector.fit(feat_vectors, max_samples=100000)
    feat_raw_projectors = pca_projector
    feat_vectors = pca_projector.transform(feat_vectors).cpu()

    # Clustering features into 2048 visual words...
    feat_cluster_centroids, feat_to_cluster_ids, centroid_distances = kmeans(
        samples=feat_vectors,
        num_centroids=256 if feat_vectors.shape[0] > 256 else feat_vectors.shape[0],
        verbose=True,
    )

    # Getting image descriptors...
    template_descs, feat_cluster_idfs = calc_tfidf_descriptors(
        feat_vectors=feat_vectors,
        feat_words=feat_cluster_centroids,
        feat_to_word_ids=feat_to_cluster_ids,
        feat_to_template_ids=feat_to_template_ids,
        num_templates=len(valid_template_ids),
        tfidf_knn_k=3,
        tfidf_soft_assign=False,
        tfidf_soft_sigma_squared=10.0,
    )

    return valid_template_ids, feat_raw_projectors, feat_cluster_centroids, template_descs, feat_cluster_idfs

def match_image_with_template_reps(img_info, extractor, valid_template_ids, feat_raw_projectors, feat_cluster_centroids, template_descs, feat_cluster_idfs, device):
    # Center Crop Image
    if isinstance(img_info, str):
        image_rgba = np.array(cv2.imread(img_info, cv2.IMREAD_UNCHANGED))
    elif isinstance(img_info, np.ndarray):
        image_rgba = img_info
    elif img_info is None:
        return [0], [-1], valid_template_ids 
    image_arr = image_rgba[:, :, :3]
    mask_image_arr = image_rgba[:, :, 3]
    if image_arr is None:
        return [0], [-1], valid_template_ids 
    image_arr, mask_image_arr = crop_image(image_arr, mask_image_arr)
    if image_arr is None:
        return [0], [-1], valid_template_ids 
    image_tensor_chw = torch.Tensor(image_arr).to(torch.float32).permute(2,0,1).to(device) / 255.0
    mask_modal_tensor = torch.Tensor(mask_image_arr).to(torch.float32).to(device) / 255.0

    # Extract Image Feature
    image_tensor_bchw = image_tensor_chw.unsqueeze(0)
    extractor_output = extractor(image_tensor_bchw)
    feature_map_chw = extractor_output["feature_maps"][0]

    grid_points = generate_grid_points(
        grid_size=(image_tensor_chw.shape[2], image_tensor_chw.shape[1]),
        cell_size=14.0,
    ).to(device)
    query_points = filter_points_by_mask(
        grid_points, mask_modal_tensor
    )
    if query_points.shape[0] > 1000000:
        perm = torch.randperm(query_points.shape[0])
        query_points = query_points[perm[: 1000000]]
    query_features = sample_feature_map_at_points(
        feature_map_chw=feature_map_chw,
        points=query_points,
        image_size=(image_tensor_chw.shape[2], image_tensor_chw.shape[1]),
    ).contiguous()

    # Transform the selected feature vectors to the PCA space.
    query_features_proj = feat_raw_projectors.transform(query_features).contiguous()
    _c, _h, _w = feature_map_chw.shape
    feature_map_chw_proj = feat_raw_projectors.transform(feature_map_chw.permute(1, 2, 0).view(-1, _c)).view(_h, _w, -1).permute(2, 0, 1)

    # Establish Correspondences
    faiss_use_gpu = False
    visual_words_knn_index = KNN(
        k=3,
        metric='l2',
        use_gpu= False
    )
    visual_words_knn_index.fit(feat_cluster_centroids.cpu())

    word_ids, word_dists = find_nearest_object_features(
        query_features=query_features_proj.cpu(),
        knn_index=visual_words_knn_index,
    )

    query_tfidf = calc_tfidf(
        feature_word_ids=word_ids.cpu(),
        feature_word_dists=word_dists.cpu(),
        word_idfs=feat_cluster_idfs.cpu(),
        soft_assignment=False,
        soft_sigma_squared=10.0,
    )

    match_feat_cos_sims = torch.nn.functional.cosine_similarity(template_descs.cpu(), query_tfidf.tile(template_descs.shape[0], 1).cpu())
    match_feat_cos_sims_cleaned = torch.nan_to_num(match_feat_cos_sims, nan=0.0)

    # Select templates with the highest cosine similarity.
    num_elements = match_feat_cos_sims_cleaned.numel()
    k = min(10, num_elements)
    if k == 0:
        return [0], [-1], valid_template_ids
    template_scores, sorted_template_ids = torch.topk(match_feat_cos_sims_cleaned, k=k, sorted=True)
    # best_view_score.append((anchor_cam_name, template_scores[0]))

    return template_scores, sorted_template_ids, valid_template_ids

    
def find_nearest_object_features(
    query_features: torch.Tensor,
    knn_index: knn_util.KNN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Find the nearest reference feature for each query feature.
    nn_dists, nn_ids = knn_index.search(query_features)
    knn_k = nn_dists.shape[1]

    # Keep only the required k nearest neighbors.
    nn_dists = nn_dists[:, :knn_k]
    nn_ids = nn_ids[:, :knn_k]

    # The distances returned by faiss are squared.
    nn_dists = torch.sqrt(nn_dists)

    return nn_ids, nn_dists

def calc_tfidf(
    feature_word_ids: torch.Tensor,
    feature_word_dists: torch.Tensor,
    word_idfs: torch.Tensor,
    soft_assignment: bool = True,
    soft_sigma_squared: float = 100.0,
) -> torch.Tensor:
    """Ref: https://www.di.ens.fr/~josef/publications/torii13.pdf"""

    device = feature_word_ids.device

    # Calculate soft-assignment weights, as in:
    # "Lost in Quantization: Improving Particular Object Retrieval in Large Scale Image Databases"
    if soft_assignment:
        word_weights = torch.exp(
            -torch.square(feature_word_dists) / (2.0 * soft_sigma_squared)
        )
    else:
        word_weights = torch.ones_like(feature_word_dists)

    # Normalize the weights such as they sum up to 1 for each query.
    word_weights = torch.nn.functional.normalize(word_weights, p=2, dim=1).reshape(-1)

    # Calculate term frequencies.
    # tf = word_weights  # https://www.cs.cmu.edu/~16385/s17/Slides/8.2_Bag_of_Visual_Words.pdf
    tf = word_weights / feature_word_ids.shape[0]  # From "Lost in Quantization".

    # Calculate inverse document frequencies.
    feature_word_ids_flat = feature_word_ids.reshape(-1)
    idf = word_idfs[feature_word_ids_flat]

    # Calculate tfidf values.
    tfidf = torch.multiply(tf, idf)

    # Construct the tfidf descriptor.
    num_words = word_idfs.shape[0]
    tfidf_desc = torch.zeros(
        num_words, dtype=word_weights.dtype, device=device
    ).scatter_add_(dim=0, index=feature_word_ids_flat.to(torch.int64), src=tfidf)

    return tfidf_desc


def calc_tfidf_descriptors(
    feat_vectors: torch.Tensor,
    feat_to_word_ids: torch.Tensor,
    feat_to_template_ids: torch.Tensor,
    feat_words: torch.Tensor,
    num_templates: int,
    tfidf_knn_k: int,
    tfidf_soft_assign: bool,
    tfidf_soft_sigma_squared: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate tf-idf descriptors.

    For each visual word i (i.e. cluster), idf is defined as log(N / N_i), where N is
    the number of images and N_i is the number of images in which visual word i appears.

    Ref: https://www.di.ens.fr/~josef/publications/torii13.pdf
    """

    device = feat_words.device
    feat_to_word_ids = feat_to_word_ids.to(device)
    feat_to_template_ids = feat_to_template_ids.to(device)
    feat_vectors = feat_vectors.to(device)

    # Calculate the idf terms (inverted document frequency).
    word_occurances = torch.zeros(len(feat_words), dtype=torch.int64, device=device)
    for template_id in range(num_templates):
        mask = feat_to_template_ids == template_id
        unique_word_ids = torch.unique(feat_to_word_ids[mask])
        word_occurances[unique_word_ids] += 1
    word_idfs = torch.log(
        torch.as_tensor(float(num_templates)) / word_occurances.to(torch.float32)
    )

    # Build a KNN index for the visual words.
    feat_knn_index = knn_util.KNN(k=tfidf_knn_k, metric="l2", use_gpu=True if device.type == "cuda" else False)
    feat_knn_index.fit(feat_words)

    # Calculate the tf-idf descriptor for each template.
    tfidf_descs = []
    for template_id in range(num_templates):
        tpl_mask = feat_to_template_ids == template_id
        word_dists, word_ids = feat_knn_index.search(feat_vectors[tpl_mask])
        tfidf = calc_tfidf(
            feature_word_ids=word_ids,
            feature_word_dists=word_dists,
            word_idfs=word_idfs,
            soft_assignment=tfidf_soft_assign,
            soft_sigma_squared=tfidf_soft_sigma_squared,
        )
        tfidf_descs.append(tfidf)
    tfidf_descs = torch.stack(tfidf_descs, dim=0)

    return tfidf_descs, word_idfs

