'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####
    # show_image(img) 
    locations = detect_face_loc_robust(img)
    if len(locations) == 0:
        return detection_results

    img_to_hwc = convert_to_hwc_uint8(img)
    height = int(img_to_hwc.shape[0])
    width = int(img_to_hwc.shape[1])

    boxes_xywh = []
    for top, right, bottom, left in locations:
        x = float(left)
        y = float(top)
        w = float(max(0, right - left))
        h = float(max(0, bottom - top))
        boxes_xywh.append([x, y, w, h])

    box_tensor = torch.tensor(boxes_xywh, dtype=torch.float32)
    if box_tensor.numel() == 0:
        return detection_results

    box_tensor[:, 0] = torch.clamp(box_tensor[:, 0], min=0.0, max=float(max(width - 1, 0)))
    box_tensor[:, 1] = torch.clamp(box_tensor[:, 1], min=0.0, max=float(max(height - 1, 0)))
    box_tensor[:, 2] = torch.clamp(box_tensor[:, 2], min=0.0)
    box_tensor[:, 3] = torch.clamp(box_tensor[:, 3], min=0.0)
    box_tensor[:, 2] = torch.minimum(box_tensor[:, 2], float(width) - box_tensor[:, 0])
    box_tensor[:, 3] = torch.minimum(box_tensor[:, 3], float(height) - box_tensor[:, 1])

    for i in range(box_tensor.shape[0]):
        detection_results.append([
            float(box_tensor[i, 0].item()),
            float(box_tensor[i, 1].item()),
            float(box_tensor[i, 2].item()),
            float(box_tensor[i, 3].item())
        ])

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    if K <= 0 or len(imgs) == 0:
        return cluster_results

    img_names = sorted(list(imgs.keys()))
    embeddings = []

    for img_name in img_names:
        # show_image(imgs[img_name]) 
        embedding = extract_face_embed(imgs[img_name])
        embeddings.append(embedding)

    points = torch.stack(embeddings, dim=0)
    assignments = kmeans_assignments(points, K)

    for img_name, cluster_id in zip(img_names, assignments.tolist()):
        if 0 <= int(cluster_id) < K:
            cluster_results[int(cluster_id)].append(img_name)

    for cluster in cluster_results:
        cluster.sort()
    
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)
def convert_to_hwc_uint8(img: torch.Tensor) -> torch.Tensor:
    if img.dim() != 3:
        raise ValueError("Expected an image tensor with 3 dimensions.")

    if img.shape[0] == 3 and img.shape[-1] != 3:
        img_hwc = img.permute(1, 2, 0)
    elif img.shape[-1] == 3:
        img_hwc = img
    else:
        raise ValueError("Expected image tensor to have 3 channels.")

    if img_hwc.dtype != torch.uint8:
        img_hwc = torch.clamp(img_hwc, 0, 255).to(torch.uint8)

    return img_hwc.contiguous().cpu()

def flip_channels_hwc(img_hwc: torch.Tensor) -> torch.Tensor:
    return torch.flip(img_hwc, dims=(2,))

def get_box_area(location: tuple) -> float:
    top, right, bottom, left = location
    return float(max(0, right - left) * max(0, bottom - top))

def safe_face_loc(img_hwc: torch.Tensor, upsample: int) -> List[tuple]:
    try:
        return face_recognition.face_locations(
            img_hwc.numpy(),
            number_of_times_to_upsample=upsample,
            model="hog"
        )
    except Exception:
        return []
    
def do_safe_face_encod(img_hwc: torch.Tensor, location: tuple) -> List[List[float]]:
    try:
        return face_recognition.face_encodings(img_hwc.numpy(), [location])
    except Exception:
        return []
    
def detect_face_loc_robust(img: torch.Tensor) -> List[tuple]:
    img_hwc = convert_to_hwc_uint8(img)
    candidates = [img_hwc, flip_channels_hwc(img_hwc)]

    best_locations: List[tuple] = []
    best_score = -1.0

    for candidate in candidates:
        for upsample in [1, 2, 0]:
            locations = safe_face_loc(candidate, upsample)
            if len(locations) == 0:
                continue

            score = 0.0
            for location in locations:
                score += get_box_area(location)

            if score > best_score:
                best_locations = locations
                best_score = score

    return best_locations

def get_largest_loc(locations: List[tuple]) -> tuple:
    best_location = locations[0]
    best_area = get_box_area(best_location)
    for location in locations[1:]:
        area = get_box_area(location)
        if area > best_area:
            best_area = area
            best_location = location

    return best_location

def simple_fallback_embed(img_hwc: torch.Tensor) -> torch.Tensor:
    img_chw = img_hwc.permute(2, 0, 1).to(torch.float32) / 255.0
    pooled = torch.nn.functional.adaptive_avg_pool2d(img_chw.unsqueeze(0), (8, 8)).reshape(-1)

    if pooled.numel() >= 128:
        return pooled[:128].to(torch.float32)

    embedding = torch.zeros(128, dtype=torch.float32)
    embedding[:pooled.numel()] = pooled
    return embedding

def extract_face_embed(img: torch.Tensor) -> torch.Tensor:
    img_hwc = convert_to_hwc_uint8(img)
    candidate_images = [img_hwc, flip_channels_hwc(img_hwc)]

    for candidate in candidate_images:
        locations = []
        for upsample in [1, 2, 0]:
            locations = safe_face_loc(candidate, upsample)
            if len(locations) > 0:
                break

        if len(locations) == 0:
            continue

        best_location = get_largest_loc(locations)
        encodings = do_safe_face_encod(candidate, best_location)
        if len(encodings) > 0:
            return torch.tensor(encodings[0], dtype=torch.float32)

        try:
            fallback_encodings = face_recognition.face_encodings(candidate.numpy())
            if len(fallback_encodings) > 0:
                return torch.tensor(fallback_encodings[0], dtype=torch.float32)
        except Exception:
            pass

    # Rare fallback if face_recognition fails on an image.
    return simple_fallback_embed(img_hwc)

def init_centroids_farthest_frst(points: torch.Tensor, K: int) -> torch.Tensor:
    num_points = points.shape[0]
    num_centroids = min(K, num_points)

    chosen_indices = [0]
    min_dists = torch.sum((points - points[0:1]) ** 2, dim=1)

    while len(chosen_indices) < num_centroids:
        min_dists[torch.tensor(chosen_indices, dtype=torch.long)] = -1.0
        next_index = int(torch.argmax(min_dists).item())
        chosen_indices.append(next_index)
        new_dists = torch.sum((points - points[next_index:next_index + 1]) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, new_dists)

    centroids = points[torch.tensor(chosen_indices, dtype=torch.long)].clone()

    if K > num_points:
        padding = points[torch.zeros(K - num_points, dtype=torch.long)].clone()
        centroids = torch.cat([centroids, padding], dim=0)

    return centroids

def kmeans_assignments(points: torch.Tensor, K: int, max_iters: int = 50) -> torch.Tensor:
    num_points = points.shape[0]
    if num_points == 0:
        return torch.empty((0,), dtype=torch.long)

    centroids = init_centroids_farthest_frst(points, K)
    prev_assignments = None

    for _ in range(max_iters):
        distances = torch.cdist(points, centroids, p=2)
        assignments = torch.argmin(distances, dim=1)

        if prev_assignments is not None and torch.equal(assignments, prev_assignments):
            break

        new_centroids = centroids.clone()
        min_distances = torch.min(distances, dim=1).values

        for cluster_id in range(K):
            mask = assignments == cluster_id
            if torch.any(mask):
                new_centroids[cluster_id] = points[mask].mean(dim=0)
            else:
                farthest_index = int(torch.argmax(min_distances).item())
                new_centroids[cluster_id] = points[farthest_index]
                min_distances[farthest_index] = -1.0

        centroid_shift = torch.max(torch.abs(new_centroids - centroids))
        centroids = new_centroids
        prev_assignments = assignments

        if float(centroid_shift.item()) < 1e-5:
            break

    final_distances = torch.cdist(points, centroids, p=2)
    return torch.argmin(final_distances, dim=1)
