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
    img_hwc = _to_hwc_uint8(img)

    locations = _safe_face_locations(img_hwc)

    # For task1 data, color order can sometimes be inconsistent depending on loading pipeline.
    if len(locations) == 0:
        locations = _safe_face_locations(torch.flip(img_hwc, dims=(2,)))

    H, W, _ = img_hwc.shape

    for top, right, bottom, left in locations:
        left = max(0, min(int(left), W))
        top = max(0, min(int(top), H))
        right = max(0, min(int(right), W))
        bottom = max(0, min(int(bottom), H))

        detection_results.append([
            float(left),
            float(top),
            float(max(0, right - left)),
            float(max(0, bottom - top))
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
    features: List[torch.Tensor] = []

    for img_name in img_names:
        img_hwc = _to_hwc_uint8(imgs[img_name])

        detections = detect_faces(img_hwc)

        if len(detections) == 0:
            feature = _image_feature(img_hwc)
        else:
            best_box = detections[0]
            best_area = best_box[2] * best_box[3]

            for box in detections:
                area = box[2] * box[3]
                if area > best_area:
                    best_area = area
                    best_box = box

            face_crop = _crop_box(img_hwc, best_box)
            feature = _image_feature(face_crop)

        features.append(_l2_normalize(feature))

    feature_tensor = torch.stack(features, dim=0)

    if feature_tensor.shape[0] <= K:
        for i, img_name in enumerate(img_names):
            cluster_results[i].append(img_name)
        return cluster_results

    assignments = _kmeans(feature_tensor, K)

    for img_name, cluster_idx in zip(img_names, assignments.tolist()):
        cluster_results[int(cluster_idx)].append(img_name)
    
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)

def _to_hwc_uint8(img: torch.Tensor) -> torch.Tensor:
    x = img.detach().cpu()

    if x.dim() != 3:
        raise ValueError("Input image must be a 3D tensor")

    # Support both CHW and HWC
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = x.permute(1, 2, 0)
    elif x.shape[-1] == 3:
        pass
    else:
        raise ValueError("Input image must have 3 channels")

    if x.dtype != torch.uint8:
        x = x.clamp(0, 255).to(torch.uint8)

    return x.contiguous()


def _safe_face_locations(img_hwc: torch.Tensor):
    try:
        return face_recognition.face_locations(img_hwc.numpy(), model="hog")
    except Exception:
        return []


def _crop_box(img_hwc: torch.Tensor, box: List[float]) -> torch.Tensor:
    H, W, _ = img_hwc.shape

    x, y, w, h = box

    left = max(0, min(int(x), W - 1))
    top = max(0, min(int(y), H - 1))
    right = max(left + 1, min(int(x + w), W))
    bottom = max(top + 1, min(int(y + h), H))

    crop = img_hwc[top:bottom, left:right, :]

    if crop.numel() == 0:
        return img_hwc

    return crop.contiguous()


def _image_feature(img_hwc: torch.Tensor) -> torch.Tensor:
    x = img_hwc.to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1 x 3 x H x W

    pooled_8 = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8)).reshape(-1)
    pooled_4 = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4)).reshape(-1)

    feature = torch.cat([pooled_8, pooled_4], dim=0)

    if feature.numel() >= 128:
        feature = feature[:128]
    else:
        padded = torch.zeros(128, dtype=torch.float32)
        padded[:feature.numel()] = feature
        feature = padded

    return feature


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x, p=2)
    if norm.item() < 1e-12:
        return x.clone()
    return x / norm


def _kmeans(data: torch.Tensor, K: int, max_iters: int = 100) -> torch.Tensor:
    N = data.shape[0]

    centroids = data[:K].clone()
    assignments = torch.full((N,), -1, dtype=torch.long)

    for _ in range(max_iters):
        distances = torch.cdist(data, centroids, p=2)
        new_assignments = torch.argmin(distances, dim=1)

        if torch.equal(new_assignments, assignments):
            break

        assignments = new_assignments
        new_centroids = []

        for k in range(K):
            members = data[assignments == k]
            if members.shape[0] == 0:
                farthest_idx = int(torch.argmax(torch.min(distances, dim=1).values).item())
                new_centroids.append(data[farthest_idx])
            else:
                new_centroids.append(torch.mean(members, dim=0))

        centroids = torch.stack(new_centroids, dim=0)

    return assignments