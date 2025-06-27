import faiss
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import numpy as np
from scipy.stats import mode

def multilabel_patching(batch_array, k, similarity_type=None, num_classes=None):
    batch_size, rows, cols = batch_array.shape

    # Determine split sizes for rows and columns
    row_sizes = [(rows * (i + 1) // k) - (rows * i // k) for i in range(k)]
    col_sizes = [(cols * (i + 1) // k) - (cols * i // k) for i in range(k)]

    # Prepare arrays for splitting
    row_splits = np.cumsum([0] + row_sizes)
    col_splits = np.cumsum([0] + col_sizes)

    # Create a new batch array for reshaped patches
    if similarity_type == 'multilabel':
        reduced_batch = np.zeros((batch_size, k, k))
    elif similarity_type == 'distribution':
        reduced_batch = np.zeros((batch_size, k, k, num_classes))

    for i in range(k):
        for j in range(k):
            # Extract patches based on computed splits
            patch = batch_array[:, row_splits[i]:row_splits[i + 1], col_splits[j]:col_splits[j + 1]]
            # Compute the mode for each patch
            if similarity_type == 'multilabel':
                patch_mode = mode(patch.reshape(batch_size, -1), axis=1).mode.squeeze()
                reduced_batch[:,i,j] = patch_mode
            elif similarity_type == 'distribution':
                patch_mode = distribution_of_classes(patch, num_classes)
                reduced_batch[:,i,j] = patch_mode
            else:
                raise NotImplementedError(f"Similarity type {similarity_type} not implemented.")

    return reduced_batch

def hellinger_distance(P, Q):
    return np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2,axis=-1))

def distribution_of_classes(array, num_classes):
    reshaped_array = array.reshape(array.shape[0], -1)    
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_classes), axis=1, arr=reshaped_array)
    distribution = counts / reshaped_array.shape[1]
    return distribution

def closest_representations(features, targets, mode="faiss" ,device='cuda'):
    if mode=="matmul":
        # Expects tensors as inputs
        features = F.normalize(features, dim=-1)
        closest_idxes = features.matmul(features.transpose(-2, -1)).topk(2)[1][:,1]
        closest_classes = targets[closest_idxes]
    elif mode=="faiss":
        # For big data, use faiss. Expects numpy arrays with float32 as inputs
        features = normalize(features, axis=1)
        faiss_search_index = faiss.IndexFlatIP(features.shape[-1])
        if 'cuda' in device:
            res = faiss.StandardGpuResources()
            faiss_search_index = faiss.index_cpu_to_gpu(res,0,faiss_search_index)
        faiss_search_index.add(features)
        _, closest_idxes = faiss_search_index.search(features, 2)  # use 2, because the closest one will be the point itself
        closest_idxes = closest_idxes[:, 1]
        closest_classes = targets[closest_idxes]
    else:
        raise NotImplementedError(f"mode {mode} not implemented.")
    return closest_classes

def recall_at_one(closest_classes, targets, type='multilabel', num_classes=1, recall_criterion='half', num_patches=3):
    if type == 'multilabel':
        ones_targets = np.where(targets == 1, targets, np.nan)
        ones_closest = np.where(targets == 1, closest_classes, np.nan)
        same_classes = (ones_targets == ones_closest).astype("float").sum(axis=-1)
        if recall_criterion == 'one':
            is_same_class = (same_classes >= 1).astype(float)
        elif recall_criterion == 'distance':
            all_classes = np.nan_to_num(ones_targets, 0).sum(axis=-1)
            is_same_class = (same_classes / all_classes).astype(float)
            nan_indices = np.isnan(is_same_class)
            is_same_class[nan_indices] = 1
        elif recall_criterion == 'all':
            all_classes = np.nan_to_num(ones_targets, 0).sum(axis=-1)
            is_same_class = (same_classes / all_classes == 1.0).astype(float)
        else: 
            raise NotImplementedError(f"Criterion {recall_criterion} not implemented.")
    elif type == 'segmentation':
        closest_dist_classes = distribution_of_classes(closest_classes, num_classes)
        original_dist_classes = distribution_of_classes(targets, num_classes)
        if recall_criterion == 'distance':
            distance = hellinger_distance(closest_dist_classes, original_dist_classes)
            is_same_class = (distance < 0.1).astype(float)
        elif recall_criterion == 'distribution_of_classes':
            is_same_class = 1 - hellinger_distance(closest_dist_classes, original_dist_classes)
        elif recall_criterion == 'patching':
            targets_ = multilabel_patching(targets, k=num_patches, similarity_type='multilabel', num_classes=num_classes).reshape(targets.shape[0],-1)
            closest_classes_ = multilabel_patching(closest_classes, k=num_patches, similarity_type='multilabel', num_classes=num_classes).reshape(targets.shape[0],-1)
            is_same_class = np.all(targets_ == closest_classes_, axis=1)
        elif recall_criterion == 'distribution_of_classes_with_patching':
            targets_ = multilabel_patching(targets, k=num_patches, similarity_type='distribution', num_classes=num_classes).reshape(targets.shape[0], num_patches**2, num_classes)
            closest_classes_ = multilabel_patching(closest_classes, k=num_patches, similarity_type='distribution', num_classes=num_classes).reshape(targets.shape[0], num_patches**2, num_classes)
            is_same_class = 1 - hellinger_distance(targets_,closest_classes_).mean(axis=-1)
        elif recall_criterion == 'multilabel':
            ones_targets = np.where(original_dist_classes > 0, original_dist_classes, np.nan)
            ones_closest = np.where(original_dist_classes > 0, closest_dist_classes, np.nan)
            same_classes = (ones_targets == ones_closest).astype("float").sum(axis=-1)
            all_classes = np.nan_to_num(ones_targets, 0).sum(axis=-1)
            is_same_class = (same_classes / all_classes == 1.0).astype(float)
        # is_same_class = (distance < 0.5).astype(float)
        else:
            NotImplementedError(f"Recall {recall_criterion} not implemented.") 
    elif type == 'onelabel':
        is_same_class = (closest_classes == targets).astype("float")
    else:
        NotImplementedError(f"Training type {type} not implemented.") 
    return is_same_class.mean(), is_same_class

