import nibabel as nib
import torch
import numpy as np


def load_nifti(file_path: str) -> (torch.Tensor, np.ndarray, nib.nifti1.Nifti1Header):
    nifti_img = nib.load(file_path)
    img_data = nifti_img.get_fdata()
    tensor_data = torch.tensor(img_data, dtype=torch.float32)
    return tensor_data, nifti_img.affine, nifti_img.header

def save_nifti(tensor: torch.Tensor, affine: np.ndarray, header: nib.nifti1.Nifti1Header, output_path: str) -> None:
    if tensor.is_cuda:
        tensor = tensor.cpu()
    img_data = tensor.numpy()
    new_nifti_img = nib.Nifti1Image(img_data, affine, header)
    nib.save(new_nifti_img, output_path)
