"""
CRC Dataset - Main dataset class for loading Zarr data
Handles multi-modal imaging (H&E + MIF) with instance segmentation
Supports multiple cancer types (CRC, Xenium, TissueNet, PanNuke, Lizard, MoNuSeg, TNBC)
"""

import os
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import pickle
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import psutil

from .hv_generator import batch_generate_hv_maps
from .augmentation import MedicalImageAugmentation


class CRCZarrDataset(Dataset):
    """
    Multi-cancer Dataset loader for Zarr format data
    
    Features:
    - Loads H&E (3 channels) + MIF (2 channels) images
    - Handles 4 mask types: HE nuclei, HE cells, MIF nuclei, MIF cells
    - Generates 4 HV maps using PanNuke method
    - Supports multiple cancer types (CRC, Xenium, TissueNet, PanNuke, Lizard, MoNuSeg, TNBC)
    - Optional memory caching for speed
    - Data augmentation support
    """
    
    def __init__(self, samples: List[str], config, training: bool = True):
        """
        Initialize dataset
        
        Args:
            samples: List of sample names (e.g., ['CRC01', 'breast', 'train', 'Fold1', 'lizard_train', 'monuseg_train', 'tnbc_train'])
            config: Config object with all settings
            training: Whether this is training set (enables augmentation)
        """
        self.samples = samples
        self.config = config
        self.training = training
        self.zarr_bases = config.zarr_bases
        
        self.patch_info = []
        self._build_patch_index()
        
        if training and config.train_augment:
            aug_config = config.get_augmentation_config()
            self.augment = MedicalImageAugmentation(aug_config)
        else:
            self.augment = None
        
        if config.verbose:
            print(f"{'Training' if training else 'Validation/Test'} Dataset: "
                  f"{len(self.samples)} samples, {len(self.patch_info)} patches")
    
    def _get_zarr_path(self, sample_name: str) -> str:
            """Get zarr path for a given sample based on naming convention"""
            # --- FIX: Check for explicit dataset prefixes first ---
            if sample_name.startswith('tissuenet'):
                return self.zarr_bases['tissuenet']
            elif sample_name.startswith('pannuke'):  # Handles pannuke_train, pannuke_val
                return self.zarr_bases['pannuke']
            elif sample_name.startswith('lizard'):
                return self.zarr_bases['lizard']
            elif sample_name.startswith('monuseg'):
                return self.zarr_bases['monuseg']
            elif sample_name.startswith('monusac'):
                return self.zarr_bases['monusac']
            elif sample_name.startswith('tnbc'):
                return self.zarr_bases['tnbc']
            elif sample_name.startswith('nuinsseg'):
                return self.zarr_bases['nuinsseg']
            elif sample_name.startswith('cryonuseg'):
                return self.zarr_bases['cryonuseg']
            elif sample_name.startswith('bc'):
                return self.zarr_bases['bc']
            elif sample_name.startswith('consep'):
                return self.zarr_bases['consep']
            elif sample_name.startswith('kumar'):
                return self.zarr_bases['kumar']
            elif sample_name.startswith('cpm17'):
                return self.zarr_bases['cpm17']
            elif sample_name.startswith('CRC'):
                return self.zarr_bases['crc']
            elif sample_name.startswith('cpm15'):
                return self.zarr_bases['cpm15']
            elif sample_name.startswith('he_dsb2018'):
                return self.zarr_bases['he_dsb2018']
            elif sample_name.startswith('fluorescence_dsb2018'):
                return self.zarr_bases['fluorescence_dsb2018']
            elif sample_name.startswith('panoptils'):
                return self.zarr_bases['panoptils']
            
            # --- Legacy checks (keep just in case, but prioritize above) ---
            elif sample_name in ['train', 'val', 'test']: 
                return self.zarr_bases['tissuenet']
            elif sample_name.startswith('Fold'):
                return self.zarr_bases['pannuke']
                
            # Default to Xenium
            else:
                return self.zarr_bases.get('xenium', list(self.zarr_bases.values())[0])
    def _get_dataset_type(self, sample_name: str) -> str:
            """Helper to identify dataset source"""
            if sample_name.startswith('tissuenet') or sample_name in ['train', 'val', 'test']:
                return 'tissuenet'
            elif sample_name.startswith('pannuke') or sample_name.startswith('Fold'):
                return 'pannuke'
            elif sample_name.startswith('lizard'):
                return 'lizard'
            elif sample_name.startswith('monuseg'):
                return 'monuseg'
            elif sample_name.startswith('monusac'):
                return 'monusac'
            elif sample_name.startswith('tnbc'):
                return 'tnbc'
            elif sample_name.startswith('nuinsseg'):
                return 'nuinsseg'
            elif sample_name.startswith('cryonuseg'):
                return 'cryonuseg'
            elif sample_name.startswith('bc'):
                return 'bc'
            elif sample_name.startswith('consep'):
                return 'consep'
            elif sample_name.startswith('kumar'):
                return 'kumar'
            elif sample_name.startswith('cpm17'):
                return 'cpm17'
            elif sample_name.startswith('CRC'):
                return 'crc'
            elif sample_name.startswith('cpm15'):
                return 'cpm15'
            elif sample_name.startswith('he_dsb2018'):
                return 'he_dsb2018'
            elif sample_name.startswith('fluorescence_dsb2018'):
                return 'fluorescence_dsb2018'
            elif sample_name.startswith('panoptils'):
                return 'panoptils'
            return 'xenium'
    
    def _build_patch_index(self):
        """Build index of all patches across samples"""
        if self.config.verbose:
            print(f"📊 Building patch index for {len(self.samples)} samples...")
        
        for sample in self.samples:
            zarr_path = self._get_zarr_path(sample)
            dataset_type = self._get_dataset_type(sample)
            
            # ========================================================================
            # DATASETS WITH SUBDIRECTORIES: TissueNet, PanNuke, MoNuSeg, TNBC, MoNuSAC
            # ========================================================================
            if dataset_type in ['tissuenet', 'pannuke', 'monuseg', 'tnbc', 'monusac', 'cpm15', 'he_dsb2018','fluorescence_dsb2018', 'panoptils']:
                split_path = os.path.join(zarr_path, sample)
                
                if not os.path.exists(split_path):
                    if os.path.basename(zarr_path) == sample:
                        split_path = zarr_path
                    else:
                        print(f"  ⚠️  {dataset_type} split/fold not found at: {split_path}")
                        continue
                
                try:
                    sub_samples = sorted([
                        d for d in os.listdir(split_path) 
                        if os.path.isdir(os.path.join(split_path, d))
                    ])
                    
                    if not sub_samples:
                        print(f"  ⚠️  {dataset_type}: No sub-folders found in {split_path}")
                        continue

                    total_patches = 0
                    for sub in sub_samples:
                        if dataset_type in ['monuseg', 'tnbc', 'monusac', 'cpm15', 'he_dsb2018','fluorescence_dsb2018', 'panoptils']:
                            zarr_file = os.path.join(split_path, sub, 'images.zarr')
                            mask_file = os.path.join(split_path, sub, 'nuclei_masks.zarr')
                            
                            # CHECK: Only proceed if both Zarr files exist
                            if os.path.exists(zarr_file) and os.path.exists(mask_file):
                                try:
                                    z = zarr.open(zarr_file, 'r')
                                    n_patches = z.shape[0]
                                    for patch_idx in range(n_patches):
                                        self.patch_info.append((sample, sub, patch_idx))
                                    total_patches += n_patches
                                except Exception as e:
                                    # Silently skip corrupt/empty zarrs
                                    pass
                        else:
                            self.patch_info.append((sample, sub, 0))
                            total_patches += 1
                        
                    if self.config.verbose:
                        print(f"  ✅ {dataset_type} {sample}: {total_patches} patches loaded")
                except Exception as e:
                    print(f"  ⚠️  {dataset_type} error reading {split_path}: {e}")
            
            # ========================================================================
            # DATASETS WITH FLAT STRUCTURE: Lizard, NuInsSeg
            # ========================================================================
            elif dataset_type in ['lizard', 'nuinsseg', 'cryonuseg', 'bc', 'consep', 'kumar', 'cpm17']:
                base_folder = os.path.join(zarr_path, sample)
                
                try:
                    zarr_file = os.path.join(base_folder, 'images.zarr')
                    if not os.path.exists(zarr_file):
                        print(f"  ⚠️  {sample}: Zarr not found at {zarr_file}")
                        continue

                    z = zarr.open(zarr_file, 'r')
                    n_patches = z.shape[0]
                    for patch_idx in range(n_patches):
                        self.patch_info.append((sample, None, patch_idx))
                    
                    if self.config.verbose:
                        print(f"  ✅ {sample} ({dataset_type}): {n_patches} patches")
                except Exception as e:
                    print(f"  ⚠️  {sample}: Could not load - {e}")
            
            # ========================================================================
            # CRC / XENIUM (with he/ and mif/ subdirectories)
            # ========================================================================
            else:
                he_zarr_path = os.path.join(zarr_path, sample, 'he', 'images.zarr')
                
                if not os.path.exists(he_zarr_path):
                    he_zarr_path = os.path.join(zarr_path, sample, 'images.zarr')

                try:
                    if not os.path.exists(he_zarr_path):
                        print(f"  ⚠️  {sample}: Zarr not found at {he_zarr_path}")
                        continue

                    z = zarr.open(he_zarr_path, 'r')
                    n_patches = z.shape[0]
                    for patch_idx in range(n_patches):
                        self.patch_info.append((sample, None, patch_idx))
                    
                    if self.config.verbose:
                        print(f"  ✅ {sample} ({dataset_type}): {n_patches} patches")
                except Exception as e:
                    print(f"  ⚠️  {sample}: Could not load - {e}")

    def __len__(self):
        return len(self.patch_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single patch with all modalities and masks"""
        sample_name, sub_sample, patch_idx = self.patch_info[idx]
        dataset_type = self._get_dataset_type(sample_name)
        zarr_path = self._get_zarr_path(sample_name)
        
        try:
            if dataset_type == 'tissuenet':
                base_folder = os.path.join(zarr_path, sample_name, sub_sample)
                
                mif_img = zarr.open(os.path.join(base_folder, 'mif/images.zarr'), 'r')[0]
                mif_nuclei_mask = zarr.open(os.path.join(base_folder, 'mif/nuclei_masks.zarr'), 'r')[0]
                mif_cell_mask = zarr.open(os.path.join(base_folder, 'mif/cell_masks.zarr'), 'r')[0]
                
                h, w = mif_img.shape[:2]
                he_img = np.zeros((h, w, 3), dtype=np.uint8)
                he_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                he_cell_mask = np.zeros((h, w), dtype=np.int32)
            
            elif dataset_type == 'fluorescence_dsb2018':
                base_folder = os.path.join(zarr_path, sample_name, sub_sample)
                
                # 1. Load raw data
                raw_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[patch_idx] # (512, 512, 3)
                mif_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[patch_idx]
                
                # 2. Extract channels
                # Based on your previous test, Ch 0 is nuclei. 
                # Since Zarr has 3 channels, let's take the first two for MIF compatibility.
                mif_img = raw_img[..., :2].astype(np.float32) 
                
                # 3. FIX NORMALIZATION: Force division by 255 because data is uint8
                mif_img /= 255.0 
                
                h, w = mif_img.shape[:2]
                # Placeholders
                he_img = np.zeros((h, w, 3), dtype=np.float32)
                he_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                he_cell_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)

            elif dataset_type == 'pannuke':
                base_folder = os.path.join(zarr_path, sample_name, sub_sample)
                
                he_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[0]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[0]
                
                h, w = he_img.shape[:2]
                he_cell_mask = np.zeros((h, w), dtype=np.int32)
                
                mif_img = np.zeros((h, w, 2), dtype=np.uint16)
                mif_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)
            elif dataset_type == 'monuseg':
                base_folder = os.path.join(zarr_path, sample_name, sub_sample)
                
                he_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[patch_idx]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[patch_idx]
                
                h, w = he_img.shape[:2]
                he_cell_mask = np.zeros((h, w), dtype=np.int32)
                
                mif_img = np.zeros((h, w, 2), dtype=np.uint16)
                mif_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)
            elif dataset_type in ['monusac', 'cpm15', 'he_dsb2018', 'panoptils']:
                base_folder = os.path.join(zarr_path, sample_name, sub_sample)
                
                he_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[patch_idx]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[patch_idx]
                
                h, w = he_img.shape[:2]
                he_cell_mask = np.zeros((h, w), dtype=np.int32)
                
                mif_img = np.zeros((h, w, 2), dtype=np.uint16)
                mif_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)
            elif dataset_type == 'tnbc':
                base_folder = os.path.join(zarr_path, sample_name, sub_sample)
                
                he_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[patch_idx]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[patch_idx]
                
                h, w = he_img.shape[:2]
                he_cell_mask = np.zeros((h, w), dtype=np.int32)
                
                mif_img = np.zeros((h, w, 2), dtype=np.uint16)
                mif_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)

            elif dataset_type == 'lizard':
                base_folder = os.path.join(zarr_path, sample_name)
                
                he_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[patch_idx]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[patch_idx]
                
                h, w = he_img.shape[:2]
                he_cell_mask = np.zeros((h, w), dtype=np.int32)
                
                mif_img = np.zeros((h, w, 2), dtype=np.uint16)
                mif_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)
            elif dataset_type == 'nuinsseg':
                base_folder = os.path.join(zarr_path, sample_name)
                
                he_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[patch_idx]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[patch_idx]
                
                h, w = he_img.shape[:2]
                he_cell_mask = np.zeros((h, w), dtype=np.int32)
                
                mif_img = np.zeros((h, w, 2), dtype=np.uint16)
                mif_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)
            elif dataset_type == 'cryonuseg':
                base_folder = os.path.join(zarr_path, sample_name)
                
                he_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[patch_idx]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[patch_idx]
                
                h, w = he_img.shape[:2]
                he_cell_mask = np.zeros((h, w), dtype=np.int32)
                
                mif_img = np.zeros((h, w, 2), dtype=np.uint16)
                mif_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)
            elif dataset_type == 'bc':
                base_folder = os.path.join(zarr_path, sample_name)
                
                # BCData has images.zarr and nuclei_masks.zarr
                he_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[patch_idx]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[patch_idx]
                
                h, w = he_img.shape[:2]
                he_cell_mask = np.zeros((h, w), dtype=np.int32) # No cell mask in BCData
                
                # Empty MIF channels
                mif_img = np.zeros((h, w, 2), dtype=np.uint16)
                mif_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)
            elif dataset_type == 'consep':
                base_folder = os.path.join(zarr_path, sample_name)
                
                # CoNSeP has images.zarr and nuclei_masks.zarr
                he_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[patch_idx]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[patch_idx]
                
                h, w = he_img.shape[:2]
                he_cell_mask = np.zeros((h, w), dtype=np.int32) # No cell mask
                
                # Empty MIF channels
                mif_img = np.zeros((h, w, 2), dtype=np.uint16)
                mif_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)
            elif dataset_type in ['kumar', 'cpm17']:
                base_folder = os.path.join(zarr_path, sample_name)
                
                # Kumar/CPM17 have images.zarr and nuclei_masks.zarr
                he_img = zarr.open(os.path.join(base_folder, 'images.zarr'), 'r')[patch_idx]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'nuclei_masks.zarr'), 'r')[patch_idx]
                
                h, w = he_img.shape[:2]
                he_cell_mask = np.zeros((h, w), dtype=np.int32) # No cell mask
                
                # Empty MIF channels
                mif_img = np.zeros((h, w, 2), dtype=np.uint16)
                mif_nuclei_mask = np.zeros((h, w), dtype=np.int32)
                mif_cell_mask = np.zeros((h, w), dtype=np.int32)
            else:
                base_folder = os.path.join(zarr_path, sample_name)
                
                he_img = zarr.open(os.path.join(base_folder, 'he/images.zarr'), 'r')[patch_idx]
                he_nuclei_mask = zarr.open(os.path.join(base_folder, 'he/nuclei_masks.zarr'), 'r')[patch_idx]
                he_cell_mask = zarr.open(os.path.join(base_folder, 'he/cell_masks.zarr'), 'r')[patch_idx]
                
                mif_img = zarr.open(os.path.join(base_folder, 'mif/images.zarr'), 'r')[patch_idx]
                mif_nuclei_mask = zarr.open(os.path.join(base_folder, 'mif/nuclei_masks.zarr'), 'r')[patch_idx]
                mif_cell_mask = zarr.open(os.path.join(base_folder, 'mif/cell_masks.zarr'), 'r')[patch_idx]

        except Exception as e:
            if self.config.verbose:
                print(f"❌ Error loading {dataset_type} ({sample_name}/{sub_sample}): {e}")
            return self.__getitem__(np.random.randint(0, len(self.patch_info)))

        he_img = he_img.astype(np.float32) / self.config.he_max_value
        mif_img = mif_img.astype(np.float32) / self.config.mif_max_value
        mif_img = np.clip(mif_img, 0.0, 1.0)
        
        he_nuclei_mask = he_nuclei_mask.astype(np.int64)
        he_cell_mask = he_cell_mask.astype(np.int64)
        mif_nuclei_mask = mif_nuclei_mask.astype(np.int64)
        mif_cell_mask = mif_cell_mask.astype(np.int64)
        
        if self.config.generate_hv_maps:
            he_n_hv = batch_generate_hv_maps(he_nuclei_mask[np.newaxis, ...], method=self.config.hv_method)[0]
            he_c_hv = batch_generate_hv_maps(he_cell_mask[np.newaxis, ...], method=self.config.hv_method)[0]
            mif_n_hv = batch_generate_hv_maps(mif_nuclei_mask[np.newaxis, ...], method=self.config.hv_method)[0]
            mif_c_hv = batch_generate_hv_maps(mif_cell_mask[np.newaxis, ...], method=self.config.hv_method)[0]
        else:
            h, w = mif_img.shape[:2]
            he_n_hv = np.zeros((2, h, w), dtype=np.float32)
            he_c_hv = np.zeros((2, h, w), dtype=np.float32)
            mif_n_hv = np.zeros((2, h, w), dtype=np.float32)
            mif_c_hv = np.zeros((2, h, w), dtype=np.float32)
        
        he_img = torch.from_numpy(he_img).permute(2, 0, 1).float()
        mif_img = torch.from_numpy(mif_img).permute(2, 0, 1).float()
        
        he_nuclei_mask = torch.from_numpy(he_nuclei_mask).long()
        he_cell_mask = torch.from_numpy(he_cell_mask).long()
        mif_nuclei_mask = torch.from_numpy(mif_nuclei_mask).long()
        mif_cell_mask = torch.from_numpy(mif_cell_mask).long()
        
        he_n_hv = torch.from_numpy(he_n_hv).float()
        he_c_hv = torch.from_numpy(he_c_hv).float()
        mif_n_hv = torch.from_numpy(mif_n_hv).float()
        mif_c_hv = torch.from_numpy(mif_c_hv).float()
        
        if self.augment is not None:
            (he_img, mif_img, he_nuclei_mask, he_cell_mask,
            mif_nuclei_mask, mif_cell_mask, he_n_hv, he_c_hv,
            mif_n_hv, mif_c_hv) = self.augment(
                he_img, mif_img, he_nuclei_mask, he_cell_mask,
                mif_nuclei_mask, mif_cell_mask, he_n_hv, he_c_hv,
                mif_n_hv, mif_c_hv
            )
        
        he_n_bin = (he_nuclei_mask > 0).long()
        he_c_bin = (he_cell_mask > 0).long()
        mif_n_bin = (mif_nuclei_mask > 0).long()
        mif_c_bin = (mif_cell_mask > 0).long()
        
        combined_img = torch.cat([he_img, mif_img], dim=0)
        
        return {
            'image': combined_img,
            'he_image': he_img,
            'mif_image': mif_img,
            'he_nuclei_mask': he_n_bin,
            'he_cell_mask': he_c_bin,
            'mif_nuclei_mask': mif_n_bin,
            'mif_cell_mask': mif_c_bin,
            'he_nuclei_hv': he_n_hv,
            'he_cell_hv': he_c_hv,
            'mif_nuclei_hv': mif_n_hv,
            'mif_cell_hv': mif_c_hv,
            'he_nuclei_instance': he_nuclei_mask,
            'he_cell_instance': he_cell_mask,
            'mif_nuclei_instance': mif_nuclei_mask,
            'mif_cell_instance': mif_cell_mask,
            'sample_name': sub_sample if sub_sample else sample_name,
            'patch_idx': patch_idx,
            'dataset_source': dataset_type 
        }


class CRCCachedDataset(CRCZarrDataset):
    """Memory-cached version of multi-cancer dataset"""
    
    def __init__(self, samples: List[str], config, training: bool = True):
        self.cache = None
        self.cache_loaded = False
        
        if config.use_cache and not config.force_regenerate_cache:
            cache_path = self._get_cache_path(config, samples, training)
            if os.path.exists(cache_path):
                print(f"📦 Loading from cache: {cache_path}")
                self._load_cache(cache_path)
                self.samples = samples
                self.config = config
                self.training = training
                self.zarr_bases = config.zarr_bases
                
                if training and config.train_augment:
                    aug_config = config.get_augmentation_config()
                    self.augment = MedicalImageAugmentation(aug_config)
                else:
                    self.augment = None
                
                return
        
        super().__init__(samples, config, training)
        
        if config.use_cache:
            print(f"🚀 Preloading {len(self.patch_info)} patches to memory...")
            self._preload_to_memory()
            
            cache_path = self._get_cache_path(config, samples, training)
            print(f"💾 Saving cache to: {cache_path}")
            self._save_cache(cache_path)
    
    def _get_cache_path(self, config, samples, training):
        import hashlib
        
        samples_str = "_".join(sorted(samples))
        samples_hash = hashlib.md5(samples_str.encode()).hexdigest()[:8]
        
        split_name = "train" if training else "val"
        cache_filename = f"combined_cache_{split_name}_{samples_hash}.pkl"
        
        return os.path.join(config.cache_dir, cache_filename)
    
    def _preload_to_memory(self):
        self.cache = []
        
        for idx in tqdm(range(len(self.patch_info)), desc="Preloading"):
            data = super().__getitem__(idx)
            self.cache.append(data)
        
        self.cache_loaded = True
        
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)
        print(f"✅ Preloaded {len(self.cache)} patches")
        print(f"   Memory usage: {memory_gb:.2f} GB")
    
    def _save_cache(self, cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'cache': self.cache,
                'patch_info': self.patch_info
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        cache_size_gb = os.path.getsize(cache_path) / (1024 ** 3)
        print(f"💾 Cache saved: {cache_size_gb:.2f} GB")
    
    def _load_cache(self, cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        self.cache = data['cache']
        self.patch_info = data['patch_info']
        self.cache_loaded = True
        
        print(f"📦 Loaded {len(self.cache)} patches from cache")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_loaded:
            data = self.cache[idx]
            
            if self.augment is not None and self.training:
                (he_img, mif_img, he_nuclei_mask, he_cell_mask,
                 mif_nuclei_mask, mif_cell_mask, he_nuclei_hv, he_cell_hv,
                 mif_nuclei_hv, mif_cell_hv) = self.augment(
                    data['he_image'], data['mif_image'],
                    data['he_nuclei_instance'], data['he_cell_instance'],
                    data['mif_nuclei_instance'], data['mif_cell_instance'],
                    data['he_nuclei_hv'], data['he_cell_hv'],
                    data['mif_nuclei_hv'], data['mif_cell_hv']
                )
                
                data = data.copy()
                data['he_image'] = he_img
                data['mif_image'] = mif_img
                data['image'] = torch.cat([he_img, mif_img], dim=0)
                data['he_nuclei_mask'] = (he_nuclei_mask > 0).long()
                data['he_cell_mask'] = (he_cell_mask > 0).long()
                data['mif_nuclei_mask'] = (mif_nuclei_mask > 0).long()
                data['mif_cell_mask'] = (mif_cell_mask > 0).long()
                data['he_nuclei_hv'] = he_nuclei_hv
                data['he_cell_hv'] = he_cell_hv
                data['mif_nuclei_hv'] = mif_nuclei_hv
                data['mif_cell_hv'] = mif_cell_hv
            
            return data
        else:
            return super().__getitem__(idx)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    batched = {}
    
    for key in batch[0].keys():
        if key in ['sample_name', 'dataset_source']:
            batched[key] = [item[key] for item in batch]
        elif key == 'patch_idx':
            batched[key] = torch.tensor([item[key] for item in batch])
        else:
            batched[key] = torch.stack([item[key] for item in batch])
    
    return batched


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_samples, val_samples, test_samples = config.get_splits()
    
    if config.verbose:
        print(f"\n{'='*80}")
        print("CREATING DATALOADERS")
        print(f"{'='*80}")
        print(f"Strategy: {config.strategy}")
        print(f"Use Cache: {config.use_cache}")
        print(f"Batch Size: {config.batch_size}")
        print(f"Num Workers: {config.num_workers}")
        
        def count_types(samples):
            crc = sum(1 for s in samples if s.startswith('CRC'))
            # FIX: Check for 'pannuke' prefix
            pn = sum(1 for s in samples if s.startswith('pannuke') or s.startswith('Fold'))
            liz = sum(1 for s in samples if s.startswith('lizard'))
            # FIX: Check for 'tissuenet' prefix
            tn = sum(1 for s in samples if s.startswith('tissuenet') or s in ['train', 'val', 'test'])
            mon = sum(1 for s in samples if s.startswith('monuseg'))
            sac = sum(1 for s in samples if s.startswith('monusac'))
            tnbc = sum(1 for s in samples if s.startswith('tnbc'))
            nuins = sum(1 for s in samples if s.startswith('nuinsseg'))
            cryo = sum(1 for s in samples if s.startswith('cryonuseg'))
            bc = sum(1 for s in samples if s.startswith('bc'))
            consep = sum(1 for s in samples if s.startswith('consep'))
            kumar = sum(1 for s in samples if s.startswith('kumar'))
            cpm17 = sum(1 for s in samples if s.startswith('cpm17'))
            cpm15 = sum(1 for s in samples if s.startswith('cpm15'))
            he_dsb = sum(1 for s in samples if s.startswith('he_dsb2018'))
            fluorescence_dsb = sum(1 for s in samples if s.startswith('fluorescence_dsb2018'))
            ptils = sum(1 for s in samples if s.startswith('panoptils'))
            # Xenium is whatever is left
            xen = len(samples) - (crc + tn + pn + liz + mon + sac + tnbc + nuins + cryo + bc + consep + kumar + cpm17 + cpm15 + he_dsb + fluorescence_dsb + ptils)
            
            return crc, xen, tn, pn, liz, mon, sac, tnbc, nuins, cryo, bc, consep, kumar, cpm17, cpm15, he_dsb, fluorescence_dsb, ptils
        # Helper for printing
        # Change this line (approx line 654):
        def print_split(name, counts):
            # Added 'f_dsb' to make it 18 variables
            (c, x, t, p, l, m, s, tnbc, n, cryo, bc, cp, kum, cpm, cpm15, dsb, f_dsb, ptils) = counts 
            # Update your print string to use f_dsb as well
            print(f"{name} split: {c} CRC + {x} Xenium + {t} TissueNet + {p} PanNuke + {l} Lizard + {m} MoNuSeg + {s} MoNuSAC + {tnbc} TNBC + {n} NuInsSeg + {cryo} CryoNuSeg + {bc} BC + {cp} CoNSeP + {kum} Kumar + {cpm} CPM17 + {cpm15} CPM15 + {dsb} HE_DSB2018 + {f_dsb} Fluo_DSB2018 + {ptils} PanopTILs")
        print_split("Train", count_types(train_samples))
        print_split("Val  ", count_types(val_samples))
        print_split("Test ", count_types(test_samples))
        print(f"{'='*80}\n")
    
    if config.strategy == 'memory' and config.use_cache:
        dataset_class = CRCCachedDataset
    else:
        dataset_class = CRCZarrDataset
    
    # ------------------------------------------------------------
    # Optional: split ONLY Xenium training patches into validation
    # ------------------------------------------------------------
    # This keeps existing dataset-specific validation sets unchanged
    # (PanNuke, TissueNet, Lizard, MoNuSeg, TNBC, etc.) and only
    # takes a random fraction from Xenium training patches for tuning.
    xenium_val_fraction = float(
        getattr(config, "xenium_val_fraction_from_train",
                getattr(config, "val_fraction_from_train", 0.0))
    )
    use_xenium_train_fraction_for_val = bool(
        getattr(config, "use_xenium_train_fraction_for_val",
                getattr(config, "use_train_fraction_for_val", False))
    )

    def _is_xenium_sample(sample_name: str) -> bool:
        known_prefixes = (
            'CRC', 'tissuenet', 'pannuke', 'Fold', 'lizard', 'monuseg',
            'monusac', 'tnbc', 'nuinsseg', 'cryonuseg', 'bc', 'consep',
            'kumar', 'cpm17', 'cpm15', 'he_dsb2018',
            'fluorescence_dsb2018', 'panoptils'
        )
        legacy_names = {'train', 'val', 'test'}
        return sample_name not in legacy_names and not sample_name.startswith(known_prefixes)

    def _concat_datasets(datasets):
        datasets = [d for d in datasets if d is not None and len(d) > 0]
        if len(datasets) == 0:
            return None
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    if use_xenium_train_fraction_for_val and xenium_val_fraction > 0:
        xenium_train_samples = [s for s in train_samples if _is_xenium_sample(s)]
        non_xenium_train_samples = [s for s in train_samples if not _is_xenium_sample(s)]

        # Existing validation samples stay as validation.
        # Only Xenium training samples are patch-split into train/val.
        non_xenium_train_dataset = (
            dataset_class(non_xenium_train_samples, config, training=True)
            if len(non_xenium_train_samples) > 0 else None
        )
        existing_val_dataset = (
            dataset_class(val_samples, config, training=False)
            if len(val_samples) > 0 else None
        )

        xenium_train_base = dataset_class(xenium_train_samples, config, training=True)
        xenium_val_base = dataset_class(xenium_train_samples, config, training=False)

        n_xenium = len(xenium_train_base)
        n_xenium_val = int(n_xenium * xenium_val_fraction)
        n_xenium_train = n_xenium - n_xenium_val

        generator = torch.Generator().manual_seed(config.random_seed)
        indices = torch.randperm(n_xenium, generator=generator).tolist()

        xenium_val_indices = indices[:n_xenium_val]
        xenium_train_indices = indices[n_xenium_val:]

        xenium_train_dataset = Subset(xenium_train_base, xenium_train_indices)
        xenium_val_dataset = Subset(xenium_val_base, xenium_val_indices)

        train_dataset = _concat_datasets([non_xenium_train_dataset, xenium_train_dataset])
        val_dataset = _concat_datasets([existing_val_dataset, xenium_val_dataset])

        if config.verbose:
            print(f"🔀 Xenium-only train/val split:")
            print(f"   Xenium train-source samples: {xenium_train_samples}")
            print(f"   Xenium total patches: {n_xenium}")
            print(f"   Xenium train patches: {n_xenium_train}")
            print(f"   Xenium val patches: {n_xenium_val}")
            print(f"   Xenium val fraction: {xenium_val_fraction:.2f}")
            print(f"   Existing non-Xenium val samples kept: {len(val_samples)} samples")
    else:
        train_dataset = dataset_class(train_samples, config, training=True)
        val_dataset = dataset_class(val_samples, config, training=False)

    test_dataset = dataset_class(test_samples, config, training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    if config.verbose:
        print(f"✅ Dataloaders created:")
        print(f"   Train: {len(train_dataset)} patches ({len(train_loader)} batches)")
        print(f"   Val: {len(val_dataset)} patches ({len(val_loader)} batches)")
        print(f"   Test: {len(test_dataset)} patches ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader