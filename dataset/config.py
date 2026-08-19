"""
Configuration loader for CRC Dataset
Loads and validates YAML config with type checking
"""

import yaml
import os
from typing import List, Optional, Dict, Any
from pathlib import Path


class Config:
    """
    Configuration loader with YAML support and type validation
    
    Usage:
        config = Config("config.yaml")
        config = Config("config.yaml", batch_size=8)  # Override specific values
    """
    
    def __init__(self, config_path: str = "config.yaml", **overrides):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML config file
            **overrides: Override any config values (e.g., batch_size=8)
        """
        self.config_path = config_path
        self._load_yaml()
        self._apply_overrides(overrides)
        self._validate()
        self._post_process()
    
    def _load_yaml(self):
        """Load YAML configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply override values from constructor"""
        # Map flat overrides to nested structure
        override_map = {
            'batch_size': ('dataloader', 'batch_size'),
            'num_workers': ('dataloader', 'num_workers'),
            'zarr_base': ('data', 'zarr_base'),
            'augment_prob': ('augmentation', 'augment_prob'),
            'train_augment': ('augmentation', 'train_augment'),
            'cache_dir': ('data', 'cache_dir'),
            'use_cache': ('memory', 'use_cache'),
            'strategy': ('memory', 'strategy'),
            'verbose': ('debug', 'verbose'),
            'max_samples': ('debug', 'max_samples'),
        }
        
        for key, value in overrides.items():
            if key in override_map:
                section, param = override_map[key]
                self._config[section][param] = value
            else:
                print(f"⚠️  Warning: Unknown override parameter: {key}")
    
    def _validate(self):
        """Validate configuration values"""
        # Check required sections
        required_sections = ['data', 'splits', 'dataloader', 'processing', 
                           'augmentation', 'hv_maps', 'filtering', 'memory', 'debug']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate ratios sum to 1.0
        splits = self._config['splits']
        ratio_sum = splits['train_ratio'] + splits['val_ratio'] + splits['test_ratio']
        if abs(ratio_sum - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
        
        # Validate paths exist (support both zarr_base and zarr_bases)
        if 'zarr_bases' in self._config['data']:
            # New multi-base format
            for dataset_name, zarr_path in self._config['data']['zarr_bases'].items():
                if not os.path.exists(zarr_path):
                    print(f"⚠️  Warning: {dataset_name} zarr path does not exist: {zarr_path}")
        elif 'zarr_base' in self._config['data']:
            # Old single-base format (backward compatibility)
            if not os.path.exists(self._config['data']['zarr_base']):
                print(f"⚠️  Warning: zarr_base path does not exist: {self._config['data']['zarr_base']}")
    
    def _post_process(self):
        """Post-process configuration (create directories, etc.)"""
        # Create cache directory
        if self._config['memory']['use_cache']:
            cache_dir = self._config['data']['cache_dir']
            os.makedirs(cache_dir, exist_ok=True)
    
    # ========================================================================
    # PROPERTY ACCESSORS (with type hints)
    # ========================================================================
    
    # Data paths - MULTI-BASE SUPPORT
    @property
    def zarr_bases(self) -> Dict[str, str]:
        """Return dictionary of zarr base paths for multiple datasets"""
        # Support new multi-base format
        if 'zarr_bases' in self._config['data']:
            return self._config['data']['zarr_bases']
        # Backward compatibility with old single zarr_base
        elif 'zarr_base' in self._config['data']:
            return {'crc': self._config['data']['zarr_base']}
        else:
            raise ValueError("Neither zarr_bases nor zarr_base found in config")
    
    @property
    def zarr_base(self) -> str:
        """Return primary zarr base (backward compatibility)"""
        if 'zarr_bases' in self._config['data']:
            # Return first zarr base for backward compatibility
            return list(self._config['data']['zarr_bases'].values())[0]
        return self._config['data']['zarr_base']
    
    @property
    def cache_dir(self) -> str:
        return self._config['data']['cache_dir']
    
    @property
    def cache_path(self) -> Optional[str]:
        if self._config['memory']['use_cache']:
            return os.path.join(
                self._config['data']['cache_dir'],
                self._config['data']['cache_filename']
            )
        return None
    
    # Splits
    @property
    def all_samples(self) -> List[str]:
        return self._config['splits']['all_samples']
    
    @property
    def train_samples(self) -> Optional[List[str]]:
        return self._config['splits']['train_samples']
    
    @property
    def val_samples(self) -> Optional[List[str]]:
        return self._config['splits']['val_samples']
    
    @property
    def test_samples(self) -> Optional[List[str]]:
        return self._config['splits']['test_samples']
    
    @property
    def use_xenium_train_fraction_for_val(self) -> bool:
        return self._config['splits'].get('use_xenium_train_fraction_for_val', False)
    
    @property
    def xenium_val_fraction_from_train(self) -> float:
        return self._config['splits'].get('xenium_val_fraction_from_train', 0.0)
    @property
    def random_seed(self) -> int:
        return self._config['splits'].get('random_seed', 42)
    # Dataloader
    @property
    def batch_size(self) -> int:
        return self._config['dataloader']['batch_size']
    
    @property
    def num_workers(self) -> int:
        return self._config['dataloader']['num_workers']
    
    @property
    def pin_memory(self) -> bool:
        return self._config['dataloader']['pin_memory']
    
    @property
    def persistent_workers(self) -> bool:
        return self._config['dataloader']['persistent_workers']
    
    @property
    def prefetch_factor(self) -> int:
        return self._config['dataloader']['prefetch_factor']
    
    # Processing
    @property
    def normalize(self) -> bool:
        return self._config['processing']['normalize']
    
    @property
    def he_max_value(self) -> float:
        return self._config['processing']['he_max_value']
    
    @property
    def mif_max_value(self) -> float:
        return self._config['processing']['mif_max_value']
    
    # Augmentation
    @property
    def train_augment(self) -> bool:
        return self._config['augmentation']['train_augment']
    
    @property
    def augment_prob(self) -> float:
        return self._config['augmentation']['augment_prob']
    
    # Memory
    @property
    def use_cache(self) -> bool:
        return self._config['memory']['use_cache']
    
    @property
    def force_regenerate_cache(self) -> bool:
        return self._config['memory']['force_regenerate_cache']
    
    @property
    def strategy(self) -> str:
        return self._config['memory']['strategy']
    
    # Debug
    @property
    def verbose(self) -> bool:
        return self._config['debug']['verbose']
    
    @property
    def max_samples(self) -> Optional[int]:
        return self._config['debug']['max_samples']
    
    # HV Maps
    @property
    def generate_hv_maps(self) -> bool:
        return self._config['hv_maps']['generate']
    
    @property
    def hv_method(self) -> str:
        return self._config['hv_maps']['method']
    
    # Filtering
    @property
    def min_instances_per_tile(self) -> int:
        return self._config['filtering']['min_instances_per_tile']
    
    @property
    def filter_empty_masks(self) -> bool:
        return self._config['filtering']['filter_empty_masks']
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_splits(self):
        """
        Get train/val/test splits
        Returns manual splits if provided, otherwise auto-generates
        """
        if (self.train_samples is not None and 
            self.val_samples is not None and 
            self.test_samples is not None):
            return self.train_samples, self.val_samples, self.test_samples
        
        # Auto-generate splits
        from sklearn.model_selection import train_test_split
        
        samples = self.all_samples
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
        
        splits_config = self._config['splits']
        
        # First split: train vs (val + test)
        train_samples, temp_samples = train_test_split(
            samples,
            test_size=(splits_config['val_ratio'] + splits_config['test_ratio']),
            random_state=splits_config['random_seed']
        )
        
        # Second split: val vs test
        val_ratio_adjusted = splits_config['val_ratio'] / (
            splits_config['val_ratio'] + splits_config['test_ratio']
        )
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(1 - val_ratio_adjusted),
            random_state=splits_config['random_seed']
        )
        
        return train_samples, val_samples, test_samples
    
    def get_augmentation_config(self):
        """Get all augmentation settings as a dict"""
        return self._config['augmentation']
    
    def get_hv_config(self):
        """Get HV map settings as a dict"""
        return self._config['hv_maps']
    
    def print_config(self):
        """Print current configuration in a readable format"""
        print("=" * 80)
        print("CRC DATASET CONFIGURATION")
        print("=" * 80)
        print(f"Config File: {self.config_path}")
        print(f"Zarr Base: {self.zarr_base}")
        print(f"Cache: {self.cache_path if self.use_cache else 'Disabled'}")
        print(f"Strategy: {self.strategy}")
        
        print(f"\n📊 Data Splits:")
        train, val, test = self.get_splits()
        print(f"  Train: {len(train)} samples")
        print(f"  Val: {len(val)} samples")
        print(f"  Test: {len(test)} samples")
        
        print(f"\n🔄 DataLoader:")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Num Workers: {self.num_workers}")
        print(f"  Pin Memory: {self.pin_memory}")
        
        print(f"\n🎨 Augmentation:")
        print(f"  Training: {self.train_augment}")
        print(f"  Probability: {self.augment_prob}")
        
        print(f"\n🎯 HV Maps:")
        print(f"  Generate: {self.generate_hv_maps}")
        print(f"  Method: {self.hv_method}")
        hv_config = self.get_hv_config()
        print(f"  HE Nuclei: {hv_config['he_nuclei']}")
        print(f"  HE Cells: {hv_config['he_cells']}")
        print(f"  MIF Nuclei: {hv_config['mif_nuclei']}")
        print(f"  MIF Cells: {hv_config['mif_cells']}")
        
        print(f"\n🔍 Filtering:")
        print(f"  Min Instances: {self.min_instances_per_tile}")
        print(f"  Filter Empty: {self.filter_empty_masks}")
        
        print("=" * 80)
    
    def save(self, output_path: str):
        """Save current configuration to a new YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        print(f"💾 Configuration saved to: {output_path}")