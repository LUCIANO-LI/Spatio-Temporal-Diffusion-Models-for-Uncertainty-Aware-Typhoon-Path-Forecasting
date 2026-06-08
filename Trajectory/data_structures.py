from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
import xarray as xr

@dataclass
class StormSample:
    storm_id: str
    times: np.ndarray
    track_lat: np.ndarray
    track_lon: np.ndarray
    track_vmax: np.ndarray
    track_pmin: Optional[np.ndarray] = None
    
    era5_dataset: Optional[xr.Dataset] = None
    era5_array: Optional[np.ndarray] = None
    
    lat_grid: Optional[np.ndarray] = None
    lon_grid: Optional[np.ndarray] = None
    
    is_real: Optional[np.ndarray] = None
    
    basin: Optional[str] = None
    year: Optional[int] = None
    
    def __len__(self) -> int:
        return len(self.times)
    
    def get_era5_at_time(self, t_idx: int) -> Optional[np.ndarray]:
        if self.era5_array is not None:
            return self.era5_array[t_idx]
        elif self.era5_dataset is not None:
            return self._extract_era5_frame(t_idx)
        return None
    
    def _extract_era5_frame(self, t_idx: int) -> np.ndarray:
        from config import data_cfg

        ds = self.era5_dataset.isel(valid_time=t_idx)
        channels = []

        for var in data_cfg.era5_3d_vars:
            if var in ds:
                for level in data_cfg.pressure_levels:
                    try:
                        data = ds[var].sel(pressure_level=level).values
                        channels.append(data[np.newaxis])
                    except (KeyError, ValueError):
                        if 'pressure_level' in ds[var].dims:
                            available = ds[var].pressure_level.values
                            nearest = available[np.argmin(np.abs(available - level))]
                            data = ds[var].sel(pressure_level=nearest).values
                            channels.append(data[np.newaxis])

        for var in data_cfg.era5_2d_vars:
            if var in ds:
                data = ds[var].values
                channels.append(data[np.newaxis])

        if channels:
            return np.concatenate(channels, axis=0)
        else:
            return np.zeros((len(data_cfg.era5_3d_vars) * len(data_cfg.pressure_levels) + len(data_cfg.era5_2d_vars),
                           data_cfg.grid_height, data_cfg.grid_width), dtype=np.float32)

@dataclass
class TrainingSample:
    cond_coords: np.ndarray
    cond_era5: np.ndarray
    cond_features: np.ndarray

    target_deltas: np.ndarray
    target_coords: np.ndarray

    storm_id: Optional[str] = None
    start_time: Optional[np.datetime64] = None

    sample_weight: float = 1.0

@dataclass
class PredictionResult:
    predicted_deltas: np.ndarray

    predicted_lat: np.ndarray
    predicted_lon: np.ndarray
    predicted_vmax: np.ndarray

    auxiliary_heatmap: Optional[np.ndarray] = None

    all_samples: Optional[List[np.ndarray]] = None

    uncertainty: Optional[np.ndarray] = None

    storm_id: Optional[str] = None
    lead_times_hours: Optional[np.ndarray] = None

@dataclass
class EvaluationMetrics:
    track_errors_km: np.ndarray
    mean_track_error_km: float
    
    errors_by_lead_time: Dict[float, float] = field(default_factory=dict)
    
    vmax_errors: Optional[np.ndarray] = None
    
    ensemble_spread: Optional[float] = None
    crps: Optional[float] = None

