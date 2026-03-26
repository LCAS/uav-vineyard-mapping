"""
Ground Control Point (GCP) Camera Calibration Script

This script uses known ground control points (pixel coordinates -> GPS coordinates)
to optimize camera parameters like focal length, sensor dimensions, etc.

Usage:
1. Add your GCPs with known pixel coordinates and GPS coordinates
2. Run the script to find optimal camera parameters
3. Use the optimized parameters in your main inference script
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import math
from typing import List, Tuple, Dict
import json
import argparse
import glob
from pathlib import Path

# Import the GPS conversion function
import image_gps_pixel_show_poles


class GCPCalibrator:
    """Calibrates camera parameters using Ground Control Points"""
    
    def __init__(self, image_path: str):
        """
        Initialize calibrator with image metadata
        
        Args:
            image_path: Path to the image containing GCPs
        """
        self.image_path = image_path
        
        # Extract EXIF data
        (self.flight_yaw, self.flight_pitch, self.flight_roll,
         self.gimbal_yaw, self.gimbal_pitch, self.gimbal_roll,
         self.gps_lat, self.gps_lon, self.gps_alt,
         self.fov, self.focal_length, self.img_h, self.img_w) = \
            image_gps_pixel_show_poles.extract_exif(image_path)
        
        # Convert to numeric
        self.flight_yaw_num = image_gps_pixel_show_poles.extract_number(self.flight_yaw)
        self.gimbal_yaw_num = image_gps_pixel_show_poles.extract_number(self.gimbal_yaw)
        if self.gimbal_yaw_num == 0.0 or self.gimbal_yaw_num is None:
            self.gimbal_yaw_num = self.flight_yaw_num
        
        self.gps_alt_num = image_gps_pixel_show_poles.extract_number(self.gps_alt)
        if self.gps_alt_num is None:
            self.gps_alt_num = 0.0
        
        self.focal_length_num = image_gps_pixel_show_poles.extract_number(self.focal_length)
        
        # Store GCPs
        self.gcps: List[Dict] = []
        
        print(f"📷 Image Metadata:")
        print(f"   Size: {self.img_w}x{self.img_h}")
        print(f"   GPS: ({self.gps_lat:.6f}, {self.gps_lon:.6f})")
        print(f"   Altitude: {self.gps_alt_num:.2f}m")
        print(f"   Gimbal Yaw: {self.gimbal_yaw_num:.2f}°")
        print(f"   Flight Yaw: {self.flight_yaw_num:.2f}°")
        if self.focal_length_num:
            print(f"   Focal Length (EXIF): {self.focal_length_num:.2f}mm")
    
    def add_gcp(self, pixel_x: int, pixel_y: int, gps_lat: float, gps_lon: float, label: str = ""):
        """
        Add a ground control point
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            gps_lat: Known latitude
            gps_lon: Known longitude
            label: Optional label for this GCP
        """
        self.gcps.append({
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'gps_lat': gps_lat,
            'gps_lon': gps_lon,
            'label': label
        })
        print(f"✓ Added GCP {len(self.gcps)}: {label if label else f'Point {len(self.gcps)}'}")
        print(f"  Pixel: ({pixel_x}, {pixel_y}) -> GPS: ({gps_lat:.8f}, {gps_lon:.8f})")
    
    @classmethod
    def from_json(cls, json_path: str):
        """
        Load calibrator from JSON file
        
        JSON format:
        {
            "image_path": "path/to/image.jpg",
            "gcps": [
                {
                    "pixel_x": 1234,
                    "pixel_y": 567,
                    "gps_lat": 53.268188,
                    "gps_lon": -0.524277,
                    "label": "Pole 1"
                },
                ...
            ]
        }
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract image path
        image_path = data.get('image_path')
        if not image_path:
            raise ValueError(f"JSON file {json_path} must contain 'image_path' field")
        
        # Create calibrator
        calibrator = cls(image_path)
        
        # Load GCPs
        gcps_data = data.get('gcps', [])
        for gcp in gcps_data:
            # Skip comment entries
            if "_comment" in gcp or "_description" in gcp:
                continue
            calibrator.add_gcp(
                gcp['pixel_x'], gcp['pixel_y'],
                gcp['gps_lat'], gcp['gps_lon'],
                gcp.get('label', '')
            )
        
        return calibrator
    
    def load_gcps_from_json(self, json_path: str):
        """
        Load GCPs from a JSON file (for existing calibrator)
        
        JSON format (array):
        [
            {
                "pixel_x": 1234,
                "pixel_y": 567,
                "gps_lat": 53.268188,
                "gps_lon": -0.524277,
                "label": "Pole 1"
            },
            ...
        ]
        """
        with open(json_path, 'r') as f:
            gcps_data = json.load(f)
        
        # Handle both array and object formats
        if isinstance(gcps_data, dict) and 'gcps' in gcps_data:
            gcps_data = gcps_data['gcps']
        
        for gcp in gcps_data:
            # Skip comment entries
            if "_comment" in gcp or "_description" in gcp:
                continue
            self.add_gcp(
                gcp['pixel_x'], gcp['pixel_y'],
                gcp['gps_lat'], gcp['gps_lon'],
                gcp.get('label', '')
            )
    
    def _compute_error(self, params: np.ndarray) -> float:
        """
        Compute the total error for given parameters
        
        Args:
            params: [focal_length_mm, sensor_width_mm, sensor_height_mm]
        
        Returns:
            Total error in meters
        """
        focal_length_mm, sensor_width_mm, sensor_height_mm = params
        
        total_error = 0.0
        
        for gcp in self.gcps:
            # Predict GPS from pixel using current parameters
            pred_lat, pred_lon = image_gps_pixel_show_poles.get_gps_from_pixel(
                gcp['pixel_x'], gcp['pixel_y'],
                self.img_w, self.img_h,
                self.flight_yaw_num, self.gimbal_yaw_num,
                self.gps_lat, self.gps_lon, self.gps_alt_num,
                focal_length_mm, sensor_width_mm, sensor_height_mm
            )
            
            # Calculate error distance
            error = self._haversine_distance(
                gcp['gps_lat'], gcp['gps_lon'],
                pred_lat, pred_lon
            )
            total_error += error ** 2
        
        # Return RMSE
        return np.sqrt(total_error / len(self.gcps))
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in meters between two GPS coordinates"""
        R = 6371000  # Earth radius in meters
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def calibrate_local_search(self, initial_params: Tuple[float, float, float],
                               param_ranges: Tuple[Tuple[float, float], ...] = None) -> Dict:
        """
        Calibrate using local optimization (faster but may find local minimum)
        
        Args:
            initial_params: (focal_length_mm, sensor_width_mm, sensor_height_mm)
            param_ranges: Optional bounds for each parameter
        
        Returns:
            Dictionary with optimized parameters and error metrics
        """
        if len(self.gcps) < 2:
            raise ValueError("Need at least 2 GCPs for calibration")
        
        print(f"\n🔍 Running Local Optimization...")
        print(f"   Initial params: focal={initial_params[0]:.2f}mm, "
              f"sensor_w={initial_params[1]:.2f}mm, sensor_h={initial_params[2]:.2f}mm")
        
        # Initial error
        initial_error = self._compute_error(np.array(initial_params))
        print(f"   Initial RMSE: {initial_error:.3f}m")
        
        # Set bounds
        if param_ranges is None:
            # Default: ±50% of initial values
            param_ranges = tuple(
                (p * 0.5, p * 1.5) for p in initial_params
            )
        
        # Optimize
        result = minimize(
            self._compute_error,
            initial_params,
            method='L-BFGS-B',
            bounds=param_ranges
        )
        
        optimal_params = result.x
        final_error = result.fun
        
        print(f"\n✅ Optimization Complete!")
        print(f"   Optimal params: focal={optimal_params[0]:.3f}mm, "
              f"sensor_w={optimal_params[1]:.3f}mm, sensor_h={optimal_params[2]:.3f}mm")
        print(f"   Final RMSE: {final_error:.3f}m")
        print(f"   Improvement: {((initial_error - final_error) / initial_error * 100):.1f}%")
        
        return {
            'focal_length_mm': optimal_params[0],
            'sensor_width_mm': optimal_params[1],
            'sensor_height_mm': optimal_params[2],
            'rmse_meters': final_error,
            'initial_rmse_meters': initial_error,
            'improvement_percent': (initial_error - final_error) / initial_error * 100,
            'method': 'local_search'
        }
    
    def calibrate_global_search(self, param_ranges: Tuple[Tuple[float, float], ...]) -> Dict:
        """
        Calibrate using global optimization (slower but more robust)
        
        Args:
            param_ranges: Bounds for (focal_length, sensor_width, sensor_height)
                         e.g., ((3.0, 6.0), (4.0, 8.0), (3.0, 6.0))
        
        Returns:
            Dictionary with optimized parameters and error metrics
        """
        if len(self.gcps) < 2:
            raise ValueError("Need at least 2 GCPs for calibration")
        
        print(f"\n🌍 Running Global Optimization (this may take a minute)...")
        print(f"   Search ranges:")
        print(f"     Focal length: {param_ranges[0][0]:.1f}-{param_ranges[0][1]:.1f}mm")
        print(f"     Sensor width: {param_ranges[1][0]:.1f}-{param_ranges[1][1]:.1f}mm")
        print(f"     Sensor height: {param_ranges[2][0]:.1f}-{param_ranges[2][1]:.1f}mm")
        
        # Use differential evolution (global optimizer)
        result = differential_evolution(
            self._compute_error,
            param_ranges,
            seed=42,
            maxiter=1000,
            popsize=15,
            tol=0.01,
            atol=0.01
        )
        
        optimal_params = result.x
        final_error = result.fun
        
        print(f"\n✅ Optimization Complete!")
        print(f"   Optimal params: focal={optimal_params[0]:.3f}mm, "
              f"sensor_w={optimal_params[1]:.3f}mm, sensor_h={optimal_params[2]:.3f}mm")
        print(f"   Final RMSE: {final_error:.3f}m")
        
        return {
            'focal_length_mm': optimal_params[0],
            'sensor_width_mm': optimal_params[1],
            'sensor_height_mm': optimal_params[2],
            'rmse_meters': final_error,
            'method': 'global_search'
        }
    
    def validate_parameters(self, focal_length_mm: float, sensor_width_mm: float, 
                           sensor_height_mm: float) -> None:
        """
        Validate parameters by showing predicted vs actual GPS for each GCP
        
        Args:
            focal_length_mm: Focal length to test
            sensor_width_mm: Sensor width to test
            sensor_height_mm: Sensor height to test
        """
        print(f"\n📊 Validation Results:")
        print(f"   Using: focal={focal_length_mm:.3f}mm, "
              f"sensor_w={sensor_width_mm:.3f}mm, sensor_h={sensor_height_mm:.3f}mm\n")
        
        errors = []
        
        for i, gcp in enumerate(self.gcps, 1):
            pred_lat, pred_lon = image_gps_pixel_show_poles.get_gps_from_pixel(
                gcp['pixel_x'], gcp['pixel_y'],
                self.img_w, self.img_h,
                self.flight_yaw_num, self.gimbal_yaw_num,
                self.gps_lat, self.gps_lon, self.gps_alt_num,
                focal_length_mm, sensor_width_mm, sensor_height_mm
            )
            
            error = self._haversine_distance(
                gcp['gps_lat'], gcp['gps_lon'],
                pred_lat, pred_lon
            )
            errors.append(error)
            
            print(f"   GCP {i}: {gcp['label']}")
            print(f"     Actual:    ({gcp['gps_lat']:.8f}, {gcp['gps_lon']:.8f})")
            print(f"     Predicted: ({pred_lat:.8f}, {pred_lon:.8f})")
            print(f"     Error:     {error:.3f}m\n")
        
        print(f"   RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.3f}m")
        print(f"   Mean Error: {np.mean(errors):.3f}m")
        print(f"   Max Error: {np.max(errors):.3f}m")
    
    def save_results(self, results: Dict, output_path: str):
        """Save calibration results to JSON file"""
        results['image_path'] = self.image_path
        results['image_size'] = [self.img_w, self.img_h]
        results['altitude_m'] = self.gps_alt_num
        results['num_gcps'] = len(self.gcps)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")


def load_multi_image_gcps(gcp_folder: str) -> List[Tuple[GCPCalibrator, Dict]]:
    """
    Load all GCP JSON files from a folder
    
    Returns:
        List of (calibrator, gcp_data) tuples
    """
    json_files = sorted(glob.glob(f"{gcp_folder}/*_gcps.json"))
    
    if not json_files:
        print(f"❌ No GCP JSON files found in {gcp_folder}")
        return []
    
    print(f"📂 Found {len(json_files)} GCP file(s) in {gcp_folder}")
    
    calibrators = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            image_path = data.get('image_path')
            if not image_path:
                print(f"⚠️  Skipping {json_file} (no image_path)")
                continue
            
            # Create calibrator for this image
            calibrator = GCPCalibrator(image_path)
            
            # Load GCPs
            gcps_data = data.get('gcps', [])
            for gcp in gcps_data:
                if "_comment" in gcp or "_description" in gcp:
                    continue
                calibrator.add_gcp(
                    gcp['pixel_x'], gcp['pixel_y'],
                    gcp['gps_lat'], gcp['gps_lon'],
                    gcp.get('label', '')
                )
            
            calibrators.append((calibrator, data))
            print(f"✓ Loaded {len(gcps_data)} GCP(s) from {Path(json_file).name}")
        
        except Exception as e:
            print(f"❌ Error loading {json_file}: {str(e)}")
    
    return calibrators


def calibrate_from_folder(gcp_folder: str, method: str = "local") -> Dict:
    """
    Calibrate camera using all GCP files in a folder
    
    Args:
        gcp_folder: Folder containing *_gcps.json files
        method: "local" or "global" optimization
    
    Returns:
        Dictionary with calibration results
    """
    calibrators = load_multi_image_gcps(gcp_folder)
    
    if not calibrators:
        print("❌ No valid GCP data found")
        return {}
    
    # Combine all GCPs with image metadata tracking
    all_gcp_data = []
    for calibrator, _ in calibrators:
        for gcp in calibrator.gcps:
            all_gcp_data.append({
                'gcp': gcp,
                'calibrator': calibrator
            })
    
    total_gcps = len(all_gcp_data)
    print(f"\n🎯 Total GCPs across {len(calibrators)} image(s): {total_gcps}")
    
    # Define error function for combined optimization
    def compute_combined_error(params: np.ndarray) -> float:
        focal_length_mm, sensor_width_mm, sensor_height_mm = params
        
        total_error = 0.0
        
        for item in all_gcp_data:
            gcp = item['gcp']
            calibrator = item['calibrator']
            
            # Predict GPS from pixel using this image's metadata
            pred_lat, pred_lon = image_gps_pixel_show_poles.get_gps_from_pixel(
                gcp['pixel_x'], gcp['pixel_y'],
                calibrator.img_w, calibrator.img_h,
                calibrator.flight_yaw_num, calibrator.gimbal_yaw_num,
                calibrator.gps_lat, calibrator.gps_lon, calibrator.gps_alt_num,
                focal_length_mm, sensor_width_mm, sensor_height_mm
            )
            
            # Calculate error distance
            error = GCPCalibrator._haversine_distance(
                gcp['gps_lat'], gcp['gps_lon'],
                pred_lat, pred_lon
            )
            total_error += error ** 2
        
        return np.sqrt(total_error / total_gcps)
    
    # Initial parameters
    INITIAL_FOCAL_LENGTH = 4.5
    INITIAL_SENSOR_WIDTH = 6.17
    INITIAL_SENSOR_HEIGHT = 4.55
    
    initial_params = (INITIAL_FOCAL_LENGTH, INITIAL_SENSOR_WIDTH, INITIAL_SENSOR_HEIGHT)
    initial_error = compute_combined_error(np.array(initial_params))
    
    print(f"\n🔍 Running {method.upper()} Optimization...")
    print(f"   Initial params: focal={initial_params[0]:.2f}mm, sensor_w={initial_params[1]:.2f}mm, sensor_h={initial_params[2]:.2f}mm")
    print(f"   Initial RMSE: {initial_error:.3f}m")
    
    if method == "global":
        # Global optimization
        result = differential_evolution(
            compute_combined_error,
            bounds=[
                (3.0, 6.0),   # Focal length range
                (4.0, 8.0),   # Sensor width range
                (3.0, 6.0)    # Sensor height range
            ],
            maxiter=1000,
            seed=42
        )
    else:
        # Local optimization (default)
        result = minimize(
            compute_combined_error,
            initial_params,
            method='L-BFGS-B',
            bounds=[
                (3.0, 6.0),
                (4.0, 8.0),
                (3.0, 6.0)
            ]
        )
    
    final_error = compute_combined_error(result.x)
    improvement = ((initial_error - final_error) / initial_error) * 100
    
    print(f"\n✅ Optimization Complete!")
    print(f"   Optimal params: focal={result.x[0]:.3f}mm, sensor_w={result.x[1]:.3f}mm, sensor_h={result.x[2]:.3f}mm")
    print(f"   Final RMSE: {final_error:.3f}m")
    print(f"   Improvement: {improvement:.1f}%")
    
    # Validation
    print(f"\n📊 Validation Results:")
    print(f"   Using: focal={result.x[0]:.3f}mm, sensor_w={result.x[1]:.3f}mm, sensor_h={result.x[2]:.3f}mm\n")
    
    errors = []
    for item in all_gcp_data:
        gcp = item['gcp']
        calibrator = item['calibrator']
        
        pred_lat, pred_lon = image_gps_pixel_show_poles.get_gps_from_pixel(
            gcp['pixel_x'], gcp['pixel_y'],
            calibrator.img_w, calibrator.img_h,
            calibrator.flight_yaw_num, calibrator.gimbal_yaw_num,
            calibrator.gps_lat, calibrator.gps_lon, calibrator.gps_alt_num,
            result.x[0], result.x[1], result.x[2]
        )
        
        error = GCPCalibrator._haversine_distance(
            gcp['gps_lat'], gcp['gps_lon'],
            pred_lat, pred_lon
        )
        errors.append(error)
        
        print(f"   GCP: {gcp['label']}")
        print(f"     Actual:    ({gcp['gps_lat']:.8f}, {gcp['gps_lon']:.8f})")
        print(f"     Predicted: ({pred_lat:.8f}, {pred_lon:.8f})")
        print(f"     Error:     {error:.3f}m")
    
    print(f"\n   RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.3f}m")
    print(f"   Mean Error: {np.mean(errors):.3f}m")
    print(f"   Max Error: {np.max(errors):.3f}m")
    
    return {
        'focal_length_mm': float(result.x[0]),
        'sensor_width_mm': float(result.x[1]),
        'sensor_height_mm': float(result.x[2]),
        'rmse_meters': float(final_error),
        'initial_rmse_meters': float(initial_error),
        'improvement_percent': float(improvement),
        'method': method,
        'num_images': len(calibrators),
        'num_gcps': total_gcps
    }


def main():
    """Main entry point with command-line argument support"""
    parser = argparse.ArgumentParser(description='Calibrate camera using ground control points')
    parser.add_argument('--gcp-folder', type=str, default='gcp_annotations',
                        help='Folder containing GCP JSON files (default: gcp_annotations)')
    parser.add_argument('--method', type=str, choices=['local', 'global'], default='local',
                        help='Optimization method: local (fast) or global (thorough) (default: local)')
    parser.add_argument('--output', type=str, default='camera_calibration_results.json',
                        help='Output JSON file for results (default: camera_calibration_results.json)')
    
    args = parser.parse_args()
    
    # Run calibration with multi-image support
    results = calibrate_from_folder(args.gcp_folder, args.method)
    
    if not results:
        return
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")
    
    print("\n" + "="*60)
    print("🎯 CALIBRATED CAMERA PARAMETERS")
    print("="*60)
    print(f"FOCAL_LENGTH_MM = {results['focal_length_mm']:.4f}")
    print(f"SENSOR_WIDTH_MM = {results['sensor_width_mm']:.4f}")
    print(f"SENSOR_HEIGHT_MM = {results['sensor_height_mm']:.4f}")
    print(f"METHOD = {results['method'].upper()}")
    print(f"IMAGES = {results['num_images']}, GCPs = {results['num_gcps']}")
    print("="*60)
    print("\n💡 Copy these values to your inference script!")


if __name__ == "__main__":
    main()
