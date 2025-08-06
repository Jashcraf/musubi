import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class OpticalCoating:
    def __init__(self, substrate_index=1.52):
        """
        Optical coating class for needle optimization
        
        Parameters:
        substrate_index: refractive index of substrate (default: glass = 1.52)
        """
        self.substrate_index = substrate_index
        self.layers = []  # [(thickness_nm, refractive_index), ...]
        
    def add_layer(self, thickness_nm, n_index):
        """Add a layer to the coating stack"""
        self.layers.append((thickness_nm, n_index))
    
    def transfer_matrix(self, wavelength_nm, angle_deg=0):
        """
        Calculate the transfer matrix for the coating stack
        
        Returns: 2x2 transfer matrix and field distribution
        """
        angle_rad = np.deg2rad(angle_deg)
        k0 = 2 * np.pi / wavelength_nm
        
        # Air index
        n0 = 1.0
        ns = self.substrate_index
        
        # Calculate angles in each layer using Snell's law
        angles = []
        for thickness, n in self.layers:
            sin_theta = n0 * np.sin(angle_rad) / n
            if sin_theta <= 1:
                angles.append(np.arcsin(sin_theta))
            else:
                # Total internal reflection - use complex angle
                angles.append(np.arcsin(sin_theta + 0j))
        
        # Overall transfer matrix
        M = np.eye(2, dtype=complex)
        
        # Field amplitudes and positions for field distribution
        positions = [0]  # Start at air-coating interface
        field_amplitudes = []
        current_pos = 0
        
        for i, ((thickness, n), theta) in enumerate(zip(self.layers, angles)):
            # Phase thickness
            beta = k0 * n * np.cos(theta) * thickness
            
            # Layer transfer matrix
            cos_beta = np.cos(beta)
            sin_beta = np.sin(beta)
            
            # Optical admittance
            eta = n * np.cos(theta)
            
            M_layer = np.array([[cos_beta, 1j * sin_beta / eta],
                               [1j * eta * sin_beta, cos_beta]])
            
            M = M @ M_layer
            
            current_pos += thickness
            positions.append(current_pos)
        
        return M, positions
    
    def reflectance(self, wavelengths_nm, angle_deg=0):
        """Calculate reflectance vs wavelength"""
        R = []
        for wl in wavelengths_nm:
            M, _ = self.transfer_matrix(wl, angle_deg)
            
            # Fresnel coefficients
            eta0 = 1.0  # Air admittance
            etas = self.substrate_index  # Substrate admittance
            
            Y = M[1, 0] / M[0, 0]  # Input admittance
            r = (eta0 - Y) / (eta0 + Y)
            R.append(abs(r)**2)
        
        return np.array(R)
    
    def electric_field_distribution(self, wavelength_nm, num_points=1000):
        """
        Calculate electric field distribution through the coating
        
        Returns:
        z_positions: position array (nm)
        field_intensity: |E|^2 normalized to incident field
        """
        M, layer_positions = self.transfer_matrix(wavelength_nm)
        
        # Calculate reflection coefficient
        eta0 = 1.0
        Y = M[1, 0] / M[0, 0] if M[0, 0] != 0 else 1e-10
        r = (eta0 - Y) / (eta0 + Y)
        
        # Total coating thickness
        total_thickness = sum(thickness for thickness, _ in self.layers)
        z_positions = np.linspace(0, total_thickness, num_points)
        
        field_intensity = np.zeros(num_points)
        k0 = 2 * np.pi / wavelength_nm
        
        current_pos = 0
        layer_idx = 0
        
        for i, z in enumerate(z_positions):
            # Find which layer we're in
            while layer_idx < len(self.layers) and z > sum(thickness for thickness, _ in self.layers[:layer_idx+1]):
                layer_idx += 1
            
            if layer_idx >= len(self.layers):
                layer_idx = len(self.layers) - 1
            
            # Get layer properties
            if layer_idx == 0:
                layer_start = 0
            else:
                layer_start = sum(thickness for thickness, _ in self.layers[:layer_idx])
            
            thickness, n = self.layers[layer_idx]
            
            # Position within the layer
            z_in_layer = z - layer_start
            
            # Calculate field (simplified - assumes normal incidence)
            beta = k0 * n
            
            # Forward and backward propagating waves
            # This is a simplified model - in reality you'd need to solve for
            # the field coefficients at each interface
            forward_amplitude = 1.0  # Incident field
            backward_amplitude = r * np.exp(2j * beta * (total_thickness - z))
            
            # Total field
            total_field = forward_amplitude * np.exp(1j * beta * z) + backward_amplitude
            field_intensity[i] = abs(total_field)**2
        
        return z_positions, field_intensity

def find_field_extrema(z_positions, field_intensity, min_distance=50):
    """
    Find field maxima and minima for needle placement
    
    Parameters:
    z_positions: position array (nm)
    field_intensity: |E|^2 field intensity
    min_distance: minimum distance between extrema (nm)
    
    Returns:
    maxima_positions, minima_positions: arrays of positions where needles can be placed
    """
    # Convert min_distance to index spacing
    dz = z_positions[1] - z_positions[0]
    min_distance_idx = int(min_distance / dz)
    
    # Find peaks (maxima)
    maxima_idx, _ = find_peaks(field_intensity, distance=min_distance_idx)
    maxima_positions = z_positions[maxima_idx]
    
    # Find valleys (minima) by finding peaks of inverted signal
    minima_idx, _ = find_peaks(-field_intensity, distance=min_distance_idx)
    minima_positions = z_positions[minima_idx]
    
    return maxima_positions, minima_positions

class NeedleOptimizer:
    def __init__(self, base_coating, target_wavelengths, target_reflectance):
        """
        Needle optimization class
        
        Parameters:
        base_coating: OpticalCoating object (starting design)
        target_wavelengths: array of wavelengths (nm)
        target_reflectance: array of target reflectance values
        """
        self.base_coating = base_coating
        self.target_wavelengths = np.array(target_wavelengths)
        self.target_reflectance = np.array(target_reflectance)
        
        # Material options for needles
        self.high_index_material = 2.35  # TiO2
        self.low_index_material = 1.46   # SiO2
        
    def identify_needle_positions(self, analysis_wavelength=550, plot=False):
        """
        Identify optimal positions for needle placement based on field analysis
        
        Parameters:
        analysis_wavelength: wavelength for field analysis (nm)
        plot: whether to plot the field distribution
        
        Returns:
        candidate_positions: list of (position, type) where type is 'max' or 'min'
        """
        # Calculate field distribution
        z_pos, field_int = self.base_coating.electric_field_distribution(analysis_wavelength)
        
        # Find extrema
        maxima_pos, minima_pos = find_field_extrema(z_pos, field_int)
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(z_pos, field_int, 'b-', label='Field intensity |E|²')
            plt.scatter(maxima_pos, np.interp(maxima_pos, z_pos, field_int), 
                       color='red', s=50, label='Maxima (high-index needles)')
            plt.scatter(minima_pos, np.interp(minima_pos, z_pos, field_int), 
                       color='green', s=50, label='Minima (low-index needles)')
            plt.xlabel('Position (nm)')
            plt.ylabel('Field Intensity |E|²')
            plt.title(f'Electric Field Distribution at {analysis_wavelength} nm')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        # Create candidate positions
        candidates = []
        for pos in maxima_pos:
            candidates.append((pos, 'max'))
        for pos in minima_pos:
            candidates.append((pos, 'min'))
        
        # Sort by position
        candidates.sort(key=lambda x: x[0])
        
        return candidates
    
    def insert_needle(self, coating, position, needle_type, thickness=10):
        """
        Insert a needle into the coating at specified position
        
        Parameters:
        coating: OpticalCoating object
        position: position to insert needle (nm)
        needle_type: 'max' for high-index, 'min' for low-index
        thickness: needle thickness (nm)
        
        Returns:
        new_coating: OpticalCoating with needle inserted
        """
        new_coating = OpticalCoating(coating.substrate_index)
        
        # Choose needle material
        if needle_type == 'max':
            needle_index = self.high_index_material
        else:  # needle_type == 'min'
            needle_index = self.low_index_material
        
        current_pos = 0
        needle_inserted = False
        
        for layer_thickness, layer_index in coating.layers:
            layer_end = current_pos + layer_thickness
            
            # Check if needle should be inserted before this layer
            if not needle_inserted and position <= current_pos:
                new_coating.add_layer(thickness, needle_index)
                needle_inserted = True
            
            # Check if needle should be inserted within this layer
            elif not needle_inserted and current_pos < position < layer_end:
                # Split the layer
                part1_thickness = position - current_pos
                part2_thickness = layer_end - position
                
                if part1_thickness > 1:  # Only add if thickness > 1nm
                    new_coating.add_layer(part1_thickness, layer_index)
                
                new_coating.add_layer(thickness, needle_index)
                
                if part2_thickness > 1:  # Only add if thickness > 1nm
                    new_coating.add_layer(part2_thickness, layer_index)
                
                needle_inserted = True
            else:
                # Add original layer
                new_coating.add_layer(layer_thickness, layer_index)
            
            current_pos = layer_end
        
        # If needle wasn't inserted yet, add it at the end
        if not needle_inserted:
            new_coating.add_layer(thickness, needle_index)
        
        return new_coating
    
    def optimize_needles(self, candidate_positions, max_needles=5):
        """
        Optimize needle placement using iterative improvement
        
        Parameters:
        candidate_positions: list from identify_needle_positions()
        max_needles: maximum number of needles to add
        
        Returns:
        optimized_coating: OpticalCoating with optimized needles
        """
        current_coating = self.base_coating
        best_error = self.calculate_error(current_coating)
        
        print(f"Initial error: {best_error:.6f}")
        
        for needle_count in range(max_needles):
            best_improvement = 0
            best_needle = None
            
            # Try each candidate position
            for pos, needle_type in candidate_positions:
                # Try different thicknesses
                for thickness in [5, 10, 20, 30]:
                    test_coating = self.insert_needle(current_coating, pos, needle_type, thickness)
                    error = self.calculate_error(test_coating)
                    improvement = best_error - error
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_needle = (pos, needle_type, thickness, test_coating)
            
            # Add the best needle if it improves performance
            if best_needle is not None and best_improvement > 1e-6:
                pos, needle_type, thickness, new_coating = best_needle
                current_coating = new_coating
                best_error -= best_improvement
                print(f"Added needle {needle_count+1}: {needle_type} at {pos:.1f}nm, "
                      f"thickness {thickness}nm, new error: {best_error:.6f}")
                
                # Remove used position to avoid clustering
                candidate_positions = [(p, t) for p, t in candidate_positions 
                                     if abs(p - pos) > 50]
            else:
                print(f"No further improvement found after {needle_count} needles")
                break
        
        return current_coating
    
    def calculate_error(self, coating):
        """Calculate merit function (error) for optimization"""
        reflectance = coating.reflectance(self.target_wavelengths)
        return np.sum((reflectance - self.target_reflectance)**2)

# Example usage
def example_needle_optimization():
    """Example: Design a notch filter at 550nm"""
    
    # Create base coating (simple quarter-wave stack)
    base_coating = OpticalCoating(substrate_index=1.52)
    
    # Add alternating high/low index layers
    for i in range(10):
        if i % 2 == 0:
            base_coating.add_layer(95, 2.35)  # TiO2, λ/4 at 550nm
        else:
            base_coating.add_layer(94, 1.46)  # SiO2, λ/4 at 550nm
    
    # Define target: notch filter with high reflection at 550nm
    wavelengths = np.linspace(400, 700, 100)
    target_R = np.ones_like(wavelengths) * 0.02  # Low reflection everywhere
    
    # High reflection in notch region (540-560 nm)
    notch_mask = (wavelengths >= 540) & (wavelengths <= 560)
    target_R[notch_mask] = 0.99
    
    # Create optimizer
    optimizer = NeedleOptimizer(base_coating, wavelengths, target_R)
    
    # Identify needle positions
    print("Identifying needle positions...")
    candidates = optimizer.identify_needle_positions(analysis_wavelength=550, plot=True)
    
    print(f"Found {len(candidates)} candidate positions:")
    for i, (pos, needle_type) in enumerate(candidates[:10]):  # Show first 10
        print(f"  {i+1}. Position: {pos:.1f} nm, Type: {needle_type}")
    
    # Optimize needle placement
    print("\nOptimizing needle placement...")
    optimized_coating = optimizer.optimize_needles(candidates, max_needles=5)
    
    # Compare results
    print("\nComparing results...")
    original_R = base_coating.reflectance(wavelengths)
    optimized_R = optimized_coating.reflectance(wavelengths)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths, original_R, 'b-', label='Original coating')
    plt.plot(wavelengths, optimized_R, 'r-', label='With needles')
    plt.plot(wavelengths, target_R, 'k--', alpha=0.5, label='Target')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.title('Reflectance Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(wavelengths, abs(original_R - target_R), 'b-', label='Original error')
    plt.plot(wavelengths, abs(optimized_R - target_R), 'r-', label='Optimized error')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Optimization Error')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nOriginal coating layers: {len(base_coating.layers)}")
    print(f"Optimized coating layers: {len(optimized_coating.layers)}")
    print(f"Original error: {optimizer.calculate_error(base_coating):.6f}")
    print(f"Optimized error: {optimizer.calculate_error(optimized_coating):.6f}")

if __name__ == "__main__":
    example_needle_optimization()
