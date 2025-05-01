import numpy as np
from scipy import ndimage

def generate_synthetic_brightfield(num_samples, size):
    """
    Generate synthetic brightfield images with simulated tissue structures.
    
    Args:
        num_samples: Number of images to generate
        size: Size of each image (size x size)
        
    Returns:
        Array of synthetic brightfield images
    """
    images = []
    
    for _ in range(num_samples):
        # Create base noise for tissue texture
        base = np.random.normal(0.5, 0.1, (size, size))
        
        # Add circular and elongated structures to simulate cells and tissue
        for _ in range(np.random.randint(3, 10)):
            # Random position
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            
            # Random radius/size
            r = np.random.randint(2, max(3, size // 5))
            
            # Random intensity (darker or lighter than surroundings)
            intensity = np.random.choice([0.2, 0.8])
            
            # Create structure mask (circle or ellipse)
            if np.random.random() > 0.5:
                # Circle
                y_grid, x_grid = np.ogrid[-y:size-y, -x:size-x]
                mask = x_grid*x_grid + y_grid*y_grid <= r*r
            else:
                # Ellipse with random orientation
                y_grid, x_grid = np.ogrid[-y:size-y, -x:size-x]
                aspect = np.random.uniform(0.3, 1.0)
                angle = np.random.uniform(0, 2*np.pi)
                x_rot = x_grid * np.cos(angle) - y_grid * np.sin(angle)
                y_rot = x_grid * np.sin(angle) + y_grid * np.cos(angle)
                mask = (x_rot*x_rot) / (r*r) + (y_rot*y_rot) / ((r*aspect)*(r*aspect)) <= 1
            
            # Apply structure to base image
            base[mask] = intensity
        
        # Smooth to make more realistic
        base = ndimage.gaussian_filter(base, sigma=0.5)
        
        # Ensure values are in [0, 1] range
        base = np.clip(base, 0, 1)
        
        images.append(base)
    
    return np.array(images)

def generate_synthetic_autofluorescence(num_samples, size, correlation_with_brightfield=0.7):
    """
    Generate synthetic autofluorescence images that have some correlation with brightfield data
    but also contain unique information.
    
    Args:
        num_samples: Number of images to generate
        size: Size of each image (size x size)
        correlation_with_brightfield: How much correlation with brightfield (0-1)
        
    Returns:
        Array of synthetic autofluorescence images
    """
    images = []
    
    # Generate brightfield images for correlation
    brightfield_images = generate_synthetic_brightfield(num_samples, size)
    
    for i in range(num_samples):
        # Start with some correlation to brightfield
        base = correlation_with_brightfield * brightfield_images[i]
        
        # Add autofluorescence-specific features (brighter spots)
        num_spots = np.random.randint(2, 8)
        for _ in range(num_spots):
            # Random position
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            
            # Random radius
            r = np.random.randint(1, max(2, size // 10))
            
            # Random intensity (bright spots)
            intensity = np.random.uniform(0.6, 1.0)
            
            # Create spot
            y_grid, x_grid = np.ogrid[-y:size-y, -x:size-x]
            mask = x_grid*x_grid + y_grid*y_grid <= r*r
            
            # Apply spot
            spot_contribution = np.zeros((size, size))
            spot_contribution[mask] = intensity
            spot_contribution = ndimage.gaussian_filter(spot_contribution, sigma=1.0)
            
            # Add to base image
            base += (1 - correlation_with_brightfield) * spot_contribution
        
        # Add noise
        noise = np.random.normal(0, 0.05, (size, size))
        base += (1 - correlation_with_brightfield) * noise
        
        # Ensure values are in [0, 1] range
        base = np.clip(base, 0, 1)
        
        # Apply gaussian filter to simulate light scatter
        base = ndimage.gaussian_filter(base, sigma=0.3)
        
        images.append(base)
    
    return np.array(images)

def generate_synthetic_hyperspectral(num_samples, size, channels=5):
    """
    Generate synthetic hyperspectral images with multiple channels.
    
    Args:
        num_samples: Number of images to generate
        size: Size of each image (size x size)
        channels: Number of spectral channels
        
    Returns:
        Array of synthetic hyperspectral images
    """
    images = []
    
    for _ in range(num_samples):
        # Initialize empty hyperspectral cube
        hyperspectral_image = np.zeros((size, size, channels))
        
        # Create base structure that's common across channels
        base_structure = np.random.normal(0.3, 0.1, (size, size))
        
        # Add some structures to the base
        for _ in range(np.random.randint(2, 5)):
            # Random position
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            
            # Random radius
            r = np.random.randint(2, max(3, size // 6))
            
            # Create structure mask
            y_grid, x_grid = np.ogrid[-y:size-y, -x:size-x]
            mask = x_grid*x_grid + y_grid*y_grid <= r*r
            
            # Apply structure to base
            base_structure[mask] = np.random.uniform(0.5, 0.9)
        
        # Smooth base structure
        base_structure = ndimage.gaussian_filter(base_structure, sigma=0.5)
        
        # Generate each spectral channel
        for channel in range(channels):
            # Start with base structure
            channel_image = base_structure.copy()
            
            # Add channel-specific features
            for _ in range(np.random.randint(1, 4)):
                # Random position
                x = np.random.randint(0, size)
                y = np.random.randint(0, size)
                
                # Random radius
                r = np.random.randint(1, max(2, size // 8))
                
                # Random intensity
                intensity = np.random.uniform(0.6, 1.0)
                
                # Create feature
                y_grid, x_grid = np.ogrid[-y:size-y, -x:size-x]
                mask = x_grid*x_grid + y_grid*y_grid <= r*r
                
                # Different channels have different probabilities of showing this feature
                channel_sensitivity = np.random.uniform(0, 1)
                
                if channel_sensitivity > 0.5:
                    # Apply feature to this channel
                    feature_contribution = np.zeros((size, size))
                    feature_contribution[mask] = intensity * channel_sensitivity
                    feature_contribution = ndimage.gaussian_filter(feature_contribution, sigma=0.8)
                    
                    channel_image += feature_contribution
            
            # Add some channel-specific noise
            channel_image += np.random.normal(0, 0.02, (size, size))
            
            # Ensure values are in [0, 1] range
            channel_image = np.clip(channel_image, 0, 1)
            
            # Add to hyperspectral cube
            hyperspectral_image[:, :, channel] = channel_image
        
        images.append(hyperspectral_image)
    
    return np.array(images)

def generate_synthetic_ground_truth(num_samples, size, biomarker_density=0.15):
    """
    Generate synthetic ground truth masks for biomarkers.
    
    Args:
        num_samples: Number of images to generate
        size: Size of each image (size x size)
        biomarker_density: Approximate density of biomarkers
        
    Returns:
        Array of synthetic ground truth masks
    """
    masks = []
    
    for _ in range(num_samples):
        # Create empty mask
        mask = np.zeros((size, size))
        
        # Determine number of biomarkers based on density and image size
        num_biomarkers = int(size * size * biomarker_density / 100) + np.random.randint(1, 5)
        
        for _ in range(num_biomarkers):
            # Random position
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            
            # Random radius (biomarkers are typically small)
            r = np.random.randint(1, max(2, size // 10))
            
            # Create biomarker mask
            y_grid, x_grid = np.ogrid[-y:size-y, -x:size-x]
            biomarker_mask = x_grid*x_grid + y_grid*y_grid <= r*r
            
            # Apply biomarker to mask with random intensity
            intensity = np.random.uniform(0.7, 1.0)
            mask[biomarker_mask] = intensity
        
        # Smooth edges of biomarkers
        mask = ndimage.gaussian_filter(mask, sigma=0.5)
        
        # Threshold to create clearer boundaries
        mask = np.where(mask > 0.3, mask, 0)
        
        # Normalize to [0, 1]
        if mask.max() > 0:
            mask = mask / mask.max()
        
        masks.append(mask)
    
    return np.array(masks)

if __name__ == "__main__":
    # Test data generation
    test_size = 20
    test_samples = 3
    
    bf = generate_synthetic_brightfield(test_samples, test_size)
    af = generate_synthetic_autofluorescence(test_samples, test_size)
    hs = generate_synthetic_hyperspectral(test_samples, test_size, channels=3)
    gt = generate_synthetic_ground_truth(test_samples, test_size)
    
    print(f"Generated test dataset with shapes:")
    print(f"Brightfield: {bf.shape}")
    print(f"Autofluorescence: {af.shape}")
    print(f"Hyperspectral: {hs.shape}")
    print(f"Ground Truth: {gt.shape}")