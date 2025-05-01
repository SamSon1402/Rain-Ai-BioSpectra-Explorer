import numpy as np
from scipy import ndimage

def preprocess_images(brightfield, autofluorescence, hyperspectral):
    """
    Preprocess images from different modalities for analysis.
    
    Args:
        brightfield: Brightfield image or batch of images
        autofluorescence: Autofluorescence image or batch of images
        hyperspectral: Hyperspectral image or batch of images
        
    Returns:
        Preprocessed versions of each input modality
    """
    # Ensure inputs are numpy arrays
    brightfield = np.array(brightfield)
    autofluorescence = np.array(autofluorescence)
    hyperspectral = np.array(hyperspectral)
    
    # Apply preprocessing to brightfield images
    preprocessed_brightfield = []
    for img in brightfield:
        # Normalize
        norm_img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Enhance contrast
        p_low, p_high = np.percentile(norm_img, [2, 98])
        contrast_img = np.clip((norm_img - p_low) / (p_high - p_low + 1e-8), 0, 1)
        
        # Denoise
        denoised_img = ndimage.gaussian_filter(contrast_img, sigma=0.5)
        
        preprocessed_brightfield.append(denoised_img)
    
    # Apply preprocessing to autofluorescence images
    preprocessed_autofluorescence = []
    for img in autofluorescence:
        # Normalize
        norm_img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Background subtraction (simple threshold-based)
        bg_threshold = np.percentile(norm_img, 20)
        bg_subtracted = np.maximum(norm_img - bg_threshold, 0)
        
        # Normalize again after background subtraction
        if bg_subtracted.max() > 0:
            bg_subtracted = bg_subtracted / bg_subtracted.max()
        
        # Denoise
        denoised_img = ndimage.median_filter(bg_subtracted, size=2)
        
        preprocessed_autofluorescence.append(denoised_img)
    
    # Apply preprocessing to hyperspectral images
    preprocessed_hyperspectral = []
    for img in hyperspectral:
        # Process each channel
        processed_channels = []
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            
            # Normalize
            norm_channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            
            # Enhance contrast
            p_low, p_high = np.percentile(norm_channel, [5, 95])
            contrast_channel = np.clip((norm_channel - p_low) / (p_high - p_low + 1e-8), 0, 1)
            
            # Denoise
            denoised_channel = ndimage.gaussian_filter(contrast_channel, sigma=0.5)
            
            processed_channels.append(denoised_channel)
        
        # Stack channels back together
        processed_hyperspectral = np.stack(processed_channels, axis=2)
        preprocessed_hyperspectral.append(processed_hyperspectral)
    
    # Convert lists back to numpy arrays
    preprocessed_brightfield = np.array(preprocessed_brightfield)
    preprocessed_autofluorescence = np.array(preprocessed_autofluorescence)
    preprocessed_hyperspectral = np.array(preprocessed_hyperspectral)
    
    return preprocessed_brightfield, preprocessed_autofluorescence, preprocessed_hyperspectral

def align_modalities(brightfield, autofluorescence, hyperspectral):
    """
    Align different imaging modalities.
    
    In a real application, this would involve image registration techniques.
    For this demo, we'll simulate small misalignments and corrections.
    
    Args:
        brightfield: Brightfield images
        autofluorescence: Autofluorescence images
        hyperspectral: Hyperspectral images
        
    Returns:
        Aligned versions of each input modality
    """
    # Ensure inputs are numpy arrays
    brightfield = np.array(brightfield)
    autofluorescence = np.array(autofluorescence)
    hyperspectral = np.array(hyperspectral)
    
    # Number of samples
    n_samples = len(brightfield)
    
    # Aligned images containers
    aligned_brightfield = []
    aligned_autofluorescence = []
    aligned_hyperspectral = []
    
    for i in range(n_samples):
        # Get current images
        bf = brightfield[i]
        af = autofluorescence[i]
        hs = hyperspectral[i]
        
        # Simulate small misalignment by shifting autofluorescence and hyperspectral
        shift_x_af = np.random.randint(-2, 3)
        shift_y_af = np.random.randint(-2, 3)
        
        shift_x_hs = np.random.randint(-2, 3) 
        shift_y_hs = np.random.randint(-2, 3)
        
        # Apply shifts to simulate misalignment
        af_misaligned = ndimage.shift(af, (shift_y_af, shift_x_af), mode='nearest')
        
        # For hyperspectral, shift each channel
        hs_misaligned = np.zeros_like(hs)
        for c in range(hs.shape[2]):
            hs_misaligned[:, :, c] = ndimage.shift(
                hs[:, :, c], (shift_y_hs, shift_x_hs), mode='nearest'
            )
        
        # "Correct" the misalignment by shifting back
        af_aligned = ndimage.shift(af_misaligned, (-shift_y_af, -shift_x_af), mode='nearest')
        
        hs_aligned = np.zeros_like(hs_misaligned)
        for c in range(hs.shape[2]):
            hs_aligned[:, :, c] = ndimage.shift(
                hs_misaligned[:, :, c], (-shift_y_hs, -shift_x_hs), mode='nearest'
            )
        
        # Store aligned images
        aligned_brightfield.append(bf)  # Reference modality
        aligned_autofluorescence.append(af_aligned)
        aligned_hyperspectral.append(hs_aligned)
    
    # Convert lists back to numpy arrays
    aligned_brightfield = np.array(aligned_brightfield)
    aligned_autofluorescence = np.array(aligned_autofluorescence)
    aligned_hyperspectral = np.array(aligned_hyperspectral)
    
    return aligned_brightfield, aligned_autofluorescence, aligned_hyperspectral

def extract_features(brightfield, autofluorescence, hyperspectral):
    """
    Extract features from different imaging modalities.
    
    Args:
        brightfield: Brightfield images
        autofluorescence: Autofluorescence images
        hyperspectral: Hyperspectral images
        
    Returns:
        Extracted features from all modalities, combined
    """
    # Ensure inputs are numpy arrays
    brightfield = np.array(brightfield)
    autofluorescence = np.array(autofluorescence)
    hyperspectral = np.array(hyperspectral)
    
    # Number of samples
    n_samples = len(brightfield)
    
    # For simplicity in this demo, create some basic statistical features
    feature_vectors = []
    
    for i in range(n_samples):
        # Get current images
        bf = brightfield[i]
        af = autofluorescence[i]
        hs = hyperspectral[i]
        
        # Extract basic features from brightfield
        bf_mean = np.mean(bf)
        bf_std = np.std(bf)
        bf_min = np.min(bf)
        bf_max = np.max(bf)
        
        # Extract basic features from autofluorescence
        af_mean = np.mean(af)
        af_std = np.std(af)
        af_min = np.min(af)
        af_max = np.max(af)
        
        # Extract basic features from hyperspectral (per channel)
        hs_features = []
        for c in range(hs.shape[2]):
            channel = hs[:, :, c]
            hs_features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])
        
        # Combine all features
        combined_features = np.array([
            bf_mean, bf_std, bf_min, bf_max,
            af_mean, af_std, af_min, af_max,
            *hs_features
        ])
        
        # Add texture-based features (simplified)
        bf_grad_x = ndimage.sobel(bf, axis=0)
        bf_grad_y = ndimage.sobel(bf, axis=1)
        bf_grad_mag = np.sqrt(bf_grad_x**2 + bf_grad_y**2)
        
        texture_features = [
            np.mean(bf_grad_mag),
            np.std(bf_grad_mag),
            np.max(bf_grad_mag)
        ]
        
        # Add to combined features
        combined_features = np.append(combined_features, texture_features)
        
        # Add to feature vectors
        feature_vectors.append(combined_features)
    
    return np.array(feature_vectors)

if __name__ == "__main__":
    # Test with some dummy data
    from synthetic_data import generate_synthetic_brightfield, generate_synthetic_autofluorescence, generate_synthetic_hyperspectral
    
    size = 20
    samples = 3
    channels = 5
    
    bf = generate_synthetic_brightfield(samples, size)
    af = generate_synthetic_autofluorescence(samples, size)
    hs = generate_synthetic_hyperspectral(samples, size, channels=channels)
    
    # Test preprocessing
    print("Testing preprocessing...")
    bf_proc, af_proc, hs_proc = preprocess_images(bf, af, hs)
    print(f"Preprocessed shapes: BF {bf_proc.shape}, AF {af_proc.shape}, HS {hs_proc.shape}")
    
    # Test alignment
    print("\nTesting alignment...")
    bf_aligned, af_aligned, hs_aligned = align_modalities(bf, af, hs)
    print(f"Aligned shapes: BF {bf_aligned.shape}, AF {af_aligned.shape}, HS {hs_aligned.shape}")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    features = extract_features(bf, af, hs)
    print(f"Extracted features shape: {features.shape}")
    print(f"Features for first sample: {features[0][:5]}...")