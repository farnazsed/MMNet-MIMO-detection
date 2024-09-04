import numpy as np


def generate_gaussian_channel(num_samples, rows, cols, output_file):
    """
    Generate a Gaussian i.i.d. channel matrix with the specified shape and save it to a .npy file.

    Parameters:
    - num_samples: Number of channel realizations to generate
    - rows: Number of rows in the channel matrix
    - cols: Number of columns in the channel matrix
    - output_file: Path to the output .npy file
    """
    # Generate Gaussian i.i.d. entries
    H = np.random.randn(num_samples, rows, cols) + 1j * np.random.randn(num_samples, rows, cols)

    # Convert complex to real if needed (depends on what your model requires)
    H_real = np.concatenate([np.real(H), np.imag(H)], axis=-1)

    # Save to .npy file
    np.save(output_file, H_real)
    print(f'Gaussian channel matrix saved to {output_file}')


# Define parameters
num_samples = 500  # Number of channel realizations
rows = 64  # Number of rows in the channel matrix
cols = 16  # Number of columns in the channel matrix
output_file = 'gaussian_channel.npy'

# Generate and save the Gaussian i.i.d. channel matrix
generate_gaussian_channel(num_samples, rows, cols, output_file)
