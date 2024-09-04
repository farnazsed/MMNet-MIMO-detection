import numpy as np


def generate_iid_gaussian_channel_matrices(num_samples, NT, NR):
    """
    Generates i.i.d. Gaussian channel matrices.

    Parameters:
    - num_samples: Number of channel matrices to generate.
    - NT: Number of transmit antennas.
    - NR: Number of receive antennas.

    Returns:
    - H_iid: Array of shape (num_samples, NR, NT) containing the channel matrices.
    """
    H_iid = (1 / np.sqrt(2)) * (np.random.randn(num_samples, NR, NT) + 1j * np.random.randn(num_samples, NR, NT))
    return H_iid


def generate_realistic_channel_matrices(num_samples, NT, NR, path_loss=1.0):
    """
    Generates realistic channel matrices using a Rayleigh fading model.

    Parameters:
    - num_samples: Number of channel matrices to generate.
    - NT: Number of transmit antennas.
    - NR: Number of receive antennas.
    - path_loss: Path loss factor to scale the channel matrices.

    Returns:
    - H_realistic: Array of shape (num_samples, NR, NT) containing the channel matrices.
    """
    H_realistic = (1 / np.sqrt(2)) * (np.random.randn(num_samples, NR, NT) + 1j * np.random.randn(num_samples, NR, NT))
    H_realistic *= path_loss
    return H_realistic


def save_channel_matrices(H, filename):
    """
    Saves the channel matrices to a .npy file.

    Parameters:
    - H: Channel matrices to save.
    - filename: Filename for the .npy file.
    """
    np.save(filename, H)
    print(f"Channel matrices saved to {filename}")


# Parameters
num_samples = 1000  # Number of channel matrices
NT = 16  # Number of transmit antennas
NR = 64  # Number of receive antennas

# Generate channel matrices
H_iid = generate_iid_gaussian_channel_matrices(num_samples, NT, NR)
H_realistic = generate_realistic_channel_matrices(num_samples, NT, NR, path_loss=0.5)

# Save channel matrices to .npy files
save_channel_matrices(H_iid, "H_iid.npy")
save_channel_matrices(H_realistic, "H_realistic.npy")
