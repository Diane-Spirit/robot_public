"""
Module for encoding point cloud data in various XYZ-RGBA formats.

This module provides functions to encode point clouds using PyTorch, performing
operations such as stream compaction, handling of NaN/infinite values, and color conversions.
No dynamic memory allocation occurs within the encoding functions; all tensors must be preallocated.
"""

from enum import Enum
from typing import Tuple, Union
import numpy as np
import torch
import time
import random as rand  # Used for generating masks (uniform distribution)


###############################
#   Encoding Helper Functions
###############################

def to_XYZ_RGBA_f32_NaN(points: torch.Tensor, 
                        output: torch.Tensor, 
                        points_reshaped: torch.Tensor, 
                        x: torch.Tensor, 
                        y: torch.Tensor, 
                        z: torch.Tensor, 
                        c: torch.Tensor, 
                        default_color: float, 
                        n_points: int, 
                        multiplier: float = 1.0) -> torch.Tensor:
    """
    Encode point cloud data in XYZ-RGBA format (float32) without allocating extra memory.
    For any NaN color values, default_color is used.

    Args:
        points (torch.Tensor): Input tensor of shape [height, width, 4].
        output (torch.Tensor): Preallocated output tensor of shape [n_points * 4].
        points_reshaped (torch.Tensor): Preallocated tensor reshaped to [n_points, 4].
        x, y, z, c (torch.Tensor): Preallocated one-dimensional tensors for each channel.
        default_color (float): Default color value for invalid points.
        n_points (int): Total number of points.
        multiplier (float, optional): Scaling factor for x, y, z. Defaults to 1.0.

    Returns:
        torch.Tensor: The filled output tensor.
    """
    points_reshaped[:] = points.view(-1, 4)  # Reshape without reallocation

    x[:] = points_reshaped[:, 0]
    y[:] = points_reshaped[:, 1]
    z[:] = points_reshaped[:, 2]
    c[:] = points_reshaped[:, 3]

    # Replace NaN in color channel with default_color
    is_nan_c = torch.isnan(c)
    c[is_nan_c] = default_color

    # Fill XYZ components into output tensor
    output[:n_points * 3:3] = x * multiplier
    output[1:n_points * 3:3] = y * multiplier
    output[2:n_points * 3:3] = z * multiplier

    # Append color values
    output[n_points * 3:] = c
    return output


def to_XYZ_RGBA_f32(points: torch.Tensor, 
                    output: torch.Tensor, 
                    points_reshaped: torch.Tensor, 
                    x: torch.Tensor, y: torch.Tensor, 
                    z: torch.Tensor, c: torch.Tensor, 
                    default_color: float, 
                    nan_color: float, 
                    n_points: int, 
                    multiplier: float = 1.0) -> int:
    """
    Encode point cloud data in XYZ-RGBA format (float32).
    Applies distance-based masking and stream compaction, handling NaN values.
    
    Args:
        points (torch.Tensor): Input tensor of shape [height, width, 4].
        output (torch.Tensor): Preallocated output tensor of shape [n_points * 4].
        points_reshaped (torch.Tensor): Preallocated reshaped tensor of shape [n_points, 4].
        x, y, z, c (torch.Tensor): Preallocated tensors for each channel.
        default_color (float): Default color for invalid points.
        nan_color (float): Color to use in case of NaN.
        n_points (int): Total number of points.
        multiplier (float, optional): Scaling factor for spatial coordinates. Defaults to 1.0.
    
    Returns:
        int: The size (in bytes) of the encoded data (computed as num_valid * 16).
    """
    points_reshaped[:] = points.view(-1, 4)
    x[:] = points_reshaped[:, 0]
    y[:] = points_reshaped[:, 1]
    z[:] = points_reshaped[:, 2]
    c[:] = points_reshaped[:, 3]

    # Compute normalized Euclidean distance
    dist = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    dist = dist / torch.max(dist)

    # Create a random mask weighted by distance
    mask_dist = (torch.rand(n_points, device=points.device) <= dist)
    x[~mask_dist] = torch.nan

    # Replace NaN in color channel
    is_nan_c = torch.isnan(c)
    c[is_nan_c] = default_color

    # Mark invalid points from x (NaN or inf)
    invalid_mask = torch.isnan(x) | torch.isinf(x)
    is_valid = ~invalid_mask
    
    # Compute valid indices using prefix sum
    valid_indices = torch.cumsum(is_valid, dim=0) - 1
    num_valid = int(valid_indices[-1].item() + 1) if n_points > 0 else 0

    if num_valid == 0:
        return 0

    # Populate output tensor for valid points
    output[:num_valid * 3:3] = x * multiplier
    output[1:num_valid * 3:3] = y * multiplier
    output[2:num_valid * 3:3] = z * multiplier

    output[num_valid * 3:] = c
    return num_valid * 16


def to_XYZ_i16_RGBA_ui8(points: torch.Tensor, output: torch.Tensor, 
                          points_reshaped: torch.Tensor, x: torch.Tensor, 
                          y: torch.Tensor, z: torch.Tensor, c: torch.Tensor, 
                          default_color: float, nan_color: float, 
                          n_points: int, multiplier: float = 1.0) -> int:
    """
    Encode point cloud data to have int16 XYZ coordinates and an 8-bit packed RGBA color.
    Performs stream compaction to discard invalid points.
    
    Args:
        points (torch.Tensor): Input tensor with shape [height, width, 4].
        output (torch.Tensor): Preallocated output tensor.
        points_reshaped (torch.Tensor): Preallocated tensor of shape [n_points, 4].
        x, y, z, c (torch.Tensor): Preallocated tensors for each channel.
        default_color (float): Default color for invalid points.
        nan_color (float): Color for NaN values.
        n_points (int): Total number of points.
        multiplier (float, optional): Scale factor for spatial coordinates.
    
    Returns:
        torch.Tensor: The output tensor containing compacted encoded data.
    """
    points_reshaped[:] = points.view(-1, 4)
    x[:] = points_reshaped[:, 0]
    y[:] = points_reshaped[:, 1]
    z[:] = points_reshaped[:, 2]
    c[:] = points_reshaped[:, 3]
    
    is_nan_c = torch.isnan(c)
    c[is_nan_c] = default_color

    invalid_mask = torch.isnan(x) | torch.isinf(x) | (z * multiplier > torch.iinfo(torch.int16).max)
    is_valid = ~invalid_mask

    valid_indices = torch.cumsum(is_valid, dim=0) - 1
    num_valid = valid_indices[-1] + 1 if n_points > 0 else 0  # Number of valid points


    if num_valid == 0:
        return torch.empty(0, dtype=output.dtype, device=output.device)  # Return empty if no valid points

    c_rgb = c[is_valid].view(torch.uint8)
    # Convert RGB to RGB332 format
    rgb332 = (((c_rgb[:-1:4] >> 5) & 0b111) << 5) | (((c_rgb[1:-1:4] >> 5) & 0b111) << 2) | ((c_rgb[2:-1:4] >> 6) & 0b11)

    num_colors = (num_valid + 1) // 2
    rgb_packed = torch.zeros(num_colors, dtype=torch.int16, device=output.device)
    rgb_packed[:num_colors - (num_valid % 2)] = (rgb332[:-1:2].to(torch.int16) << 8) | rgb332[1::2].to(torch.int16)
    if num_valid % 2 == 1:
        rgb_packed[-1] = (rgb332[-1].to(torch.int16) << 8)

    # Fill output with compacted XYZ coordinates and color data.
    output[:num_valid * 3:3] = (x[is_valid] * multiplier).to(torch.int16)
    output[1:num_valid * 3:3] = (y[is_valid] * multiplier).to(torch.int16)
    output[2:num_valid * 3:3] = (z[is_valid] * multiplier).to(torch.int16)
    output[num_valid*3 : (num_valid*3) + num_colors] = rgb_packed
    

    #print("num_valid: ",num_valid)
    return (num_valid * 6) + num_valid + (num_valid % 2)



def to_XYZ_i16_RGBA_i16(points: torch.Tensor, output: torch.Tensor, points_reshaped: torch.Tensor, 
                    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, c: torch.Tensor, 
                    default_color: float, nan_color: float, n_points: int, multiplier: float = 1.0) -> int:
    """
    Encode point cloud data with int16 XYZ coordinates and RGBA colors split into two int16 values.
    Uses stream compaction to remove invalid points.

    Args:
        points (torch.Tensor): Input tensor [height, width, 4].
        output (torch.Tensor): Preallocated output tensor.
        points_reshaped (torch.Tensor): Preallocated reshaped tensor [n_points, 4].
        x, y, z, c (torch.Tensor): Preallocated channel tensors.
        default_color (float): Default color for invalid points.
        nan_color (float): Color for NaN values.
        n_points (int): Total number of points.
        multiplier (float, optional): Scaling factor. Defaults to 1.0.
    
    Returns:
        int: The number of bytes in the output data.
    """
    points_reshaped[:] = points.view(-1, 4)
    x[:] = points_reshaped[:, 0]
    y[:] = points_reshaped[:, 1]
    z[:] = points_reshaped[:, 2]
    c[:] = points_reshaped[:, 3]
    
    is_nan_c = torch.isnan(c)
    c[is_nan_c] = default_color

    invalid_mask = torch.isnan(x) | torch.isinf(x) | (z * multiplier > torch.iinfo(torch.int16).max)
    is_valid = ~invalid_mask
    valid_indices = torch.cumsum(is_valid, dim=0) - 1
    num_valid = valid_indices[-1] + 1 if n_points > 0 else 0  # Number of valid points


    if num_valid == 0:
        return torch.empty(0, dtype=output.dtype, device=output.device)  # Return empty if no valid points

    c_i32 = (c[is_valid]).view(torch.int32)

    

    # Fill the output tensor with the compacted XYZ values
    output[:num_valid * 3:3] = (x[is_valid] * multiplier).to(torch.int16)
    output[1:num_valid * 3:3] = (y[is_valid] * multiplier).to(torch.int16)
    output[2:num_valid * 3:3] = (z[is_valid] * multiplier).to(torch.int16)
    output[num_valid * 3 + 0:num_valid * 5:2] = (c_i32 & 0xFFFF).to(torch.int16)   
    output[num_valid * 3 + 1:num_valid * 5:2] = (c_i32 >> 16).to(torch.int16)

    return num_valid


def to_XYZ_RGBA_i16_NaN(points: torch.Tensor, 
                         output: torch.Tensor, 
                         points_reshaped: torch.Tensor, 
                         x: torch.Tensor, y: torch.Tensor, 
                         z: torch.Tensor, c: torch.Tensor, 
                         default_color: float, n_points: int, 
                         multiplier: float = 1.0) -> int:
    """
    Encode point cloud data in XYZ-RGBA format where coordinates are int16,
    handling NaN values. Tensors are preallocated.

    Args:
        points (torch.Tensor): Input tensor [height, width, 4].
        output (torch.Tensor): Preallocated output tensor.
        points_reshaped (torch.Tensor): Preallocated reshaped tensor [n_points, 4].
        x, y, z, c (torch.Tensor): Preallocated channel tensors.
        default_color (float): Default color for invalid points.
        n_points (int): Total number of points.
        multiplier (float, optional): Scaling factor. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Encoded output tensor.
    """
    points_reshaped[:] = points.view(-1, 4)
    x[:] = points_reshaped[:, 0]
    y[:] = points_reshaped[:, 1]
    z[:] = points_reshaped[:, 2]
    c[:] = points_reshaped[:, 3]

    is_nan_c = torch.isnan(c)
    c[is_nan_c] = default_color

    output[:n_points * 3:3] = (x * multiplier).to(torch.int16)
    output[1:n_points * 3:3] = (y * multiplier).to(torch.int16)
    output[2:n_points * 3:3] = (z * multiplier).to(torch.int16)

    output[n_points * 3 + 0:n_points * 5:2] = (c.view(torch.int32) & 0xFFFF).to(torch.int16)
    output[n_points * 3 + 1:n_points * 5:2] = (c.view(torch.int32) >> 16).to(torch.int16)

    return output


###############################
#      DIANE MULTINOMIAL
###############################

def diane_multinomial_i16(points: torch.Tensor, 
                           output: torch.Tensor, 
                           points_reshaped: torch.Tensor, 
                           x: torch.Tensor, y: torch.Tensor, 
                           z: torch.Tensor, c: torch.Tensor, 
                           lut_idx_x: torch.Tensor, lut_idx_y: torch.Tensor,
                           lut_idx_z: torch.Tensor, lut_idx_c: torch.Tensor,
                           bw: float, fps: float, dist_guarant: float,
                           default_color: float, nan_color: float, 
                           n_points: int, multiplier: float = 1.0, squid: int = 0) -> int:
    """
    Encode point cloud data using a multinomial sampling strategy based on point priorities.
    A priority is computed based on the inverse normalized distance and validity, with an extra boost 
    for points within a guaranteed distance. Then, a multinomial sampling selects the points to be transmitted.

    Args:
        points (torch.Tensor): Input tensor [height, width, 4].
        output (torch.Tensor): Output buffer (preallocated).
        points_reshaped (torch.Tensor): Preallocated reshaped tensor [n_points, 4].
        x, y, z, c (torch.Tensor): Preallocated channel tensors.
        lut_idx_x, lut_idx_y, lut_idx_z, lut_idx_c (torch.Tensor): Lookup indices for writing output.
        bw (float): Available bandwidth (bits per second).
        fps (float): Desired frame rate (fps).
        dist_guarant (float): Guaranteed distance threshold (meters).
        default_color (float): Default color for invalid points.
        nan_color (float): Color for NaN values.
        n_points (int): Total number of points.
        multiplier (float, optional): Scaling factor. Defaults to 1.0.
        squid (int, optional): Flag for special color processing. Defaults to 0.
    
    Returns:
        int: Size (in bytes) of the encoded output buffer.
    """
    PSZ = 4 * 16  # bits per point
    MFS = bw / fps  # Maximum frame size in bits

    max_points = int(np.min([MFS / PSZ, n_points]))
    max_points = int(np.max([max_points, int((n_points * 0.10) / fps)]))
    max_points = int((max_points // 150) * 150)

    print(max_points)  # Consider replacing with structured logging

    points_reshaped[:] = points.view(-1, 4)
    x[:] = points_reshaped[:, 0]
    y[:] = points_reshaped[:, 1]
    z[:] = points_reshaped[:, 2]
    c[:] = points_reshaped[:, 3]

    is_nan_c = torch.isnan(c)
    c[is_nan_c] = default_color

    invalid_mask = torch.isnan(x) | torch.isinf(x) | (z * multiplier > torch.iinfo(torch.int16).max)
    # Compute priority: inverse of normalized distance multiplied by valid mask
    is_valid = torch.nan_to_num((1/( (z/(torch.max(z[~invalid_mask])+1e-6)) +1e-6)) * ~invalid_mask, nan=0) # priority based on distance and validity

    # Boost priority for points within the distance guarantee
    is_valid[z < dist_guarant] = torch.max(is_valid)
    
    # Sample valid indices based on computed priority
    selected_indices = torch.multinomial(is_valid, max_points, replacement=False)
    num_valid = x[selected_indices].shape[0]
    
    if num_valid == 0:
        return torch.empty(0, dtype=output.dtype, device=output.device)  # Return empty if no valid points

    c_rgb = c[selected_indices].view(torch.uint8).to(torch.int16)
    rgb664 = (((c_rgb[:-1:4] >> 2) & 0b111111) << 10) | (((c_rgb[1:-1:4] >> 2) & 0b111111) << 4) | ((c_rgb[2:-1:4] >> 4) & 0b1111)


    color_squid = 0x03F0

    target_idx_x = lut_idx_x[:num_valid]
    target_idx_y = lut_idx_y[:num_valid]
    target_idx_z = lut_idx_z[:num_valid]
    target_idx_c = lut_idx_c[:num_valid]


    # Fill the output tensor with the compacted XYZ values
    output[target_idx_x] = (x[selected_indices] * multiplier).to(torch.int16)
    output[target_idx_y] = (y[selected_indices] * multiplier).to(torch.int16)
    output[target_idx_z] = (z[selected_indices] * multiplier).to(torch.int16)
    if squid == 0:
        output[target_idx_c] = rgb664
    else:
        output[target_idx_c] = color_squid
    

    #print("num_valid: ",num_valid)
    return num_valid * 8


def diane_multinomial_temporal_i16(points: torch.Tensor, 
                                    output: torch.Tensor, 
                                    points_reshaped: torch.Tensor, 
                                    x: torch.Tensor, y: torch.Tensor, 
                                    z: torch.Tensor, c: torch.Tensor, 
                                    bw: float, fps: float, dist_guarant: float, 
                                    _unused_is_valid,  # Provided for interface compatibility
                                    default_color: float, nan_color: float, 
                                    n_points: int, multiplier: float = 1.0) -> int:
    """
    Temporal variant of the multinomial encoding to select points over time.
    The function follows similar logic to diane_multinomial_i16.
    
    Args:
        points (torch.Tensor): Input tensor [height, width, 4].
        output (torch.Tensor): Output buffer.
        points_reshaped (torch.Tensor): Preallocated reshaped tensor.
        x, y, z, c (torch.Tensor): Preallocated channel tensors.
        bw (float): Available bandwidth (bps).
        fps (float): Frame rate.
        dist_guarant (float): Guaranteed distance.
        _unused_is_valid: Unused parameter.
        default_color (float): Default color.
        nan_color (float): Color for NaN.
        n_points (int): Total number of points.
        multiplier (float, optional): Scale factor.
    
    Returns:
        int: Size (in bytes) of encoded data.
    """
    # For brevity, similar logic as diane_multinomial_i16 is applied.
    PSZ = 4 * 16
    MFS = bw / fps
    max_points = int(np.min([MFS / PSZ, n_points]))
    points_reshaped[:] = points.view(-1, 4)
    x[:] = points_reshaped[:, 0]
    y[:] = points_reshaped[:, 1]
    z[:] = points_reshaped[:, 2]
    c[:] = points_reshaped[:, 3]
    
    is_nan_c = torch.isnan(c)
    c[is_nan_c] = default_color

    invalid_mask = torch.isnan(x) | torch.isinf(x) | (z * multiplier > torch.iinfo(torch.int16).max)
    is_valid = torch.nan_to_num((1 / ((z/(torch.max(z[~invalid_mask])+1e-6)) + 1e-6)) * (~invalid_mask), nan=0)
    is_valid[z < dist_guarant] = torch.max(is_valid)
    selected_indices = torch.multinomial(is_valid, max_points, replacement=False)
    num_valid = x[selected_indices].shape[0]

    if num_valid == 0:
        return torch.empty(0, dtype=output.dtype, device=output.device)  # Return empty if no valid points

    c_rgb = c[selected_indices].view(torch.uint8).to(torch.int16)
    rgb664 = (((c_rgb[:-1:4] >> 2) & 0b111111) << 10) | (((c_rgb[1:-1:4] >> 2) & 0b111111) << 4) | ((c_rgb[2:-1:4] >> 4) & 0b1111)
    
    output[:num_valid * 3:3] = (x[selected_indices] * multiplier).to(torch.int16)
    output[1:num_valid * 3:3] = (y[selected_indices] * multiplier).to(torch.int16)
    output[2:num_valid * 3:3] = (z[selected_indices] * multiplier).to(torch.int16)
    output[num_valid*3:num_valid*4] = rgb664
    return num_valid * 8


###############################
#          Enums & Utils
###############################

class Encoding(Enum):
    XYZ_RGBA_f32 = 0
    XYZ_RGBA_f32_NaN = 1
    XYZ_RGBA_i16 = 2
    XYZ_RGBA_i16_NaN = 3
    XYZ_i16_RGBA_ui8 = 4
    DIANE_MULTINOMIAL_i16 = 5
    default = XYZ_RGBA_f32_NaN


def color_to_float(color: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0)) -> float:
    """
    Convert an RGBA tuple to a packed 32-bit float.
    
    Args:
        color (tuple): RGBA values.
    
    Returns:
        float: 32-bit float representation.
    """
    r, g, b, a = color
    packed = (int(r * 255) << 24) | (int(g * 255) << 16) | (int(b * 255) << 8) | int(a * 255)
    return float(np.frombuffer(packed.to_bytes(4, byteorder='little'), dtype=np.float32)[0])


def new_lut(output_size: int, block_size: int = 150, elements_per_coordinate: int = 3, 
            elements_per_color: int = 1, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate lookup tables for writing output in blocks. This facilitates organized 
    writes in the DIANE multinomial encoding.

    Args:
        output_size (int): Total size of the output buffer.
        block_size (int): Number of points per block.
        elements_per_coordinate (int): Elements (e.g., 3 for XYZ) per point.
        elements_per_color (int): Elements for color.
        device (str): Torch device.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Lookup indices for x, y, z, and color.
    """
    elements_per_block_xyz = block_size * elements_per_coordinate
    elements_per_block_color = block_size * elements_per_color
    elements_per_block = elements_per_block_xyz + elements_per_block_color
    points_lut = output_size // (elements_per_coordinate + elements_per_color)

    k_indices = torch.arange(points_lut, device=device, dtype=torch.long)
    block_idx = k_indices // block_size
    idx_within_block = k_indices % block_size
    output_block_start = block_idx * elements_per_block

    lut_idx_x = output_block_start + idx_within_block * elements_per_coordinate
    lut_idx_y = lut_idx_x + 1
    lut_idx_z = lut_idx_x + 2
    lut_idx_c = output_block_start + elements_per_block_xyz + idx_within_block 

    return lut_idx_x, lut_idx_y, lut_idx_z, lut_idx_c


###############################
#          Encoder Class
###############################

class Encoder:
    """
    Encoder class for point cloud data.

    Attributes:
        resolution: Resolution object with 'height' and 'width' attributes.
        encoding (Encoding): Selected encoding type.
        default_color (float): Packed float default color.
        nan_color (float): Packed float for NaN.
    """
    def __init__(self, resolution, encoding: Encoding = Encoding.default, 
                 default_color: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0), 
                 nan_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)):
        self.resolution = resolution
        self.n_points = resolution.height * resolution.width
        self.encoding = encoding
        self.default_color = color_to_float(default_color)
        self.nan_color = color_to_float(nan_color)
        self._init_fields()

    def _init_fields(self) -> None:
        """Initialize input and output fields (tensors) based on resolution and chosen encoding."""
        self.points = torch.empty((self.resolution.height, self.resolution.width, 4),
                                  dtype=torch.float32, device='cuda')
        self.points_reshaped = torch.empty((self.n_points, 4),
                                           dtype=torch.float32, device='cuda')
        self.x = torch.empty((self.n_points,), dtype=torch.float32, device='cuda')
        self.y = torch.empty((self.n_points,), dtype=torch.float32, device='cuda')
        self.z = torch.empty((self.n_points,), dtype=torch.float32, device='cuda')
        self.c = torch.empty((self.n_points,), dtype=torch.float32, device='cuda')
        
        if self.encoding in {Encoding.XYZ_RGBA_f32, Encoding.XYZ_RGBA_f32_NaN}:
            self.output = torch.empty((self.n_points * 4,), dtype=torch.float32, device='cuda')
        elif self.encoding == Encoding.XYZ_RGBA_i16:
            self.output = torch.empty((self.n_points * 5,), dtype=torch.int16, device='cuda')
        elif self.encoding == Encoding.XYZ_RGBA_i16_NaN:
            self.output = torch.empty((self.n_points * 5,), dtype=torch.int16, device='cuda')
        elif self.encoding == Encoding.XYZ_i16_RGBA_ui8:
            self.output = torch.empty((self.n_points * 5,), dtype=torch.int16, device='cuda')
        elif self.encoding == Encoding.DIANE_MULTINOMIAL_i16:
            self.lut_idx_x, self.lut_idx_y, self.lut_idx_z, self.lut_idx_c = new_lut(
                output_size=self.n_points * 8,
                block_size=150,
                elements_per_coordinate=3,
                elements_per_color=1,
                device='cuda'
            )
            self.output = torch.empty((self.n_points * 5,), dtype=torch.int16, device='cuda')
        else:
            raise ValueError("Unsupported encoding type.")

    def encode(self, numpy_pc: np.ndarray, bw: float, fps: float = 20, 
               dist_guarant: float = 2.0, squid: int = 0) -> bytes:
        """
        Encode the provided point cloud into a bytes object.

        Args:
            numpy_pc (np.ndarray): Input point cloud as a numpy array.
            bw (float): Available bandwidth in bits per second.
            fps (float, optional): Frames per second. Defaults to 20.
            dist_guarant (float, optional): Guaranteed distance threshold. Defaults to 2.0.
            squid (int, optional): Special processing flag. Defaults to 0.

        Returns:
            bytes: The encoded point cloud data.
        """
        self.points[:] = torch.tensor(numpy_pc, dtype=torch.float32, device='cuda')
        bufferSize = None

        if self.encoding == Encoding.XYZ_RGBA_f32:
            bufferSize = to_XYZ_RGBA_f32(
                self.points, self.output, self.points_reshaped,
                self.x, self.y, self.z, self.c,
                self.default_color, self.nan_color,
                self.n_points
            )
        elif self.encoding == Encoding.XYZ_RGBA_f32_NaN:
            to_XYZ_RGBA_f32_NaN(
                self.points, self.output, self.points_reshaped,
                self.x, self.y, self.z, self.c,
                self.default_color, self.n_points
            )
        elif self.encoding == Encoding.XYZ_RGBA_i16:
            bufferSize = to_XYZ_i16_RGBA_i16(
                self.points, self.output, self.points_reshaped,
                self.x, self.y, self.z, self.c,
                self.default_color, self.nan_color,
                self.n_points, 10.0
            )
        elif self.encoding == Encoding.XYZ_RGBA_i16_NaN:
            to_XYZ_RGBA_i16_NaN(
                self.points, self.output, self.points_reshaped,
                self.x, self.y, self.z, self.c,
                self.default_color, self.n_points, 10.0
            )
        elif self.encoding == Encoding.XYZ_i16_RGBA_ui8:
            bufferSize = to_XYZ_i16_RGBA_ui8(
                self.points, self.output, self.points_reshaped,
                self.x, self.y, self.z, self.c,
                self.default_color, self.nan_color,
                self.n_points, 10.0
            )
        elif self.encoding == Encoding.DIANE_MULTINOMIAL_i16:
            bufferSize = diane_multinomial_i16(
                self.points, self.output, self.points_reshaped,
                self.x, self.y, self.z, self.c,
                self.lut_idx_x, self.lut_idx_y, self.lut_idx_z, self.lut_idx_c,
                bw, fps, dist_guarant,
                self.default_color, self.nan_color,
                self.n_points, 10.0, squid
            )
        else:
            raise ValueError("Invalid encoding type.")
        
        output_pinned = self.output.to("cpu", non_blocking=True).pin_memory()
        if bufferSize is not None:
            return output_pinned.numpy().tobytes()[:bufferSize]
        else:
            return self.output.cpu().numpy().tobytes()