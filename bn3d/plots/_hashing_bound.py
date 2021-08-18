"""
Make plots for hashing bound.
"""

import numpy as np
import itertools
import scipy.optimize
from scipy.interpolate import griddata


def get_project_axes():
    x_axis, y_axis, z_axis = np.eye(3)
    h_axis = y_axis - x_axis
    h_axis = h_axis/np.sqrt(h_axis.dot(h_axis))
    xy_midpoint = (x_axis + y_axis)/2
    v_axis = z_axis - xy_midpoint
    v_axis = v_axis/np.sqrt(v_axis.dot(v_axis))
    return h_axis, v_axis


def project_triangle(point):
    point = np.array(point)
    h_axis, v_axis = get_project_axes()
    h_coord = h_axis.dot(point)
    v_coord = v_axis.dot(point)
    return h_coord, v_coord


def reverse_project(hv_coords):
    h_coord, v_coord = hv_coords
    h_axis, v_axis = get_project_axes()
    center = np.ones(3)/3
    point = center + h_axis*h_coord + v_axis*v_coord
    return point


def get_eta_bias(point, axis=2):
    denominator = np.delete(point, axis).sum()
    if denominator == 0:
        return np.inf
    else:
        return point[axis]/denominator


def generate_points_triangle():
    r_z_list = [
        eta/(1 + eta)
        for eta in [0.5, 1, 3, 10, 30, 100]
    ] + [1.0]
    radials = (np.array(r_z_list) - 1/3)/(2/3)
    azimuthals = np.arange(0, 1, 1)
    points = generate_points(radials, azimuthals)

    combined_points = np.unique(np.concatenate([points]), axis=0)

    noise_parameters = [
        dict(zip(['r_x', 'r_y', 'r_z'], map(float, p + 0)))
        for p in combined_points
    ]

    return noise_parameters


def generate_points(radials, azimuthals):
    x_channel, y_channel, z_channel = np.eye(3)
    depolarizing = np.ones(3)/3
    axis_combinations = [
        (x_channel, y_channel),
        (y_channel, z_channel),
        (z_channel, x_channel),
    ]
    points = []
    for radial, azimuthal in itertools.product(radials, azimuthals):
        for axis_1, axis_2 in axis_combinations:
            points.append(
                (
                    depolarizing*(1 - radial)
                    + axis_1*radial
                )*(1 - azimuthal)
                + (
                    depolarizing*(1 - radial)
                    + axis_2*radial
                )*azimuthal
            )
    points = np.unique(points, axis=0)
    points = points.round(12)
    return points


def get_hashing_bound(point):
    r_x, r_y, r_z = point

    def max_rate(p):
        p_array = np.array([1 - p, p*r_x, p*r_y, p*r_z])
        h_array = np.zeros(4)
        for i in range(4):
            if p_array[i] != 0:
                h_array[i] = -p_array[i]*np.log2(p_array[i])
        entropy = h_array.sum()
        return 1 - entropy

    solutions = scipy.optimize.fsolve(max_rate, 0)
    return solutions[0]


def annotate_point(
    plt, point, func, offset=(0, 0.01), color='red', marker='^'
):
    value = func(point)
    h, v = project_triangle(point)
    plt.plot(h, v, marker, markersize=7, color=color)
    plt.text(
        h + offset[0], v + offset[1],
        '{:.2f}'.format(value),
        color=color,
        ha='left', va='bottom'
    )


def plot_sample_points(plt, points, markersize=1):
    horizontal_coords, vertical_coords = np.array([
        project_triangle(p) for p in points
    ]).T
    plt.plot(
        horizontal_coords, vertical_coords,
        'k.',
        markersize=markersize,
        label='Sample'
    )


def plot_hashing_bound(plt, pdf=None):
    """Plot hashing bound."""

    # Rough points to sample.
    r_z_list = [
        eta/(1 + eta)
        for eta in [0.5, 1, 3, 10, 30, 100, 300, 1000]
    ] + [1.0]
    radials = (np.array(r_z_list) - 1/3)/(2/3)
    azimuthals = np.arange(0, 1, 0.1)
    points = generate_points(radials, azimuthals)

    # Projected coordinates of points.
    horizontal_coords, vertical_coords = np.array([
        project_triangle(p) for p in points
    ]).T

    # Finer points for the hashing bound.
    fine_points = generate_points(
        np.linspace(0, 1, 51), np.linspace(0, 1, 51)
    )
    fine_hashing_bounds = np.array([
        get_hashing_bound(p) for p in fine_points
    ]).round(6)

    # Interpolated data.
    grid_h, grid_v = np.mgrid[
        min(horizontal_coords):max(horizontal_coords):300j,
        min(vertical_coords):max(vertical_coords):300j
    ]
    grid_hb = griddata(
        np.array([project_triangle(p) for p in fine_points]),
        fine_hashing_bounds,
        (grid_h, grid_v),
        method='cubic'
    )

    plt.contourf(
        grid_h,
        grid_v,
        grid_hb,
        extent=(0, 1, 0, 1),
        origin='lower',
        cmap='cividis',
        vmin=0,
        vmax=0.5
    )
    plt.plot(
        horizontal_coords, vertical_coords,
        'k.',
        markersize=1,
        label='Sample'
    )
    x_channel, y_channel, z_channel = np.eye(3)
    depolarizing = np.ones(3)/3
    plt.text(
        *(project_triangle(x_channel) + np.array([-0.05, -0.05])),
        'X', fontsize=16, color='red',
        va='center', ha='center', family='serif', style='italic'
    )
    plt.text(
        *(project_triangle(y_channel) + np.array([0.05, -0.05])),
        'Y', fontsize=16, color='darkgreen',
        va='center', ha='center', family='serif', style='italic'
    )
    plt.text(
        *(project_triangle(z_channel) + np.array([0, 0.05])),
        'Z', fontsize=16, color='darkblue',
        va='center', ha='center', family='serif', style='italic'
    )
    annotate_point(plt, depolarizing, get_hashing_bound, color='k')
    annotate_point(plt, x_channel, get_hashing_bound, color='red')
    annotate_point(
        plt, y_channel, get_hashing_bound, color='darkgreen',
        offset=(-0.05, 0.01)
    )
    annotate_point(
        plt, z_channel, get_hashing_bound, color='darkblue',
        offset=(0, -0.1)
    )
    annotate_point(
        plt, [0.5, 0, 0.5],
        get_hashing_bound,
        color='purple'
    )
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Zero-Rate Hashing Bound', fontsize=16)
    plt.gca().set_aspect(1)
    plt.legend(loc='upper left')

    if pdf is not None:
        plt.savefig(pdf, bbox_inches='tight')

    plt.show()
