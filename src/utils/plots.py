"""plot.py"""

import os
import imageio
import cv2
import torch
import numpy as np
import open3d as o3d
import time
import pyvista as pv
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize


def plot_2D_image(z_net, z_gt, step, var=5, output_dir='outputs'):
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 1)
    ax3 = fig.add_subplot(1, 3, 3)

    # Oculta los bordes de los ejes
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax1.spines['left'].set_color((0.8, 0.8, 0.8))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax2.spines['left'].set_color((0.8, 0.8, 0.8))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax3.spines['left'].set_color((0.8, 0.8, 0.8))

    ax1.set_title('Thermodynamics-informed GNN')
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth')
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')
    ax3.set_title('Thermodynamics-informed GNN error')
    ax3.set_xlabel('X'), ax3.set_ylabel('Y')

    # Asegura una escala igual en ambos ejes
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')

    # Adjust ranges
    X, Y = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy()
    z_min, z_max = z_gt[step, :, var].min(), z_gt[step, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    # Initial snapshot
    q1_net, q3_net = z_net[step, :, 0], z_net[step, :, 1]
    q1_gt, q3_gt = z_gt[step, :, 0], z_gt[step, :, 1]
    var_net, var_gt = z_net[step, :, var], z_gt[step, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb in zip(Xb, Yb):
        ax1.plot([xb], [yb], 'w')
        ax2.plot([xb], [yb], 'w')
        ax3.plot([xb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q3_net, c=var_net,
                     vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max,
                     vmin=z_min)
    s3 = ax3.scatter(q1_net, q3_net, c=var_error,
                     vmax=var_error_max, vmin=var_error_min)
    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    fig.savefig(os.path.join(output_dir, f'beam_{step}.png'))

    # Oculta las marcas de los ejes y las etiquetas
    ax1.tick_params(axis='both', which='both', length=0)
    plt.savefig(os.path.join(output_dir, f'beam.svg'), format="svg")


def plot_2D(z_net, z_gt, save_dir, var=5):
    T = z_net.size(0)
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 1)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')
    ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
    ax3.set_xlabel('X'), ax3.set_ylabel('Y')

    # Adjust ranges
    X, Y = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy()
    z_min, z_max = z_gt[:, :, var].min(), z_gt[:, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())

    # Initial snapshot
    q1_net, q3_net = z_net[0, :, 0], z_net[0, :, 1]
    q1_gt, q3_gt = z_gt[0, :, 0], z_gt[0, :, 1]
    var_net, var_gt = z_net[-1, :, var], z_gt[-1, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb in zip(Xb, Yb):
        ax1.plot([xb], [yb], 'w')
        ax2.plot([xb], [yb], 'w')
        ax3.plot([xb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q3_net, c=var_net,
                     vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max,
                     vmin=z_min)
    s3 = ax3.scatter(q1_net, q3_net, c=var_error,
                     vmax=var_error_max, vmin=var_error_min)

    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.set_title(f'Thermodynamics-informed GNN, f={str(snap)}'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y')
        ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
        ax3.set_xlabel('X'), ax3.set_ylabel('Y')
        # Bounding box
        for xb, yb in zip(Xb, Yb):
            ax1.plot([xb], [yb], 'w')
            ax2.plot([xb], [yb], 'w')
            ax3.plot([xb], [yb], 'w')
        # Scatter points
        q1_net, q3_net = z_net[snap, :, 0], z_net[snap, :, 1]
        q1_gt, q3_gt = z_gt[snap, :, 0], z_gt[snap, :, 1]

        var_net, var_gt = z_net[snap, :, var], z_gt[snap, :, var]
        var_error = var_gt - var_net

        ax1.scatter(q1_net, q3_net, c=var_net,
                    vmax=z_max, vmin=z_min)
        ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max,
                    vmin=z_min)
        ax3.scatter(q1_net, q3_net, c=var_error,
                    vmax=var_error_max, vmin=var_error_min)
        # fig.savefig(os.path.join('images/', f'beam_{snap}.png'))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=20)

    # Save as gif
    # save_dir = os.path.join(output_dir, 'beam.mp4')
    anim.save(save_dir, writer=writergif)
    plt.close('all')


def plot_1D(z_net, z_gt, q_0, save_dir, var=0):
    y_var = 0
    q_0 = q_0.cpu()
    T = z_net.size(0)
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 1)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y')
    ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
    ax3.set_xlabel('X'), ax3.set_ylabel('Y')

    # Adjust ranges
    X, Y = q_0[:, 0].numpy(), z_gt[:, :, y_var].numpy()
    z_min, z_max = z_gt[:, :, var].min(), z_gt[:, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min()]).max()
    Xb = 0.5 * X.max()  * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())

    # Initial snapshot
    q1_net, q3_net = X, z_net[0, :, y_var]
    q1_gt, q3_gt = X, z_gt[0, :, y_var]
    var_net, var_gt = z_net[-1, :, var], z_gt[-1, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb in zip(Xb, Yb):
        ax1.plot([xb], [yb], 'w')
        ax2.plot([xb], [yb], 'w')
        ax3.plot([xb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q3_net, c=var_net,
                     vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max,
                     vmin=z_min)
    s3 = ax3.scatter(q1_net, q3_net, c=var_error,
                     vmax=var_error_max, vmin=var_error_min)

    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.set_title(f'Thermodynamics-informed GNN, f={str(snap)}'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y')
        ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
        ax3.set_xlabel('X'), ax3.set_ylabel('Y')
        # Bounding box
        for xb, yb in zip(Xb, Yb):
            ax1.plot([xb], [yb], 'w')
            ax2.plot([xb], [yb], 'w')
            ax3.plot([xb], [yb], 'w')
        # Scatter points
        q1_net, q3_net = X, z_net[snap, :, y_var]
        q1_gt, q3_gt = X, z_gt[snap, :, y_var]

        var_net, var_gt = z_net[snap, :, var], z_gt[snap, :, var]
        var_error = var_gt - var_net

        ax1.scatter(q1_net, q3_net, c=var_net,
                    vmax=z_max, vmin=z_min)
        ax2.scatter(q1_gt, q3_gt, c=var_gt, vmax=z_max,
                    vmin=z_min)
        ax3.scatter(q1_net, q3_net, c=var_error,
                    vmax=var_error_max, vmin=var_error_min)
        # fig.savefig(os.path.join('images/', f'beam_{snap}.png'))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=20)

    # Save as gif
    # save_dir = os.path.join(output_dir, 'beam.mp4')
    anim.save(save_dir, writer=writergif)
    plt.close('all')


def plot_3D(z_net, z_gt, save_dir, var=5):
    T = z_net.size(0)

    fig = plt.figure(figsize=(30, 20))
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2 = fig.add_subplot(1, 3, 1, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
    ax2.set_title('Ground Truth'), ax2.grid()
    ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')
    ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
    ax3.set_xlabel('X'), ax3.set_ylabel('Y'), ax3.set_zlabel('Z')
    ax1.view_init(elev=0., azim=90)
    ax2.view_init(elev=0., azim=90)
    ax3.view_init(elev=0., azim=90)

    # Adjust ranges
    X, Y, Z = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy(), z_gt[:, :, 2].numpy()
    z_min, z_max = z_gt[:, :, var].min(), z_gt[:, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    # Initial snapshot
    q1_net, q2_net, q3_net = z_net[0, :, 0], z_net[0, :, 2], z_net[0, :, 1]
    q1_gt, q2_gt, q3_gt = z_gt[0, :, 0], z_gt[0, :, 2], z_gt[0, :, 1]
    var_net, var_gt = z_net[-1, :, var], z_gt[-1, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax1.plot([xb], [zb], [yb], 'w')
        ax2.plot([xb], [zb], [yb], 'w')
        ax3.plot([xb], [zb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q2_net, q3_net, c=var_net, s=100, alpha=0.5,
                     vmax=z_max, vmin=z_min)
    s2 = ax2.scatter(q1_gt, q2_gt, q3_gt, c=var_gt, vmax=z_max, s=100, alpha=0.5,
                     vmin=z_min)
    s3 = ax3.scatter(q1_net, q2_net, q3_net, c=var_error, s=100, alpha=0.5,
                     vmax=var_error_max, vmin=var_error_min)

    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)

    # Animation
    def animate(snap):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.set_title(f'Thermodynamics-informed GNN, f={str(snap)}'), ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        ax2.set_title('Ground Truth'), ax2.grid()
        ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')
        ax3.set_title('Thermodynamics-informed GNN error'), ax3.grid()
        ax3.set_xlabel('X'), ax3.set_ylabel('Y'), ax3.set_zlabel('Z')
        # Bounding box
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [zb], [yb], 'w')
            ax2.plot([xb], [zb], [yb], 'w')
            ax3.plot([xb], [zb], [yb], 'w')
        # Scatter points
        q1_net, q2_net, q3_net = z_net[snap, :, 0], z_net[snap, :, 2], z_net[snap, :, 1]
        q1_gt, q2_gt, q3_gt = z_gt[snap, :, 0], z_gt[snap, :, 2], z_gt[snap, :, 1]
        var_net, var_gt = z_net[snap, :, var], z_gt[snap, :, var]
        var_error = var_gt - var_net
        ax1.scatter(q1_net, q2_net, q3_net, c=var_net, s=100, alpha=0.5,
                    vmax=z_max, vmin=z_min)
        ax2.scatter(q1_gt, q2_gt, q3_gt, c=var_gt, vmax=z_max, s=100, alpha=0.5,
                    vmin=z_min)
        ax3.scatter(q1_net, q2_net, q3_net, c=var_error, s=100, alpha=0.5,
                    vmax=var_error_max, vmin=var_error_min)
        fig.savefig(os.path.join(r'/home/atierz/Documentos/experiments/Foam_visco/3D/frames/', f'beam_{snap}.png'))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=8)

    # Save as gif
    # save_dir = os.path.join(output_dir, 'beam.gif')
    anim.save(save_dir, writer=writergif)


def plot_image3D(z_net, z_gt, save_folder, var=5, step=-1, n=[]):
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    plt.axis('off')
    ax2 = fig.add_subplot(1, 3, 1, projection='3d')
    plt.axis('off')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plt.axis('off')
    ax1.set_title('Thermodynamics-informed GNN')
    ax2.set_title('Ground Truth')
    ax3.set_title('Thermodynamics-informed GNN error')
    ax1.view_init(elev=10., azim=90)
    ax2.view_init(elev=10., azim=90)
    ax3.view_init(elev=10., azim=90)
    # Oculta los bordes de los ejes
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax1.spines['left'].set_color((0.8, 0.8, 0.8))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax2.spines['left'].set_color((0.8, 0.8, 0.8))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_color((0.8, 0.8, 0.8))
    ax3.spines['left'].set_color((0.8, 0.8, 0.8))
    # Adjust ranges
    X, Y, Z = z_gt[:, :, 0].numpy(), z_gt[:, :, 1].numpy(), z_gt[:, :, 2].numpy()
    z_min, z_max = z_gt[:, :, var].min(), z_gt[:, :, var].max()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Initial snapshot
    q1_net, q2_net, q3_net = z_net[step, :, 0], z_net[step, :, 2], z_net[step, :, 1]
    q1_gt, q2_gt, q3_gt = z_gt[step, :, 0], z_gt[step, :, 2], z_gt[step, :, 1]
    # var_net = calculateBorders(z_net[-1, :, :3], h, r1, r2)
    # var_gt = calculateBorders(z_gt[-1, :, :3], h, r1, r2)
    var_net, var_gt = z_net[step, :, var], z_gt[step, :, var]
    var_error = var_gt - var_net
    var_error_min, var_error_max = var_error.min(), var_error.max()
    # Bounding box
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax1.plot([xb], [zb], [yb], 'w')
        ax2.plot([xb], [zb], [yb], 'w')
        ax3.plot([xb], [zb], [yb], 'w')
    # Scatter points
    # ax1.set(xlim=(-0.04, 0.04), ylim=(-0.04, 0.04), zlim=(-0.01, 0.08))
    glass_index = np.where(n == 0)
    fluid_index = np.where(n == 1)
    ax1.scatter(q1_net[glass_index], q2_net[glass_index], q3_net[glass_index], alpha=0.1)
    s1 = ax1.scatter(q1_net[fluid_index], q2_net[fluid_index], q3_net[fluid_index], alpha=0.8, c=var_gt[fluid_index],
                     vmax=z_max, vmin=z_min)
    ax2.scatter(q1_gt[glass_index], q2_gt[glass_index], q3_gt[glass_index], alpha=0.1)
    s2 = ax2.scatter(q1_gt[fluid_index], q2_gt[fluid_index], q3_gt[fluid_index], alpha=0.8, c=var_gt[fluid_index],
                     vmax=z_max, vmin=z_min)

    ax3.scatter(q1_net[glass_index], q2_net[glass_index], q3_net[glass_index], alpha=0.1)
    s3 = ax3.scatter(q1_net[fluid_index], q2_net[fluid_index], q3_net[fluid_index], alpha=0.8, c=var_error[fluid_index],
                     vmax=var_error_max, vmin=var_error_min)

    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    fig.colorbar(s2, ax=ax2, location='bottom', pad=0.08)
    fig.colorbar(s3, ax=ax3, location='bottom', pad=0.08)
    # Asegura una escala igual en ambos ejes
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')

    plt.savefig(os.path.join(save_folder, f"grafico{str(var)}.svg"), format="svg")


def generatePlot2(i_size, j_size, particleList, variableList, tensorData1, tensorData2, titleList=[], title=' '):
    fig = plt.figure(figsize=(15, 7))
    # plt.title(title)

    for i in range(i_size):
        for j in range(j_size):
            index = i * j_size + j
            ax1 = fig.add_subplot(i_size, j_size, index + 1)
            ax1.set_title(f'{title} particle: {str(j)}'), ax1.grid()
            ax1.plot(np.asarray(tensorData1[:, particleList[j], variableList[i]]), linestyle='dotted',
                     label=titleList[0])
            ax1.plot(np.asarray(tensorData2[:, particleList[j], variableList[i]]), label=titleList[1])
            ax1.legend()


def plotError(gt, z_net, L2_list, state_variables, dataset_dim, output_dir_exp):
    n_nodes = gt.shape[1]

    fig = plt.figure(figsize=(20, 20))

    for i, name in enumerate(L2_list):
        ax1 = fig.add_subplot(len(state_variables), 2, i * 2 + 1)
        ax1.set_title(name), ax1.grid()
        ax1.plot((gt[:, :, i]).sum((1)), linestyle='dotted', color='purple', label=f'{name} GT')
        ax1.plot((z_net.numpy()[:, :, i]).sum((1)), color='purple', label=f'{name} net')
        ax1.legend()

        ax2 = fig.add_subplot(len(state_variables), 2, i * 2 + 2)
        ax2.set_title('Error L2'), ax2.grid()
        ax2.plot(L2_list[name], color='purple', label=f'{name} GT')
    plt.savefig(os.path.join(output_dir_exp, 'L2_error.png'))

    if dataset_dim == 2:
        generatePlot2(2, 4, [0, int(n_nodes / 3) - 1, int(2 * n_nodes / 3) - 1, -1], [0, 1], gt, z_net.numpy(),
                      titleList=['gt', 'predicted'], title='Position')
        plt.savefig(os.path.join(output_dir_exp, 'position_error.png'))
        generatePlot2(2, 4, [0, int(n_nodes / 3) - 1, int(2 * n_nodes / 3) - 1, -1], [2, 3], gt, z_net.numpy(),
                      titleList=['gt', 'predicted'], title='Velocity')
        plt.savefig(os.path.join(output_dir_exp, 'velocity_error.png'))
        generatePlot2(3, 4, [0, int(n_nodes / 3) - 1, int(2 * n_nodes / 3) - 1, -1], [4, 5, 6, 7], gt, z_net.numpy(),
                      titleList=['gt', 'predicted'], title='SS')
        plt.savefig(os.path.join(output_dir_exp, 'ss_error.png'))


def generate_pointclud(z_net, n, name=''):
    # Crear la nube de puntos inicial
    pcd = o3d.geometry.PointCloud()
    xyz = np.asarray(z_net[0, :, 0:3].to(torch.float64))
    try:
        pcd.points = o3d.utility.Vector3dVector(xyz)
    except Exception as e:
        print(f"Error: {e}")

    data = z_net[0, :, -1]
    norm = Normalize(vmin=data.min(), vmax=data.max())
    cmap = plt.get_cmap('viridis')
    colors = cmap(norm(data))
    colors[n == 0, :] = np.array([0.8, 0.8, 0.8, 1])
    colores = o3d.utility.Vector3dVector(colors[:, :-1].astype(np.float64))
    pcd.colors = colores

    # Crear la ventana de visualización con dimensiones explícitas
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    # view_control = visualizer.get_view_control()
    visualizer.add_geometry(pcd)

    # Guardar la posición inicial de la cámara fuera del bucle
    # initial_view = view_control.convert_to_pinhole_camera_parameters()

    for i in range(1, z_net.shape[0]):
        # No restaurar la vista en cada iteración, innecesario
        # view_control.convert_from_pinhole_camera_parameters(initial_view)
        # view_control.set_zoom(0.8)

        # Actualizar la nube de puntos
        z_net_i = np.asarray(z_net[i, :, 0:3].to(torch.float64))
        pcd.points = o3d.utility.Vector3dVector(z_net_i)

        data = z_net[i, :, -1]
        norm = Normalize(vmin=data.min(), vmax=data.max())
        colors = cmap(norm(data))
        colors[n == 0, :] = np.array([0.8, 0.8, 0.8, 1])
        colores = o3d.utility.Vector3dVector(colors[:, :-1].astype(np.float64))
        pcd.colors = colores

        # Actualizar la visualización y guardar el frame
        visualizer.update_geometry(pcd)
        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.capture_screen_image(f'images/{name}_frame_{i}.png', do_render=True)

        # Pequeña pausa para permitir que la visualización se actualice correctamente
        time.sleep(0.05)

    # Cerrar la ventana al finalizar
    visualizer.destroy_window()



def video_plot_3D(z_net, z_gt, save_dir, n=[]):
    generate_pointclud(z_gt, n, name='gt')
    generate_pointclud(z_net, n, name='net')
    image_lst = []
    for i in range(z_net.shape[0] - 1):
        try:
            frame_gt = cv2.cvtColor(cv2.imread(f'images/gt_frame_{i + 1}.png'), cv2.COLOR_BGR2RGB)
            frame_net = cv2.cvtColor(cv2.imread(f'images/net_frame_{i + 1}.png'), cv2.COLOR_BGR2RGB)

            frame_gt = cv2.resize(frame_gt[100:-100, 450:-450, :], None, fx=0.8, fy=0.8)
            frame_net = cv2.resize(frame_net[100:-100, 450:-450, :], None, fx=0.8, fy=0.8)

            # Asegúrate de que ambas imágenes tengan la misma altura
            altura = min(frame_gt.shape[0], frame_net.shape[0])

            # Concatena las imágenes horizontalmente
            imagen_concatenada = np.concatenate((frame_gt[:altura, :], frame_net[:altura, :]), axis=1)
            image_lst.append(imagen_concatenada)
        except:
            print()

    imageio.mimsave(save_dir, image_lst, fps=10, loop=1)



def plot_3D_mp(z_net, plot_info, frame):
    T = len(plot_info)
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax1.set_title('Thermodynamics-informed GNN'), ax1.grid()
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
    ax1.view_init(elev=0., azim=90)
    # Adjust ranges
    X, Y, Z = z_net[:, :, 0].numpy(), z_net[:, :, 1].numpy(), z_net[:, :, 2].numpy()
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Initial snapshot
    q1_net, q2_net, q3_net = z_net[frame, :, 0], z_net[frame, :, 2], z_net[frame, :, 1]
    # Bounding box
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax1.plot([xb], [zb], [yb], 'w')
    # Scatter points
    s1 = ax1.scatter(q1_net, q2_net, q3_net, c=plot_info[0].cpu() / plot_info[0].cpu())
    # Colorbar
    fig.colorbar(s1, ax=ax1, location='bottom', pad=0.08)
    i = 1
    # Animation
    def animate(snap):
        ax1.cla()
        ax1.set_title(f'Thermodynamics-informed GNN, f={str(snap)}')
        ax1.grid()
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')

        # Bounding box
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [zb], [yb], 'w')

        # Scatter points
        q1_net, q2_net, q3_net = z_net[frame, :, 0], z_net[frame, :, 2], z_net[frame, :, 1]
        ax1.scatter(q1_net, q2_net, q3_net, c=plot_info[snap].cpu() / plot_info[0].cpu())

        # Guardar la figura para cada frame
        fig.savefig(os.path.join(
            r'/home/atierz/Documentos/experiments/Foam_visco/3D/frames/',
            f'2beam_{frame}_{snap}.png'
        ))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=8)
    # Save as gif
    # save_dir = os.path.join(output_dir, 'beam.gif')
    save_dir = f'{frame}_messagePassing.gif'
    anim.save(save_dir, writer=writergif)