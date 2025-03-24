# --------------------------------------------------------------
# Tree DBH Detection with Forks using Python & Open3D
# Author: AODM
# Date: 22 March 2025
# Description: Detect tree DBH and forked trees from Point Cloud,
#              fit ellipse, calculate DBH, export PLY and CSV
# --------------------------------------------------------------

import numpy as np
import open3d as o3d
import laspy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from skimage.measure import EllipseModel
from sklearn.decomposition import PCA

def read_las(las_file):
    las = laspy.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).T
    classifications = las.classification
    return points, classifications

def normalize_height(all_pts, ground_pts):
    ground_z = griddata(ground_pts[:, :2], ground_pts[:, 2], all_pts[:, :2],
                        method='linear', fill_value=np.nan)
    return all_pts[:, 2] - ground_z

def pca_ratio(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]

def fit_ellipse_2d(x, y):
    data = np.column_stack([x, y])
    ellipse = EllipseModel()
    if ellipse.estimate(data):
        return ellipse.params  # xc, yc, a, b, theta
    else:
        return None

def fork_detection(cluster_pts):
    fork_pcd = o3d.geometry.PointCloud()
    fork_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
    fork_labels = np.array(fork_pcd.cluster_dbscan(eps=0.2, min_points=30))
    forks = []
    for fork_id in np.unique(fork_labels):
        if fork_id == -1:
            continue
        fork_pts = cluster_pts[fork_labels == fork_id]
        if len(fork_pts) < 100:
            continue
        forks.append(fork_pts)
    return forks

def process_clusters(las_file, z_min=1.2, z_max=1.4):
    points, classifications = read_las(las_file)
    print(f"Total Points: {len(points)}")
    ground_pts = points[classifications == 2]
    print(f"Ground Points: {len(ground_pts)}")

    norm_heights = normalize_height(points, ground_pts)
    dbh_mask = (norm_heights >= z_min) & (norm_heights <= z_max)
    dbh_points = points[dbh_mask]
    print(f"DBH Slice Points: {len(dbh_points)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dbh_points)
    labels = np.array(pcd.cluster_dbscan(eps=0.08, min_points=50))
    unique_clusters = np.unique(labels[labels >= 0])
    print(f"Found {len(unique_clusters)} clusters")

    results = []

    for cluster_id in unique_clusters:
        cluster_pts = dbh_points[labels == cluster_id]
        if len(cluster_pts) < 200:
            continue

        min_x, min_y = np.min(cluster_pts[:, 0]), np.min(cluster_pts[:, 1])
        max_x, max_y = np.max(cluster_pts[:, 0]), np.max(cluster_pts[:, 1])
        bbox_mask = (points[:, 0] >= min_x) & (points[:, 0] <= max_x) & \
                    (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
        cluster_full = points[bbox_mask]
        cluster_heights = normalize_height(cluster_full, ground_pts)

        if not np.any(cluster_heights < 0.1):
            print(f"Cluster {cluster_id} rejected - No ground connection")
            continue

        if pca_ratio(cluster_pts) > 5:
            print(f"Cluster {cluster_id} rejected - Elongated")
            continue

        print(f"Cluster {cluster_id} large, performing fork detection")
        forks = fork_detection(cluster_pts)
        print(f"Detected {len(forks)} forks")

        for fork_idx, fork_pts in enumerate(forks):
            x, y = fork_pts[:, 0], fork_pts[:, 1]
            ellipse_params = fit_ellipse_2d(x, y)
            if ellipse_params is None:
                print(f"Fork {fork_idx} ellipse fitting failed")
                continue

            xc, yc, a, b, theta = ellipse_params
            dbh = (a + b)  # DBH in meters
            print(f"Fork {fork_idx}: Center ({xc:.2f}, {yc:.2f}), a={a:.2f}, b={b:.2f}, DBH={dbh:.3f} m")

            results.append({
                'Tree_ID': cluster_id,
                'Fork_ID': fork_idx,
                'Center_X': xc,
                'Center_Y': yc,
                'Ellipse_a': a,
                'Ellipse_b': b,
                'Ellipse_angle': theta,
                'DBH_m': dbh,
                'Point_Count': len(fork_pts)
            })

            fork_pcd = o3d.geometry.PointCloud()
            fork_pcd.points = o3d.utility.Vector3dVector(fork_pts)
            o3d.io.write_point_cloud(f"tree{cluster_id}_fork{fork_idx}.ply", fork_pcd)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(x - xc, y - yc, s=2, color='blue')
            t = np.linspace(0, 2 * np.pi, 300)
            ellipse_x = a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
            ellipse_y = a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
            ax.plot(ellipse_x, ellipse_y, 'r-', linewidth=1)

            ax.plot([-a*np.cos(theta), a*np.cos(theta)], 
                    [-a*np.sin(theta), a*np.sin(theta)], 'r-', linewidth=0.5)
            ax.plot([-b*np.sin(theta), b*np.sin(theta)], 
                    [b*np.cos(theta), -b*np.cos(theta)], 'r-', linewidth=0.5)

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.savefig(f"tree{cluster_id}_fork{fork_idx}_ellipse.png", dpi=300)
            plt.close()

    df = pd.DataFrame(results)
    df.to_csv("dbh_results_fork_ellipse.csv", index=False)
    print("\nExported DBH results with forks and DB
