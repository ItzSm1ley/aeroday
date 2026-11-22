import numpy as np

# ---------------- CAMERA SETTINGS -------------------------
# World axes: X = east, Y = north, Z = up (meters).
# yaw   = direction the camera faces, in degrees. 0 = +X, 90 = +Y.
# pitch = tilt in degrees. 0 = level, negative = looking down.
# hfov  = horizontal field of view of the lens, in degrees.
CAM1 = dict(pos=(0, 0, 10),   yaw=45,  pitch=-15, hfov=78, width=1920, height=1080)
CAM2 = dict(pos=(50, 40, 10), yaw=225, pitch=-15, hfov=78, width=1920, height=1080)
# ---------------------------------------------------------------


def pixel_to_ray(cam, px, py):
    """Turn a pixel (px, py) into a 3D direction in world coordinates."""
    yaw, pitch = np.radians(cam["yaw"]), np.radians(cam["pitch"])
    forward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    up = np.cross(right, forward)

    focal = (cam["width"] / 2) / np.tan(np.radians(cam["hfov"]) / 2)  # focal length in pixels
    x = (px - cam["width"] / 2) / focal
    y = (py - cam["height"] / 2) / focal  # image y goes DOWN

    d = forward + x * right - y * up
    return d / np.linalg.norm(d)


def triangulate(pixel1, pixel2, cam1=CAM1, cam2=CAM2):
    """
    pixel1 = (x, y) pixel of the object in Camera 1
    pixel2 = (x, y) pixel of the object in Camera 2
    Returns ((X, Y, Z), gap) in meters.
    gap = how far apart the two sight lines pass (small = good match).
    """
    p1, p2 = np.array(cam1["pos"], float), np.array(cam2["pos"], float)
    d1, d2 = pixel_to_ray(cam1, *pixel1), pixel_to_ray(cam2, *pixel2)

    # Find the closest points on the two sight lines: p1 + t*d1 and p2 + s*d2
    w = p1 - p2
    b = d1 @ d2
    d, e = d1 @ w, d2 @ w
    denom = 1 - b * b
    if denom < 1e-9:
        raise ValueError("The two sight lines are parallel - can't intersect.")
    t = (b * e - d) / denom
    s = (e - b * d) / denom
    if t < 0 or s < 0:
        raise ValueError("Intersection is behind a camera - check yaw/pitch settings.")

    point1, point2 = p1 + t * d1, p2 + s * d2
    midpoint = (point1 + point2) / 2           # best guess of the real point
    gap = np.linalg.norm(point1 - point2)
    return tuple(round(float(v), 2) for v in midpoint), round(float(gap), 2)


if __name__ == "__main__":
    xyz, gap = triangulate(pixel1=(1521, 633), pixel2=(732, 556))
    print(f"Object at X={xyz[0]} m, Y={xyz[1]} m, Z={xyz[2]} m  (gap {gap} m)")