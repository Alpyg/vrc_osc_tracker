import io
import matplotlib.pyplot as plt
from .tracking import world_landmarks


def plot_landmarks():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if world_landmarks is not None:
        for idx in range(len(world_landmarks)):
            x_coords = []
            y_coords = []
            z_coords = []

            for landmark in world_landmarks[idx]:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
                z_coords.append(landmark.z)

            ax.scatter(x_coords, y_coords, z_coords, c="r", marker="o")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Pose Landmarks in 3D")

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
