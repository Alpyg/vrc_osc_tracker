from math import sqrt

from mediapipe.tasks.python.components.containers.landmark import Landmark


class Tracker:
    def __init__(
        self,
        namespace: str,
        position: tuple[float, float, float] = (0, 0, 0),
        rotation: tuple[float, float, float] = (0, 0, 0),
        offset: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
    ):
        self.namespace = namespace
        self.position = position
        self.rotation = rotation
        self.offset = offset
        self.scale = scale

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    @property
    def rot_x(self):
        return self.rotation[0]

    @property
    def rot_y(self):
        return self.rotation[1]

    @property
    def rot_z(self):
        return self.rotation[2]

    def update_position(
        self,
        landmark: Landmark,
        base_alpha=0.1,
        distance_threshold=0.1,
    ):
        """Lerp the position towards the new position with adaptive alpha if the landmark visibility is >= 0.5.

        Args:
            landmark (Landmark): The target position to lerp to.
            base_alpha (float): The base alpha value for lerping.
            distance_threshold (float): The distance threshold to adjust alpha.
        """
        if landmark.visibility < 0.5 or landmark.presence < 0.5:
            return

        alpha = self.adaptive_alpha(landmark, base_alpha, distance_threshold)
        self.position = self.lerp(landmark, alpha)

    def lerp(
        self,
        landmark: Landmark,
        alpha: float,
    ) -> tuple[float, float, float]:
        """Linearly interpolate between the current position and a new Landmark.

        Args:
            landmark (Landmark): The new landmark.
            alpha (float): The interpolation factor (0 <= alpha <= 1).

        Returns:
            tuple: The interpolated point (x, y, z).
        """
        return (
            self.x + (landmark.x - self.x) * alpha,
            self.y + (landmark.y - self.y) * alpha,
            self.z + (landmark.z - self.z) * alpha,
        )

    def adaptive_alpha(
        self,
        landmark: Landmark,
        base_alpha=0.1,
        distance_threshold=0.1,
    ) -> float:
        """Calculate an adaptive alpha value based on distance.

        Args:
            landmark (Landmark): The new landmark to compare.
            base_alpha (float): The base alpha value for lerping.
            distance_threshold (float): The distance threshold to adjust alpha.

        Returns:
            float: The adaptive alpha value for lerping.
        """
        distance = sqrt(
            (self.x - landmark.x) ** 2
            + (self.y - landmark.y) ** 2
            + (self.z - landmark.z) ** 2
        )
        alpha = min(base_alpha + (distance / distance_threshold) * base_alpha, 1)
        return alpha

    def send_osc(self, osc_client):
        """Send the current position to an OSC address, applying the offset."""
        if self.namespace == "":
            return

        osc_client.send_message(
            f"/tracking/trackers/{self.namespace}/position",
            [
                -(self.x + self.offset[0]) * self.scale[0],
                (self.y + self.offset[1]) * self.scale[1],
                -(self.z + self.offset[2]) * self.scale[2],
            ],
        )

    def __repr__(self):
        return (
            f"Tracker(position=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}), "
            f"rotation={self.rotation}, "
            f"offset={self.offset}, "
            f"osc_address='/tracking/trackers/{self.namespace}', "
            f"scale={self.scale})"
        )
