import numpy as np
from scipy.spatial.transform import Rotation as R


class BaseTransform:
    def __init__(self, total_steps=-1):
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1


class PointCloudNormalize(BaseTransform):
    def __init__(self, mode='shape_unit'):
        super().__init__()
        self.mode = mode

    def __call__(self, x):
        if self.mode == 'box':
            shift = (np.max(x, 0) + np.min(x, 0)) / 2
            scale = np.max(np.linalg.norm(x - shift, axis=1))
        elif self.mode == 'shape_unit':
            shift = x.mean(axis=0)
            scale = x.std()
        elif self.mode == 'mean':
            shift = x.mean(axis=0)
            scale = np.max(np.linalg.norm(x - shift, axis=1))

        return (x - shift) / scale


class RandomRotation(BaseTransform):
    def __init__(self, low, high, axis):
        super().__init__()
        self.low = low
        self.high = high
        self.axis = axis

    def __call__(self, x):
        rot_matrix = self.get_rotation_matrix()
        return x @ rot_matrix

    def get_rotation_matrix(self):
        rotation_matrix = R.from_euler(
            self.axis,
            np.random.randint(self.low, self.high + 1, len(self.axis)),
            degrees=True
        ).as_matrix()

        return rotation_matrix


class RandomJitter(BaseTransform):
    def __init__(self, std, clip_bound):
        super().__init__()
        self.std = std
        self.clip_bound = clip_bound

    def __call__(self, x):
        noise = self.std * np.random.randn(*x.shape)
        noise = np.clip(noise, -self.clip_bound, self.clip_bound)

        return x + noise


class RandomScale(BaseTransform):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def __call__(self, x):
        scale_vector = np.random.uniform(self.low, self.high, 3)
        return x * scale_vector


class MeshNetRandomRotation(RandomRotation):
    def __init__(self, low, high, axis):
        super().__init__(low, high, axis)

    def __call__(self, x):
        return super().__call__(x.reshape(-1, 5, 3)).reshape(-1, 15)


class RandomFlip(BaseTransform):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            x[:, 0] *= -1

        return x


class MeshNetRandomJitter(RandomJitter):
    def __init__(self, std, clip_bound):
        super().__init__(std, clip_bound)

    def __call__(self, x):
        jittered = super().__call__(x[:, :12])
        return np.concatenate((jittered, x[:, 12:]), 1)


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)

        return x

    def step(self):
        for t in self.transforms:
            t.step()
