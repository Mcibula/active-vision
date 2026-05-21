import numpy as np
from scipy.fftpack import dct
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

from structures import Trajectory


class TrajVectorizer:
    def __init__(
            self,
            target_len: int = 60,
            num_pcoeffs: int = 15,
            num_rcoeffs: int = 15
    ) -> None:
        self.target_len: int = target_len
        self.n_pos: int = num_pcoeffs
        self.n_rot: int = num_rcoeffs

    def __call__(self, traj: Trajectory) -> np.ndarray:
        positions = traj.positions
        orientations = traj.orientations

        if positions is None or orientations is None or len(positions) < 5:
            return np.zeros(shape=(self.n_pos * 3 + self.n_rot * 3))

        d_pos = np.diff(positions, axis=0)

        pos_resampled = self._resample(d_pos)
        pos_spectral: np.ndarray = dct(
            x=pos_resampled, axis=0,
            type=2, norm='ortho'
        )
        pos_coeffs: np.ndarray = pos_spectral[:self.n_pos].flatten()

        rots = Rotation.from_euler('xyz', orientations)
        r_cur = rots[:-1]
        r_next = rots[1:]
        r_diff = r_next * r_cur.inv()
        d_rot = r_diff.as_rotvec()

        rot_resampled = self._resample(d_rot)
        rot_spectral: np.ndarray = dct(
            x=rot_resampled, axis=0,
            type=2, norm='ortho'
        )
        rot_coeffs: np.ndarray = rot_spectral[:self.n_rot].flatten()

        return np.concatenate((pos_coeffs, rot_coeffs))

    def _resample(self, data: np.ndarray) -> np.ndarray:
        n = len(data)

        if n == self.target_len:
            return data

        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, self.target_len)

        f = interp1d(
            x=x_old, y=data, axis=0,
            kind='linear',
            fill_value='extrapolate'
        )

        return f(x_new)
