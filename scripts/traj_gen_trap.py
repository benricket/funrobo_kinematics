
import numpy as np

from funrobo_kinematics.core.trajectory_generator import MultiAxisTrajectoryGenerator, MultiSegmentTrajectoryGenerator

class Trapezoidal():
    """
    Trapezoidal trajectory
    """

    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof

    
    def solve(self, q0, qf, qd0, qdf, T):
        """
        Compute cubic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)

        self.q0 = q0
        self.qf = qf

        # compute peak velocity
        self.V = (1.5*(qf - q0))/T
        # blend time
        self.tb = (q0 - qf + self.V*T)/self.V
        # peak acceleration
        self.alpha = self.V / self.tb

    
    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))
        for i in range(self.ndof): # iterate through all DOFs
            #j = np.arange(nsteps)
            #t = (tf - t0)*(j/nsteps) # time at current step
            for j in range(nsteps):
                t = (tf - t0)*(j/nsteps)
                if t < self.tb[i]:
                    q = self.q0[i] + self.alpha[i] * t**2 * 0.5
                    qd = self.alpha[i] * t
                    qdd = self.alpha[i]
                elif t < tf - self.tb[i]:
                    q = 0.5*(self.qf[i] + self.q0[i] - self.V[i]*tf) + self.V[i]*t
                    qd = self.V[i]
                    qdd = 0
                else:
                    q = self.qf[i] - 0.5 * self.alpha[i] * tf**2 + self.alpha[i] * tf * t - 0.5 * self.alpha[i] * t**2
                    qd = self.alpha[i] * (tf - t)
                    qdd = -1 * self.alpha[i]
                
                X[i, 0, j] = q
                X[i, 1, j] = qd
                X[i, 2, j] = qdd

        return t, X



class CubicPolynomial():
    """
    Cubic interpolation with position and velocity boundary constraints.
    """

    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof

    
    def solve(self, q0, qf, qd0, qdf, T):
        """
        Compute cubic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        qd0 = np.zeros_like(q0) if qd0 is None else np.asarray(qd0, dtype=float)
        qdf = np.zeros_like(q0) if qdf is None else np.asarray(qdf, dtype=float)
        
        A = np.array(
                [[1, t0, t0**2, t0**3],
                 [0, 1, 2*t0, 3*t0**2],
                 [1, tf, tf**2, tf**3],
                 [0, 1, 2*tf, 3*tf**2]
                ])

        b = np.vstack([
            q0,
            qd0,
            qf,
            qdf
        ])
        self.coeff = np.linalg.solve(A, b)
        

    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))
        for i in range(self.ndof): # iterate through all DOFs
            c = self.coeff[:, i]

            q = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3
            qd = c[1] + 2 * c[2] * t + 3 * c[3] * t**2
            qdd = 2 * c[2] + 6 * c[3] * t

            X[i, 0, :] = q      # position
            X[i, 1, :] = qd     # velocity
            X[i, 2, :] = qdd    # acceleration

        return t, X




def main():
    ndof = 2
    method = Trapezoidal(ndof=ndof)
    mode = "joint"

    # --------------------------------------------------------
    # Point-to-point multi-axis trajectory generator
    # --------------------------------------------------------

    # traj = MultiAxisTrajectoryGenerator(method=method,
    #                                     mode=mode,
    #                                     ndof=ndof)
    
    # traj.solve(q0=-30, qf=60, T=1)
    # traj.generate(nsteps=20)

    # --------------------------------------------------------
    # Via point multi-axis trajectory generator
    # --------------------------------------------------------

    traj = MultiSegmentTrajectoryGenerator(method=method,
                                           mode=mode,
                                           ndof=ndof,
                                            )
    via_points = [[-30, 30], [0, 45], [30, 15], [50, -30]]

    traj.solve(via_points, T=2)
    traj.generate(nsteps_per_segment=20)
    
    
    # plotter
    traj.plot()

if __name__ == "__main__":
    main()