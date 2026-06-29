import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

X = np.array([[0, 1],
              [1, 0]], dtype=DTYPE)

Y = np.array([[0, -1j],
              [1j, 0]], dtype=DTYPE)

Z = np.array([[1, 0],
              [0, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""

    b = Bloch()

    # Global phase
    b.alpha = np.angle(np.linalg.det(g)) / 2.0

    # Remove global phase
    u = np.exp(-1j * b.alpha) * g

    # Rotation angle
    cos_half = np.real(np.trace(u)) / 2.0
    cos_half = np.clip(cos_half, -1.0, 1.0)
    b.theta = 2.0 * np.arccos(cos_half)

    sin_half = np.sin(b.theta / 2.0)

    # Identity rotation: axis is arbitrary
    if np.isclose(sin_half, 0.0):
        b.n = np.array([1.0, 0.0, 0.0])
        return b

    # Rotation axis using the Pauli matrices
    nx = np.real((-1j / 2.0) * np.trace(X @ u)) / sin_half
    ny = np.real((-1j / 2.0) * np.trace(Y @ u)) / sin_half
    nz = np.real((-1j / 2.0) * np.trace(Z @ u)) / sin_half

    b.n = np.array([nx, ny, nz])
    b.n /= np.linalg.norm(b.n)

    return b


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
# TODO: fill in the two orthogonal rotation axes (each a length-3
# unit vector [x, y, z])
cot = 1.0 / np.tan(np.pi / 8.0)

# (z - x) / sqrt(2)
zx = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)

# y-axis
y = np.array([0.0, 1.0, 0.0])

n1 = np.sqrt(2.0) * cot * zx + y
n1 /= np.linalg.norm(n1)

n2 = np.sqrt(2.0) * cot * y - zx
n2 /= np.linalg.norm(n2)

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)


def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """
    # TODO(student): implement using the steps above.
    raise NotImplementedError("n1n2n1_angles is not implemented yet")


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
        (k * LAMBDA_PI) mod 2*pi  ~=  angle   (within `tolerance`)
    Since LAMBDA_PI / (2 pi) is irrational, such a k always exists; search
    k = 1, 2, 3, ... and return the first one whose wrapped multiple lands within
    `tolerance` of `angle` (compare both as angles in [0, 2 pi)).

    Hint:
      * wrap an angle into [0, 2 pi)
      * the angular distance between two wrapped angles a, b is
        min(|a - b|, TWO_PI - |a - b|) (so 0.01 and 2*pi - 0.01 count as close).
    """
    # TODO(student): implement using the hint above.
    raise NotImplementedError("approx_angle_with_tolerance is not implemented yet")


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

      1. Get the Bloch form of u (to_bloch), then factor its rotation into the
         three frame angles with n1n2n1_angles:
             alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
         alpha and gamma are rotations about a1 (realized by powers of M1);
         beta is a rotation about a2 (realized by powers of M2).

      2. Convert each angle to an integer power with approx_angle_with_tolerance:
             k = approx_angle_with_tolerance(alpha, tolerance)   # power of M1
             l = approx_angle_with_tolerance(beta,  tolerance)   # power of M2
             m = approx_angle_with_tolerance(gamma, tolerance)   # power of M1
         (Mind the relationship between a target rotation angle and the base
         angle each application of M1/M2 adds.)

      3. Return (k, l, m).
    """
    # TODO(student): implement using the steps above.
    def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
        """Find the first integer k such that
       (k * LAMBDA_PI) mod 2π is within tolerance of angle.
       """

        angle = angle % TWO_PI

        k = 1

        while True:
            candidate = (k * LAMBDA_PI) % TWO_PI

            diff = abs(candidate - angle)
            diff = min(diff, TWO_PI - diff)

            if diff <= tolerance:
                return k

            k += 1