"""Verify the explicit Cartesian real spherical harmonics used in nr_pic_plummer.cpp."""
import numpy as np
from scipy.special import sph_harm_y

S4PI = 1.0/np.sqrt(4*np.pi)

def Ycart(x, y, z):
    """Orthonormal REAL spherical harmonics l=0..4 on the unit sphere, in the order
    (l, m) with m = -l..l, matching the C++ table."""
    P = np.pi
    x2, y2, z2 = x*x, y*y, z*z
    out = []
    out.append(np.full_like(x, 0.5/np.sqrt(P)))                       # 0,0
    c = np.sqrt(3/(4*P))
    out += [c*y, c*z, c*x]                                            # 1,-1 1,0 1,1
    c15 = 0.5*np.sqrt(15/P)
    out += [c15*x*y, c15*y*z, 0.25*np.sqrt(5/P)*(3*z2-1.0),
            c15*x*z, 0.25*np.sqrt(15/P)*(x2-y2)]                      # l=2
    a = 0.25*np.sqrt(35/(2*P)); b = 0.5*np.sqrt(105/P); d = 0.25*np.sqrt(21/(2*P))
    out += [a*y*(3*x2-y2), b*x*y*z, d*y*(5*z2-1.0),
            0.25*np.sqrt(7/P)*z*(5*z2-3.0), d*x*(5*z2-1.0),
            0.25*np.sqrt(105/P)*z*(x2-y2), a*x*(x2-3*y2)]              # l=3
    e = 0.75*np.sqrt(35/P); f = 0.75*np.sqrt(35/(2*P)); g = 0.75*np.sqrt(5/P)
    h = 0.75*np.sqrt(5/(2*P))
    out += [e*x*y*(x2-y2), f*y*z*(3*x2-y2), g*x*y*(7*z2-1.0),
            h*y*z*(7*z2-3.0), (3.0/16.0)*np.sqrt(1/P)*(35*z2*z2-30*z2+3.0),
            h*x*z*(7*z2-3.0), 0.375*np.sqrt(5/P)*(x2-y2)*(7*z2-1.0),
            f*x*z*(x2-3*y2), (3.0/16.0)*np.sqrt(35/P)*(x2*(x2-3*y2)-y2*(3*x2-y2))]
    return np.array(out)

def Yref(x, y, z):
    """scipy reference: real harmonics built from the complex ones."""
    th = np.arccos(np.clip(z, -1, 1)); ph = np.arctan2(y, x)
    out = []
    for l in range(5):
        for m in range(-l, l+1):
            if m == 0:
                v = sph_harm_y(l, 0, th, ph).real
            elif m > 0:
                v = np.sqrt(2)*(-1)**m*sph_harm_y(l, m, th, ph).real
            else:
                v = np.sqrt(2)*(-1)**m*sph_harm_y(l, -m, th, ph).imag
            out.append(v)
    return np.array(out)

rng = np.random.default_rng(7)
v = rng.normal(size=(3, 20000)); v /= np.linalg.norm(v, axis=0)
A = Ycart(*v); B = Yref(*v)
print("max |Ycart - Yref| per (l,m):")
i = 0
bad = 0
for l in range(5):
    for m in range(-l, l+1):
        d = np.max(np.abs(A[i]-B[i]))
        s = "OK" if d < 1e-12 else "*** MISMATCH ***"
        if d >= 1e-12: bad += 1
        print(f"  l={l} m={m:+d}: {d:.3e} {s}")
        i += 1
print("orthonormality check  4pi<Y_a Y_b> - delta_ab:",
      float(np.max(np.abs(4*np.pi*(A@A.T)/A.shape[1] - np.eye(25)))))
print("bad =", bad)
