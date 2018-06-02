import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D, proj3d, art3d

# Function to set orthogonal projection:
# https://codeyarns.com/2014/11/13/how-to-get-orthographic-projection-in-3d-plot-of-matplotlib/ 
def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    # -0.0001 added for numerical stability as suggested in:
    # http://stackoverflow.com/questions/23840756
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,a,b],
                     [0,0,-0.0001,zback]]) 

def plot_sproj3d(n=[1,0,1], offset=0., **kwargs):
    n = np.asarray(n)/np.linalg.norm(n) # normalize n

    r = kwargs.pop('r', 1.) # radius of the sphere
    p = kwargs.pop('p', r) # z coordinate of the South pole
    bound = kwargs.pop('bound', 1.1)
    show = kwargs.pop('show', True)
    only_positive = kwargs.pop('only_positive', False)

    # Coordinates of the center of the circle expressed as a translation along the direction n (i.e., C = O + n*T)
    C = np.zeros(3) + n*offset
    C = C.reshape(3,1)
    r_prime = (r**2. - offset**2.)**.5 # Radius of the circle

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')


    if (n[0] == 0.) & (n[1] == 0.):
        a = np.asarray([1,0,0]).reshape(3,1)
        b = np.asarray([0,1,0]).reshape(3,1)
    else:
        # parametrization of the circle
        # http://demonstrations.wolfram.com/ParametricEquationOfACircleIn3D/
        a = np.asarray([-n[1], n[0], 0]).reshape(3,1)
        a = a/np.linalg.norm(a)
        b = np.asarray([-n[0]*n[2], -n[1]*n[2], n[0]**2+n[1]**2]).reshape(3,1)
        b = b/np.linalg.norm(b)

    t1, t2 = 0., 2*np.pi
    if only_positive == True:
        s = float(C[2]/(b[2]*r_prime))
        if np.abs(s) < 1.:
            t1, t2 = -np.arcsin(s), np.arcsin(s) + np.pi
            if np.sin((t1+t2)/2.) < 0.:
                t1 += 2*np.pi

    t = np.linspace(t1, t2, 50)
    P = r_prime*np.cos(t)*a + r_prime*np.sin(t)*b + C
    ax.plot(P[0], P[1], P[2]) # plot circle

    Q = np.ndarray(P.shape)
    Q[0], Q[1], Q[2] = p*P[0]/(p+P[2]), p*P[1]/(p+P[2]), 0. # stereographic projection of the circle (trace)
    sel = (np.abs(Q[0]) <= bound) & (np.abs(Q[1]) <= bound)
    Q = Q[:,sel]
    ax.plot(Q[0], Q[1], 'g.') # plot trace of the circle
    ax.quiver(0, 0, 0, n[0], n[1], n[2], pivot='tail', length=r) # draw an arrow corresponding to the vector n
    xp, yp = p*n[0]/(p+n[2]), p*n[1]/(p+n[2]) # stereographic projection of the pole
    ax.plot([xp], [yp], [0], 'go')

    m = Q.shape[1]
    x, y, z = np.ndarray(3*m), np.ndarray(3*m), np.ndarray(3*m)
    x[::3], x[1::3], x[2::3] = 0, Q[0], P[0][sel]
    y[::3], y[1::3], y[2::3] = 0, Q[1], P[1][sel]
    z[::3], z[1::3], z[2::3] = -p, 0., P[2][sel]
    # draw [red] lines connecting the South pole to the circle and its stereographic projection
    ax.plot(x, y, z, color='r', linewidth=.25)

    # add patch corresponding to the plane z = 0
    plane = plt.Circle((0,0), 1., color='g', alpha=.1)
    ax.add_patch(plane)
    art3d.pathpatch_2d_to_3d(plane, z=0, zdir='z')

    # draw a [red] line connecting the South pole to the pole
    ax.plot([0,xp,n[0]], [0,yp,n[1]], [-p,0,n[2]], color='r', linewidth=.25)

    # # draw vertexes (vertices) of a cube
    # x, y, z = np.mgrid[-1:2:2, -1:2:2, -1:2:2]
    # ax.plot(bound*x.ravel(), bound*y.ravel(), bound*z.ravel(), lw=0)#'k.')
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_zlim(-bound, bound)

    # draw sphere
    # http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:11j]
    x=r*np.cos(u)*np.sin(v)
    y=r*np.sin(u)*np.sin(v)
    z=r*np.cos(v)
    ax.plot_wireframe(x, y, z, color='k', linewidth=.1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    proj3d.persp_transformation = orthogonal_proj # orthogonal projection
    
    if show == True:
        plt.show()

    return ax

# ax = plot_sproj3d(n=[1,2,3], offset=0.5, show=True, only_positive=True)
# ax = plot_sproj3d(n=[1.,1.,1.], offset=.95, p=0.)
