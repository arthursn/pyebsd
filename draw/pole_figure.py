import numpy as np

def uvw_label(uvw, s='\gamma'):
    label = r'$['
    for index in uvw:
        if index < 0:
            label += '\\bar{%d}' % np.abs(index)
        else:
            label += '%d' % index
    label += ']_' + s + '$'
    return label

def draw_circle_frame(ax, **kwargs):
    kw = dict(c='k')
    kw.update(kwargs)
    t = np.linspace(0, 2*np.pi, 360)
    ax.plot(np.cos(t), np.sin(t), **kw)
    ax.plot([-1,1], [0,0], **kw)
    ax.plot([0,0], [-1,1], **kw)

def draw_projection(ax, d, **kwargs):
    kw = dict(marker='o', markeredgewidth=0.0)
    kw.update(kwargs)

    d = np.asarray(d)/np.linalg.norm(d) # normalize d
    c0, c1 = d[0]/(1.+d[2]), d[1]/(1.+d[2])
    ax.plot(c0, c1, **kw)

    return ax

def draw_circle(ax, n, ang=90, r=1., p=1., **kwargs):
    """
    r : radius of the sphere
    p : coordinates of the south pole
    """
    n = np.asarray(n)/np.linalg.norm(n) # normalize n
    ang = np.radians(ang)
    
    C = np.zeros(3) + n*np.cos(ang)
    C = C.reshape(3,1)
    r_prime = r*np.sin(ang) # Radius of the circle

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
    if b[2]*r_prime != 0.:
        s = float(C[2]/(b[2]*r_prime))
        if np.abs(s) < 1.:
            t1, t2 = -np.arcsin(s), np.arcsin(s) + np.pi
            if np.sin((t1+t2)/2.) < 0.:
                t1 += 2*np.pi

    t = np.linspace(t1, t2, 50)
    P = r_prime*np.cos(t)*a + r_prime*np.sin(t)*b + C
    xp, yp = p*P[0]/(p+P[2]), p*P[1]/(p+P[2]) # stereographic projection of the circle (trace)
    ax.plot(xp, yp, **kwargs)

    return ax    

def draw_trace(ax, n=[1,0,1], offset=0., r=1., p=1., **kwargs):
    n = np.asarray(n)/np.linalg.norm(n) # normalize n

    # Coordinates of the center of the circle expressed as a translation along the direction n (i.e., C = O + n*T)
    C = np.zeros(3) + n*offset
    C = C.reshape(3,1)
    r_prime = (r**2. - offset**2.)**.5 # Radius of the circle

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
    if b[2]*r_prime != 0.:
        s = float(C[2]/(b[2]*r_prime))
        if np.abs(s) < 1.:
            t1, t2 = -np.arcsin(s), np.arcsin(s) + np.pi
            if np.sin((t1+t2)/2.) < 0.:
                t1 += 2*np.pi

    t = np.linspace(t1, t2, 50)
    P = r_prime*np.cos(t)*a + r_prime*np.sin(t)*b + C
    xp, yp = p*P[0]/(p+P[2]), p*P[1]/(p+P[2]) # stereographic projection of the circle (trace)
    ax.plot(xp, yp, **kwargs)

    return ax

def draw_wulff_net(ax, step=9., theta=0., n=None, **kwargs):
    """
        step: angle between two adjacent traces
        theta: azimuthal angle
    """
    if n:
        step = 180./n 
    theta = theta*np.pi/180.
    ctheta, stheta = np.cos(theta), np.sin(theta)
    
    kw = dict(c='k', lw='.5')
    kw.update(kwargs)
    for t in np.arange(-90., 90., step)[1:]*np.pi/180.:
        draw_trace(ax, n=[-stheta,ctheta,0], offset=np.sin(t), **kw) # draw latitude traces
        draw_trace(ax, n=[ctheta*np.cos(t),stheta*np.cos(t),np.sin(t)], **kw) # draw longitude traces
    
    return ax

def draw_std_traces(ax, **kwargs):
    kw = dict(c='k', lw='.5')
    kw.update(kwargs)
    
    draw_trace(ax, n=[0,1,1], **kw)
    draw_trace(ax, n=[0,-1,1], **kw)
    draw_trace(ax, n=[1,0,1], **kw)
    draw_trace(ax, n=[-1,0,1], **kw)
    draw_trace(ax, n=[1,1,0], **kw)
    draw_trace(ax, n=[-1,1,0], **kw)
    
    return ax
