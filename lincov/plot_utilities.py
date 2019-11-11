import numpy as np
from numpy.linalg import eig
from scipy.stats import chi2

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from mpl_toolkits.mplot3d import Axes3D, art3d

from lincov.frames import rotate_x, rotate_y, rotate_z

def projected_error_ellipsoid_points(C, n=100):
    """Get points for the projections of a 3D error ellipsoid.

    Reference:
    
    * Johnson, A. J. (2015). error_ellipse. Accessed 29 Oct 2019.
      https://www.mathworks.com/matlabcentral/fileexchange/4705-error_ellipse

    Args:
      C   3x3 positive definite covariance matrix
      n   number of points to draw

    Returns:
        A tuple whose members are each a numpy array of 1D
    coordinates.

    """
    eigval, eigvec = eig(C)
    
    p  = np.linspace(0, 2*np.pi, n) # angles around circle
    
    xy = np.vstack((np.cos(p), np.sin(p))).T.dot(np.diag(np.sqrt(eigval))).dot(eigvec.T)
    
    x  = xy[:,0]
    y  = xy[:,1]
    #z = np.zeros_like(x)
    return (x, y)#, z)


#def error_ellipsoid_points(C, n=100):
#    eigval, eigvec = eig(C)
#    
#    u = np.linspace(0, 2*np.pi, n)
#    v = np.linspace(0, np.pi, n)
#
#    x = np.outer(np.cos(u), np.sin(v))
#    y = np.outer(np.sin(u), np.sin(v))
#    z = np.outer(np.ones_like(u), np.cos(v))
#
#    a, b, c = np.sqrt(eigval).dot(eigvec.T)
#    
#    Tz = rotate_z(eigvec[1,0] / eigvec[0,0])
#    Tx = rotate_x(eigvec[2,0] / np.sqrt(eigvec[0,0]**2 + eigvec[1,0]**2))
#
#    return x * a, y * b, z * c


def error_ellipsoid(cov,
                    mu         = np.zeros(3),
                    dof        = 3,
                    #confidence = 0.95,
                    title      = None,
                    xlabel     = 'x',
                    ylabel     = 'y',
                    zlabel     = 'z',
                    aspect     = 'equal',
                    label      = None,
                    axes       = None,
                    linewidth  = 1):
    """Draw 3D error ellipsoid as a projection.

    Reference:
    
    * Johnson, A. J. (2015). error_ellipse. Accessed 29 Oct 2019.
      https://www.mathworks.com/matlabcentral/fileexchange/4705-error_ellipse

    Args:
      cov         3x3 covariance matrix (must be positive definite!)
      mu          length 3 mean vector
      dof         number of degrees of freedom to use for confidence 
                  interval; this defaults to 3, but you might set it 
                  higher if cov is drawn from a larger covariance 
                  matrix.
      confidence  the confidence interval (default: 0.95)
      title       optional title for the plot

    Returns:
        A figure and the tuple of axes that make up the figure.

    """
    if axes is None:
        fig = plt.figure()
        ax00 = fig.add_subplot(221)
        ax01 = fig.add_subplot(223, sharex=ax00)
        ax10 = fig.add_subplot(222, sharey=ax00)
        axes = (ax00, ax01, ax10)
        existing_min, existing_max = None, None
        #ax11 = fig.add_subplot(224, projection='3d', sharex=ax00, sharey=ax00)
    else:
        fig = axes[0].get_figure()
        ax00 = axes[0]
        ax01 = axes[1]
        ax10 = axes[2]
        existing_min, existing_max = ax00.get_xlim()
        
    plt.setp(ax00.get_xticklabels(), visible=False)
    plt.setp(ax10.get_yticklabels(), visible=False)
    fig.subplots_adjust(hspace=0, wspace=0)

    if title is not None:
        fig.suptitle(title)
    
    
    #lam, v = eig(cov)
    cxy = cov[0:2,0:2]
    cyz = cov[1:3,1:3]
    cxz = np.array([[cov[0,0], cov[0,2]],
                    [cov[2,0], cov[2,2]]])

    k = 1.0 #np.sqrt(chi2.isf(1.0 - confidence, dof))
    print("k = {}".format(k))

    x,y = projected_error_ellipsoid_points(cxy)
    h1 = ax00.plot(mu[0] + k * x, mu[1] + k * y, linewidth=linewidth, label=label, alpha=0.7)
    ax00.set_ylabel(ylabel)
    ax00.grid(True)
    #h1 = axes00.plot(mu[0] + k * x, mu[1] + k * y, mu[2] + k * z, c='r') # 3d projection

    y,z = projected_error_ellipsoid_points(cyz)
    h2 = ax10.plot(mu[2] + k * z, mu[1] + k * y, linewidth=linewidth, label=label, alpha=0.7)
    ax10.set_xlabel(zlabel)
    ax10.grid(True)
    #h2 = axes01.plot(mu[0] + k * x, mu[1] + k * y, mu[2] + k * z, c='g') # 3d projection

    x,z = projected_error_ellipsoid_points(cxz)
    h3 = ax01.plot(mu[0] + k * x, mu[2] + k * z, linewidth=linewidth, label=label, alpha=0.7)
    ax01.set_xlabel(xlabel)
    ax01.set_ylabel(zlabel)
    ax01.grid(True)
    #h3 = axes10.plot(mu[0] + k * x, mu[1] + k * y, mu[2] + k * z, c='b') # 3d projection

    # Make sure axes are scaled appropriately for any new data added
    this_min = np.min((mu[0] + k * x, mu[1] + k * y, mu[2] + k * z))
    this_max = np.max((mu[0] + k * x, mu[1] + k * y, mu[2] + k * z))
    if existing_min is not None:
        this_min = np.min((this_min, existing_min))
        this_max = np.max((this_max, existing_max))

    min_max = (this_min, this_max)
    ax00.set_xlim(min_max)
    ax00.set_ylim(min_max)
    ax01.set_ylim(min_max)
    ax10.set_xlim(min_max)

    #x,y,z = error_ellipsoid_points(cov)
    #h3 = ax11.plot_surface(x,y,z, rstride=4,cstride=4, alpha=0.5, color='grey')

    return fig, axes
    
