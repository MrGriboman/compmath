import numpy as np
import matplotlib.pyplot as plt


def exact(x, y, t):
    return t * np.exp(x + y)    


def f(x, y, t):
    return (1 - 2 * t) * np.exp(x + y)    


def l(size):
    lmat = -2 * np.identity(size + 1)
    ones1 = np.ones(size + 1)
    lmat += np.diag(ones1[1:], -1) + np.diag(ones1[:-1], 1)
    return lmat


def solve(nt=10, ny=10, nx=10):
    at, bt = 0, 1
    ax, bx = 0, 1
    ay, by = 0, 1

    tl, ht = np.linspace(at, bt, nt + 1, retstep=True)
    xl, hx = np.linspace(ax, bx, nx + 1, retstep=True)
    yl, hy = np.linspace(ay, by, ny + 1, retstep=True)

    u = np.zeros((nt + 1, ny + 1, nx + 1))

    # u[0] = exact(*np.meshgrid(xl, yl), 0)
    for ti in range(nt+1):
        u[ti] = exact(*np.meshgrid(xl, yl), tl[ti])

    l1 = -l(nx) / hx ** 2
    l2 = -l(ny) / hy ** 2

    I1 = np.identity(len(l1))
    I2 = np.identity(len(l2))

    mat1 = (I1 + ht / 2 * l1)
    mat2 = (I2 + ht / 2 * l2)

    mat1[[0, -1]] = 0
    mat1[0, 0] = mat1[-1, -1] = 1
    mat2[[0, -1]] = 0
    mat2[0, 0] = mat2[-1, -1] = 1

    for ti in range(1, nt + 1):
        print(f"\r{ti}", end="")
        t2 = (tl[ti] + tl[ti-1])/2
        htf = ht/2 * f(*np.meshgrid(xl, yl), t2)

        tmp = (I2 - ht / 2 * l2) @ u[ti - 1]
        tmp += htf
        tmp2 = np.zeros_like(tmp)
        for yi in range(ny+1):
            for j in [0, -1]:
                tmp[yi,j] = exact(xl[j], yl[yi], t2)
            tmp2[yi] = np.linalg.solve(mat1, tmp[yi])

        tmp = (I1 - ht / 2 * l1) @ tmp2.T
        tmp = tmp.T + htf
        tmp2 = np.zeros_like(tmp)
        for xi in range(nx+1):
            for j in [0, -1]:
                tmp[j, xi] = exact(xl[xi], yl[j], tl[ti])
            tmp2[:, xi] = np.linalg.solve(mat2, tmp[:, xi])

        u[ti,1:-1,1:-1] = tmp2.copy()[1:-1,1:-1]


    return xl, yl, tl, u



from matplotlib.animation import FuncAnimation


def plot_3d():
    nt = 1000
    n = 10
    x, y, t, u = solve(nt, n, n)

    xx, yy = np.meshgrid(x, y)
    ue = np.zeros_like(u)
    for ti in range(nt + 1):
        ue[ti] = exact(xx, yy, t[ti])

    ht = 1 / nt
    zlim = [np.min([u, ue]) - 2 * ht, np.max([u, ue]) + 2 * ht]
    # print(zlim)

    def plot3d(ax, solution, ti):
        return ax.plot_surface(xx, yy, solution[ti],
                               cmap='plasma',
                               linewidth=0,
                               antialiased=False)

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

    def update(ti):
        ax[0].clear()
        ax[1].clear()
        fig.suptitle(f"t = {np.round(t[ti], 2)}")
        ax[0].set_title("Численное")
        ax[1].set_title("Точное")
        ax[0].axes.set_zlim3d(bottom=zlim[0], top=zlim[1])
        ax[1].axes.set_zlim3d(bottom=zlim[0], top=zlim[1])
        plot3d(ax[0], u, ti)
        plot3d(ax[1], ue, ti)
        return

    update(0)

    frames, interval = range(0, nt + 1, nt // 10), 1000 / 10
    # print(nt + 1, nt // 10, frames)
    # print(f"ms: {interval}")

    ani = FuncAnimation(fig,
                        update,
                        frames=frames,
                        blit=False,
                        interval=interval)

    plt.show()

    diff = np.zeros(nt + 1)
    for ti in range(nt + 1):
        diff[ti] = np.max(abs(u[ti] - ue[ti])[:, :])
    # print(diff)

    plt.figure("Модуль разности")
    plt.plot(t, diff)

    plt.show()

def error_plot():
    n = 100
    ntl = [10, 100, 1000]
    for nt in ntl:
        x, y, t, u = solve(nt, n, n)
        xx, yy = np.meshgrid(x, y)
        ue = np.zeros_like(u)
        for ti in range(len(u)):
            ue[ti] = exact(xx, yy, t[ti])

        diff = np.zeros(len(u))
        for ti in range(len(u)):
            diff[ti] = np.max(abs(u[ti] - ue[ti])[:, :])

        print("", np.max(diff))

plot_3d()
# error_plot()