import numpy as np
import matplotlib.pyplot as plt

from cr_mech_coli import crm_fit

if __name__ == "__main__":
    bound = 6.0
    cutoff = 3.5

    x = np.linspace(0.01, 3.9, 200)

    data = [
        (2, 1, ":"),
        (3, 2, "-."),
        (4, 2, "--"),
        (5, 3, "-"),
    ]

    fig_ax = None
    for en, em, ls in data:
        f = crm_fit.plot_mie_potential(x, 1, en, em, 1, bound, cutoff, fig_ax, ls)
        fig_ax = (f[0], f[1])
    fig, ax = fig_ax

    ax.set_xlim(0, np.max(x))
    ax.set_ylim(-2.5, 3)
    ax.legend()
    ax.set_title("Mie Potential")

    ax.set_xlabel("Distance [R$_1$+R$_2$]")
    ax.set_ylabel("Interaction Strength [V$_0$]")

    fig.savefig("docs/source/_static/mie-potential-shapes.png")
    fig.savefig("docs/source/_static/mie-potential-shapes.pdf")

    plt.close(fig)

    data2 = [(1.0, ":"), (1.5, "-."), (3.0, "--")]
    fig_ax = None
    for sti, ls in data2:
        label = f"ω={sti:3.1f}/(R$_1$+R$_2$)"
        f = crm_fit.plot_morse_potential(
            x,
            1,
            sti,
            1,
            cutoff,
            fig_ax,
            ls,
            label=label,
            yoffset=-1,
        )
        fig_ax = (f[0], f[1])
    fig, ax = fig_ax

    ax.set_xlim(0, np.max(x))
    xmin = -1
    xmax = 4
    dx = 0.05 * (xmax - xmin)
    ax.set_ylim(xmin - dx, xmax + dx)
    ax.legend()
    ax.set_title("Morse Potential")

    ax.set_xlabel("Distance [R$_1$+R$_2$]")
    ax.set_ylabel("Interaction Strength [V$_0$]")

    fig.savefig("docs/source/_static/morse-potential-shapes.png")
    fig.savefig("docs/source/_static/morse-potential-shapes.pdf")
