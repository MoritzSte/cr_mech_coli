import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle
from matplotlib import rc, rc_context
import pyvista as pv
import cr_mech_coli as crm
from copy import deepcopy

from cr_mech_coli.plotting import COLOR1


def plot_3d_rod(points, radius):
    pos = np.zeros((points.shape[0], 3), dtype=np.float32)
    pos[:, :2] = points
    vel = 0 * pos
    interaction = crm.PhysicalInteraction(
        crm.MiePotentialF32(radius, 0.1, 0.1, 0.1, 1.0, 2.0)
    )
    agent = crm.RodAgent(pos, vel, interaction)

    domain_size = np.max(points) + radius
    plotter = pv.Plotter(
        off_screen=True,
        window_size=[int(np.ceil(domain_size)) * 300] * 2,
    )
    p1 = np.array([-radius] * 3)
    p2 = np.array([domain_size, domain_size, -radius])
    p3 = np.array([-radius, domain_size, -radius])
    rect = pv.Rectangle(np.array([p1, p2, p3]))
    plotter.add_mesh(rect)
    bounds = (-radius, domain_size, -radius, domain_size, -radius, radius)
    pv.Plotter.view_xy(plotter, bounds=bounds)
    pv.Plotter.enable_parallel_projection(plotter)
    plotter.camera.tight(padding=0)
    plotter.camera.position = (*plotter.camera.position[:2], 100 * domain_size)
    camera = plotter.camera.copy()
    plotter.clear_actors()

    # Prepare perlin noise
    freq1 = [0.689, 0.562, 0.683]
    freq2 = [4.97, 5.2, 5.18]
    noise1 = pv.perlin_noise(0.80, freq1, (0, 0, 0))
    noise2 = pv.perlin_noise(0.40, freq2, (0, 0, 0))

    def noise(p):
        n1 = noise1.EvaluateFunction(p)
        n2 = noise2.EvaluateFunction(p)
        return n1 + n2

    z_resolution = 50
    theta_resolution = 100
    mesh = pv.Sphere(
        center=agent.pos[0],
        radius=float(agent.radius),
        theta_resolution=theta_resolution,
    )
    meshes = [mesh]
    for pos1, pos2 in zip(agent.pos[:-1], agent.pos[1:]):
        center = 0.5 * (pos1 + pos2)
        direction = pos2 - center
        height = float(np.linalg.norm(pos1 - pos2))
        cylinder = pv.CylinderStructured(
            center=center,
            direction=direction,
            radius=float(agent.radius),
            height=height,
            theta_resolution=theta_resolution,
            z_resolution=z_resolution,
        )
        meshes.append(cylinder)
        sphere = pv.Sphere(
            center=pos2,
            radius=float(agent.radius),
            theta_resolution=theta_resolution,
        )
        meshes.append(sphere)

    mesh = pv.merge(meshes)
    mesh = mesh.triangulate()
    mesh = mesh.extract_surface(nonlinear_subdivision=3)

    import pyacvd

    clus = pyacvd.Clustering(mesh)
    clus.subdivide(3)
    clus.cluster(20000, iso_try=20)
    mesh = clus.create_mesh()

    mesh["scalars1"] = [noise1.EvaluateFunction(p) for p in mesh.points]
    mesh = mesh.warp_by_scalar("scalars1")

    # mesh.smooth(n_iter=100, inplace=True, relaxation_factor=0.7)
    mesh.smooth(n_iter=20, inplace=True)
    mesh.subdivide(2)

    mesh["scalars2"] = [noise2.EvaluateFunction(p) for p in mesh.points]
    mesh = mesh.warp_by_scalar("scalars2")
    mesh.smooth(n_iter=5, inplace=True)

    actor = plotter.add_mesh(
        mesh,
        # show_edges=True,
        show_scalar_bar=False,
        scalars=None,
        color=[150, 150, 150],
        smooth_shading=True,
        metallic=0,
        roughness=1,
    )
    actor.UseBoundsOff()

    plotter.disable_anti_aliasing()
    img = np.array(plotter.screenshot())
    plotter.close()

    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()
    plt.close(fig)


def plot_3d_grid(points, radius):
    domain_size = np.max(points) + radius
    plotter = pv.Plotter(
        off_screen=True,
        window_size=[int(np.ceil(domain_size)) * 300] * 2,
    )
    p1 = np.array([-radius] * 3)
    p2 = np.array([domain_size, domain_size, -radius])
    p3 = np.array([-radius, domain_size, -radius])
    rect = pv.Rectangle(np.array([p1, p2, p3]))
    plotter.add_mesh(rect)
    bounds = (-radius, domain_size, -radius, domain_size, -radius, radius)
    pv.Plotter.view_xy(plotter, bounds=bounds)
    pv.Plotter.enable_parallel_projection(plotter)
    plotter.camera.tight(padding=0)
    plotter.camera.position = (*plotter.camera.position[:2], 100 * domain_size)
    plotter.clear_actors()
    axes = pv.AxesAssembly(shaft_radius=0.05)
    pv.Plotter.add_orientation_widget(plotter, axes, viewport=(0.7, 0.0, 1.0, 0.3))

    def add_mesh(mesh, color="white", edge_color="black"):
        plotter.add_mesh(
            mesh,
            show_edges=True,
            show_scalar_bar=False,
            opacity=0.9,
            color=color,
            edge_color=edge_color,
        )

    add_mesh(pv.Sphere(center=points[0], radius=radius), color=COLOR1)
    for j in range(max(points.shape[0] - 1, 0)):
        # Add sphere at junction
        add_mesh(pv.Sphere(center=points[j + 1], radius=radius), color=COLOR1)

        # Otherwise add cylinders
        pos1 = points[j]
        pos2 = points[j + 1]
        center = 0.5 * (pos1 + pos2)
        direction = pos2 - center
        height = float(np.linalg.norm(pos1 - pos2))
        cylinder = pv.Cylinder(
            center=center, direction=direction, radius=radius, height=height
        )
        add_mesh(cylinder)

    plotter.screenshot(filename="docs/source/_static/imaging-mesh.png")
    plotter.save_graphic(filename="docs/source/_static/imaging-mesh.pdf")


if __name__ == "__main__":
    points = np.array(
        [
            [0, 0],
            [1, 0.7],
            [2, 1.9],
            [3, 2.6],
            [4, 3.8],
            [5, 4.7],
        ]
    )
    points = np.array([points[:, 0], points[:, 1], np.zeros(len(points))]).T
    radius = 0.8

    # plot_3d_rod(points, radius)
    # plot_3d_rod(points, radius)

    plot_3d_grid(points, radius)
