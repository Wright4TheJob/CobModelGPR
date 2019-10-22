"""Perform material analysis and compares model to data."""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle


class PlotSettings():
    """Create settings for plotting functions."""

    def __init__(self):
        """Input all settings to instance."""
        self.show = False
        self.title = 'Untitled'
        self.filename = 'Untitled'
        self.save = True
        self.scale = 100
        self.n = 100
        self.x_label = '$x$'
        self.y_label = '$f(x)$'
        self.z_label = '$f(x,y)$'


def make_ternary_dict(x, y):
    """Create a dictionary of location pairs and resulting values.

    :param x: A list of 2- or 3-element lists noting independant parameters
    :type x: list
    :param y: A 1-D list of values mapping to the position list
    :type x: list
    :returns:  dict -- the dictionary to send to ternary heatmap data function.
    :raises: AttributeError, KeyError

    """
    new_dict = {}
    for x_point, y_point in zip(x, y):
        new_key = (x_point[0], x_point[1])
        new_val = y_point
        new_dict[new_key] = new_val
    return new_dict


def ternary_dat(data, data_points=None, settings=None):
    """Generate a ternary plot heatmap from data dictionary.

    :param data: A dictionary using a value pair as a key and returning
    the measured value.
    :type data: dict

    :param save: Save an image of the resulting plot.
    :type save: bool, optional

    :param show: Display an image of the resulting plot.
    :type show: bool, optional

    :param title: Title to be used for the figure and saved file.
    :type title: string, optional

    :param scale: Scale for data and ternary plot.
    :type scale: int, optional

    """
    import ternary
    if settings is None:
        settings = PlotSettings()

    filename = settings.title
    filename += '.png'
    figure, tax = ternary.figure(scale=settings.scale)
    figure.set_size_inches(10, 8)
    tax.boundary(linewidth=2.0)
    tax.heatmap(data, cmap=None, style="triangular")
    tax.set_title('Gaussian Process Regression Model - Compression')
    tax.ticks(axis='lbr', linewidth=1, multiple=5)
    fontsize = 20
    tax.left_axis_label("Straw", fontsize=fontsize)
    tax.right_axis_label("Clay", fontsize=fontsize)
    tax.bottom_axis_label("Sand", fontsize=fontsize)
    tax.clear_matplotlib_ticks()
    if data_points is not None:
        tax.scatter(data_points,
                    marker='o',
                    color='red',
                    label="Experimental Results",
                    zorder=2)
    tax._redraw_labels()
    if settings.save:
        tax.savefig(filename)
    if settings.show:
        tax.show()


def ternary_analytical(func, settings=None):
    """Create a ternary plot heatmap from analytical model.

    Figure is scaled from 0-100. Input function is expected to accept values in
    this range.

    :param func: A function accepting a 2- or 3-parameter list and returning a
    single value.
    :type func: function

    :param save: Save an image of the resulting plot.
    :type save: bool, optional

    :param show: Display an image of the resulting plot.
    :type show: bool, optional

    :param title: Title to be used for the figure and saved file.
    :type title: string, optional

    :param scale: Scale for data and ternary plot.
    :type scale: int, optional

    """
    if settings is None:
        settings = PlotSettings()

    x_plot_data = []
    for i in range(0, settings.scale+1):
        for j in range(0, settings.scale+1):
            if i + j <= settings.scale:
                x_plot_data.append([i, j, settings.scale-i-j])
    y_plot_data = [func(point) for point in x_plot_data]

    dict_data = make_ternary_dict(x_plot_data, y_plot_data)
    ternary_dat(dict_data, settings=settings)


def ternary_gpr(mixes, values, model, stddevs=False, settings=None):
    """Create a ternary plot with scatter points and GPR heatmap.

    Figure is scaled from 0-100. Input points are expected to be normalized to
    this range.

    :param mixes: A list of 3-element lists of test mixes used.
    :type mixes: list

    :param values: The resulting value of the measured property of the mix.
    :type values: list

    :param model: A trained gaussinan process regressor object.
    :type model: Gaussian Process Regressor.

    :param stddevs: The resulting standard deviation of the measured
    property of the mix.
    :type stddevs: list

    :param settings: Configured settings for plotting.
    :type scale: :class:`Settings`

    :raises: ValueError

    """
    if settings is None:
        settings = PlotSettings()

    if len(mixes) != len(values):
        raise ValueError('Length of x and mean lists must be equal')
    if stddevs:
        if len(values) != len(stddevs):
            raise ValueError('Length of mean and stddev lists must be equal')

    data_points = []
    for point in mixes:
        data_points.append((point[0], point[1], point[2]))

    x_plot_data = []
    for i in range(0, 101):
        for j in range(0, 101):
            if i + j <= 100:
                x_plot_data.append([i, j, 100-i-j])
    y_plot_data = model.predict(np.array(x_plot_data), return_std=False)

    ternary_dat(
        make_ternary_dict(x_plot_data, y_plot_data),
        data_points=data_points, settings=settings)


def train_gpr(xs, ys, stddevs=False):
    """Train a GPR model on 2D data and return model for use.

    :param xs: List of independant conditions tested. If 2D, list of tuples.
    :type xs: List

    :param ys: List of response values, equal in length to xs.
    :type ys: List.

    :returns: Trained Gaussian Process Regressor object.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF

    if len(xs) != len(ys):
        raise ValueError('Length of x and mean lists must be equal')
    if stddevs:
        if len(xs) != len(stddevs):
            raise ValueError('Length of mean and stddev lists must be equal')

    # kernel = 1**2*Matern(length_scale=0.2, nu=0.5)
    kernel = 0.1**2*RBF(length_scale=0.2)
    if not stddevs:
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=1000)
    else:
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel,
            alpha=np.array([dev**2 for dev in stddevs]),
            n_restarts_optimizer=20)

    # print(xs)
    x_array = np.array(xs)
    if len(x_array[0]) == 1:
        xs = np.atleast_2d(x_array).T
    y_array = np.array(ys)
    # fit the data using Maximum Likelihood Estimation of parameters
    gaussian_process.fit(x_array, y_array)
    goodness = gaussian_process.score(x_array, y_array)
    print("R^2 = %2.4f" % (goodness))
    return gaussian_process


def plot_contour(model, xrange=[0, 1], yrange=[0, 1], settings=None,
                 standard=2.01):
    """Create a contour plot over given bounds from model provided.

    :param model: An object containing a 'predict' function returning a value.
    :type model: Object

    """

    if settings is None:
        settings = PlotSettings()

    X, Y, Z = create_surface(model, xrange=xrange, yrange=yrange, n=settings.n)

    filename = settings.title
    filename += '.png'

    # origin = 'lower'
    plt.figure()
    CS = plt.contour(X, Y, Z, levels=(0.9, 1.0, 1.25, 1.5, 1.75, 1.9, 2.0,
                     2.068, 2.1, 2.2, 2.3, ),
                     colors=['k', 'k', 'k', 'k', 'k', 'k', 'k', 'r', 'k', 'k',
                             'k'])
    # CS2 = plt.contour(CS, levels=CS.levels[19:20], colors='r', origin=origin)
    # Make a colorbar for the ContourSet returned by the contourf call.
    # cbar = plt.colorbar(CS)
    # Add the contour line levels to the colorbar
    # cbar.add_lines(CS2)
    plt.clabel(CS, inline=1, fontsize=10)
    # cbar.ax.set_ylabel('Strength [MPa]')
    plt.title(settings.title)
    plt.xlabel(settings.x_label)
    plt.ylabel(settings.y_label)

    if settings.save:
        plt.savefig(filename)
    if settings.show:
        plt.show()


def plot_spinning_surface(positions, values, model, settings=None, steps=60):
    """Plots n images rotating a surface and 3D scatter points."""
    step_angle = 360/steps
    angles = [230 + step_angle * i for i in range(0, steps)]
    number = 1
    base_title = settings.title
    for angle in angles:
        settings.filename = base_title + str(number)
        plot_surface(positions, values, model, settings=settings,
                     rotation=angle)
        number = number + 1


def plot_surface(positions, values, model, settings=None, rotation=230):
    """Create a response surface and scatter points in ordinary 3D space."""
    from matplotlib import cm

    if settings is None:
        settings = PlotSettings()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    xs = [x[1] for x in positions]
    ys = [x[2] for x in positions]
    zs = values

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    ax.scatter(xs, ys, zs, c='r', marker='o')

    x_max, x_min = extrema(xs)
    x_lims = [x_min, x_max]
    y_max, y_min = extrema(ys)
    y_lims = [y_min, y_max]
    X, Y, Z = create_surface(model, xrange=x_lims, yrange=y_lims, n=100)

    # Plot the surface.
    surf = ax.plot_wireframe(X, Y, Z, color='k', rcount=25, ccount=25)
    # surf = ax.plot_wireframe(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    ax.view_init(elev=20., azim=rotation)
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    filename = settings.filename
    filename += '.png'

    ax.set_title(settings.title)
    ax.set_xlabel('Clay Mass Fraction')
    ax.set_ylabel('Straw Mass Fraction')
    ax.set_zlabel(settings.y_label)
    # ax.legend(loc='upper left')
    if settings.save:
        plt.savefig(filename)
    if settings.show:
        plt.show()
    else:
        plt.close()


def create_surface(model, xrange=[0, 1], yrange=[0, 1], n=10):
    """Create a response surface over specified range from given model.

    :param model: Generator for response surface, accepts 3-value list input.
    :type model: Object, containing 'predict' function which returns value.

    """
    X = np.linspace(xrange[0], xrange[1], n)
    Y = np.linspace(yrange[0], yrange[1], n)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            remainder = 1 - X[i, j] - Y[i, j]
            Z[i, j] = model.predict([[remainder, X[i, j], Y[i, j]]])
    return X, Y, Z


def plot_2d(xs, means, models, x_axis="straw", stddevs=False, settings=None):
    """Create a 2D plot with scatter points and GPR model and confidence bound.

    :param xs: A 1-D list of x values.
    :type xs: list

    :param means: 1-D list of mean responses. Must be the same length as xs.
    :type means: list

    :param models: List of models to plot against data.
    :type models: Object, containing 'predict' function which returns value.

    :param stddevs: Standard deviation of each response.
    :type stddevs: list, optional

    :param settings: Configured plotting settings object.
    :type title: :class:`Settings`

    :raises: ValueError

    """
    if settings is None:
        settings = PlotSettings()

    if len(xs) != len(means):
        raise ValueError('Length of x and mean lists must be equal')
    if stddevs:
        if len(means) != len(stddevs):
            raise ValueError('Length of mean and stddev lists must be equal')

    # Independant data transforms
    xarray = np.array(xs)
    if len(xs[0]) == 1:
        xarray = np.atleast_2d(xarray).T

    x_max, x_min = extrema(xs)

    # mesh input space for evaluations of function and prediction
    n = 100
    x_pred = np.array(distribute_points(x_min, x_max, n=n))
    # predict based on meshed x-axis (using MSE)
    curves = []
    model_index = 0

    for model in models:
        if x_axis == "straw":
            xpoints = [x[2] for x in xs]
            x_plot = [x[2] for x in x_pred]
        elif x_axis == "clay":
            xpoints = [x[1] for x in xs]
            x_plot = [x[1] for x in x_pred]
        curve = [x_plot]
        if model_index == 0:
            y_pred, sigma = model.predict(x_pred, return_std=True)
            curve.append(y_pred)
            curve.append(sigma)
        else:
            y_pred = model.predict(x_pred, return_std=False)
            curve.append(y_pred)
        curves.append(curve)
        model_index = model_index + 1
    # plot
    labels = ['GPR Model', 'Analytical Model']
    plot_2d_dumb(xpoints, means, stddevs=stddevs, curves=curves,
                 labels=labels, settings=settings)


def plot_2d_dumb(xpoints, ypoints, stddevs=False,
                 curves=False, labels=False, settings=None):
    """Plot points and curves on a 2D plot.

    :param xpoints: List of x-positions of points.
    :type xpoints: List of lists

    :param ypoints: List of y-positions of points.
    :type xpoints: List of lists

    :param stddevs: Optional, standard deviation for plotting error bars.
    :type stddevs: List

    :param curves: X, y, and optional std. dev. values of curves to plot.
    :type curves: List of pairs of lists

    """
    if settings is None:
        settings = PlotSettings()

    if len(xpoints) != len(ypoints):
        raise ValueError('Length of x and mean lists must be equal')
    if stddevs:
        if len(ypoints) != len(stddevs):
            raise ValueError('Length of mean and stddev lists must be equal')

    # plot
    filename = settings.title
    filename += '.png'
    plt.figure()
    if stddevs is None:
        plt.plot(xpoints, ypoints, 'r.', markersize=10, label=u'Observations')
    else:
        plt.errorbar(
            xpoints, ypoints, stddevs, fmt='r.',
            markersize=10, label=u'Observations')

    if labels is False:
        labels = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth",
                  "Seventh", "Eight", "Ninth", "strength"]
        if curves:
            labels = labels[:len(curves)]

    colors = ['b-', 'k-', 'g-', 'r-']
    if curves:
        colors = colors[:len(curves)]
    if curves:
        for curve, label, color in zip(curves, labels, colors):
            xs = curve[0]
            ys = curve[1]
            if len(curve) == 3:
                sigmas = curve[2]
            else:
                sigmas = []
            plt.plot(xs, ys, color, label=label)
            if len(sigmas) != 0:
                plt.fill(
                    np.concatenate([xs, xs[::-1]]),
                    np.concatenate(
                        [ys - 1.96*np.array(sigmas),
                         (ys + 1.9600*np.array(sigmas))[::-1]]),
                    alpha=0.5, fc='b', ec='None', label=label)
    plt.title(settings.title)
    plt.xlabel(settings.x_label)
    plt.ylabel(settings.y_label)
    plt.legend(loc='upper left')
    if settings.save:
        plt.savefig(filename)
    if settings.show:
        plt.show()


def plot_scatter(xs, ys, stddevs=False, settings=None):
    """Plot scatter points using settings object.
    :param xs: X values of points to be plotted.
    :type xs: List

    :param ys: Y values of points to be plotted.
    :type ys: List

    :param stddevs: Optional, standard deviation for plotting error bars.
    :type stddevs: List

    :param settings: Configured plotting settings object.
    :type title: :class:`Settings`

    """

    if settings is None:
        settings = PlotSettings()

    if len(xs) != len(ys):
        raise ValueError('Length of x and mean lists must be equal')
    if stddevs:
        if len(ys) != len(stddevs):
            raise ValueError('Length of mean and stddev lists must be equal')

    xs = np.array(xs)

    # response data transforms
    ys = np.array(ys)
    stddevs = np.array(stddevs)

    # plot
    filename = settings.title
    filename += '.png'
    plt.figure()
    if stddevs is None:
        plt.plot(
            xs, ys, 'r.',
            markersize=10, label=u'Observations')
    else:
        plt.errorbar(
            xs, ys, stddevs, fmt='r.', markersize=10, label=u'Observations')

    plt.title(settings.title)
    plt.xlabel(settings.x_label)
    plt.ylabel(settings.y_label)
    if settings.save:
        plt.savefig(filename)
    if settings.show:
        plt.show()


def plot_box_whisker(data, settings=None):
    """Create box and whisker plot for data.

    :param data: Data to be plotted - y axis values only.
    :type data: list of lists

    """
    if settings is None:
        settings = PlotSettings()
    # plot
    filename = settings.title
    filename += '.png'
    plt.figure()
    plt.boxplot(data)
    plt.title(settings.title)
    plt.xlabel(settings.x_label)
    plt.ylabel(settings.y_label)
    if settings.save:
        plt.savefig(filename)
    if settings.show:
        plt.show()
    return


def extrema(data):
    """Find the two points furthest away from each other.

    :param data: N-dimensional data points to analyze.
    :type data: List

    :returns: Touple of data points furthest apart.

    :raises: ValueError
    """
    distances = []
    if len(data) < 2:
        raise ValueError('Must have at least two data points to get distance.')

    for from_point in data:
        new_distances = []
        for to_point in data:
            dist = vlen(from_point, to_point)
            new_distances.append(dist)
        distances.append(new_distances)
    max_dist = 0
    max_i = -1
    max_j = -1
    for i in range(0, len(distances)):
        for j in range(0, len(distances[i])):
            if distances[i][j] > max_dist:
                max_dist = distances[i][j]
                max_i = i
                max_j = j

    return (data[max_i], data[max_j])


def vlen(pointa, pointb):
    """Calculate vector length between two points."""
    from numbers import Number
    sum_squares = 0
    if isinstance(pointa, np.ndarray):
        pointa = pointa.tolist()
        pointb = pointb.tolist()
    if isinstance(pointa, list):
        for dim_a, dim_b in zip(pointa, pointb):
            sum_squares += (dim_a - dim_b)**2
            dist = np.sqrt(sum_squares)
    elif isinstance(pointa, Number):
        dist = abs(pointa - pointb)
    else:
        print("Vector length cannot be calcualted between: " + repr(pointa) +
              ", " + repr(pointb))
    return dist


def plot_data_model(data_x, data_y, model_x, model_y,
                    data_stddev=False,
                    model_stddev=False,
                    settings=None):
    """Plot data points and model curve with error bounds."""
    # create plot bounds
    x_max = max(data_x)
    x_min = min(data_y)
    dx = x_max - x_min
    plot_margin = 0.2  # 10% of delta size
    x_min_plot = x_min - plot_margin*dx
    x_max_plot = x_max + plot_margin*dx

    if settings is None:
        settings = PlotSettings()

    filename = settings.title
    filename += '.png'
    plt.figure()
    if not data_stddev:
        plt.plot(data_x, data_y, 'r.', markersize=10, label=u'Observations')
    else:
        plt.errorbar(
            data_x.ravel(),
            data_y, data_stddev,
            fmt='r.',
            markersize=10,
            label=u'Observations')

    plt.plot(model_x, model_y, 'b-', label=u'Model')
    if not model_stddev:
        plt.fill(
            np.concatenate([model_x, model_x[::-1]]),
            np.concatenate([model_y - 1.96*model_stddev,
                            (model_y + 1.9600*model_stddev)[::-1]]),
            alpha=0.5, fc='b', ec='None', label='95% confidence interval')
    plt.title(settings.title)
    plt.xlabel(settings.x_label)
    plt.xlabel(settings.y_label)
    plt.xlim(x_min_plot, x_max_plot)
    plt.legend(loc='upper left')
    if settings.save:
        plt.savefig(filename)
    if settings.show:
        plt.show()


def distribute_points(start, end, n=1000):
    """Space points for plotting sand, clay, and straw.

    :param start: Initial point for spacing. 1 Value for each data dimension.
    :type start: List

    :param end: Final point for spacing. Same length as start.
    :type end: List

    :returns: List of points equally spaced from start to end.
    """
    sand_points = np.linspace(start[0], end[0], num=n).tolist()
    clay_points = np.linspace(start[1], end[1], num=n).tolist()
    straw_points = np.linspace(start[2], end[2], num=n).tolist()
    points = [[x, y, z] for x, y, z in zip(
        sand_points, clay_points, straw_points)]
    return points


def main():
    """Primary script for manual and GPR model fitting and plotting.

    Performs GPR fits and manual strength plotting for each of the tests
    of cob. Saves plot files.

    """
    # import TernaryPlots
    import mechanical
    compressive_cob = mechanical.Cob()
    compressive_cob.predict_function = compressive_cob.compressive_strength
    bending_cob = mechanical.Cob()
    bending_cob.predict_function = bending_cob.flexure_strength

    # data from transient study
    # mixes in the order [sand, clay, straw]
    base_mix = [[0.655, 0.33, 0.005]]
    base_comp_strength = [1.9908]
    base_comp_stddev = [0.1481]
    base_bend_strength = [1.0845]
    base_bend_stddev = [0.14762]
    # straw vs strength test, ratios by weight
    straw_mixes = [
        [0.9136, 0.0752, 0.0112],
        [0.9113, 0.0807, 0.0080],
        [0.9201, 0.0762, 0.0037],
        [0.9175, 0.0825, 0.00]]

    straw_comp_strengths = [1.3480, 1.5342, 1.2107, 1.0292]
    straw_comp_stddevs = [0.2891, 0.2707, 0.2488, 0.1277]

    straw_bend_strengths = [0.7053, 0.8223, 0.6809, 0.7555]
    straw_bend_stddevs = [0.1016, 0.1064, 0.0437, 0.1399]
    # clay vs strength test
    clay_mixes = [
        [0.9471, 0.0496, 0.0033],
        [0.8543, 0.1420, 0.0036],
        [0.8055, 0.1904, 0.0041],
        [0.7374, 0.2584, 0.0041]]

    clay_comp_strengths = [0.7735, 2.4542, 2.3447, 2.2901]
    clay_comp_stddevs = [0.7489, 0.2007, 0.0957, 0.2236]

    clay_bend_strengths = [0.7786, 0.8947, 0.8697, 0.8206]
    clay_bend_stddevs = [0.2290, 0.1326, 0.1850, 0.2129]

    """
    location_mix = [[0.8994, 0.0950,	0.0056]]
    location_strength_comp = [1.3546]
    location_stddev_comp = [0.1306]
    location_strength_bend = [1.0487]
    location_stddev_bend = [0.0854]

    peak_mix = [[0.8039, 0.1868, 0.0093]]
    peak_strength_comp = [1.9992]
    peak_stddev_comp = [0.0344]
    peak_strength_bend = [0.80063]
    peak_stddev_bend = [0.02670]
    """

    # putting all sets together
    all_mixes = base_mix + straw_mixes + clay_mixes #+ location_mix + peak_mix
    all_comp_strengths = base_comp_strength + \
        straw_comp_strengths + clay_comp_strengths #+ location_strength_comp + \
        # peak_strength_comp  # + matrix_comp_strengths
    all_comp_stddevs = base_comp_stddev + \
        straw_comp_stddevs + clay_comp_stddevs #+ location_stddev_comp + \
        # peak_stddev_comp  # + matrix_comp_stddevs
    all_bend_strengths = base_bend_strength + \
        straw_bend_strengths + clay_bend_strengths # + location_strength_bend + \
        # peak_strength_bend
    all_bend_stddevs = base_bend_stddev + \
        straw_bend_stddevs + clay_bend_stddevs # + location_stddev_bend + \
        # peak_stddev_bend

    plot_straw_mixes_comp = straw_mixes  # + [clay_mixes[2]]
    plot_straw_comp_strengths = straw_comp_strengths
    # + [clay_comp_strengths[2]]
    plot_straw_comp_stddevs = straw_comp_stddevs  # + [clay_comp_stddevs[2]]

    plot_straw_mixes_bend = straw_mixes  # + [clay_mixes[2]]
    plot_straw_bend_strengths = straw_bend_strengths
    # + [clay_comp_strengths[2]]
    plot_straw_bend_stddevs = straw_bend_stddevs  # + [clay_comp_strengths[2]]

    plot_clay_mixes_comp = [straw_mixes[2]] + clay_mixes
    plot_clay_comp_strengths = [straw_comp_strengths[2]] + clay_comp_strengths
    plot_clay_comp_stddevs = [straw_comp_stddevs[2]] + clay_comp_stddevs

    plot_clay_mixes_bend = [straw_mixes[2]] + clay_mixes
    plot_clay_bend_strengths = [straw_comp_strengths[2]] + clay_bend_strengths
    plot_clay_bend_stddevs = [straw_comp_strengths[2]] + clay_bend_stddevs

    print('Training GPR models...')
    gpr_comp = train_gpr(all_mixes, all_comp_strengths, all_comp_stddevs)
    # gpr_bend = train_gpr(all_mixes, all_bend_strengths, all_bend_stddevs)

    # Pickle compressive model
    file_Name = "cob_compression_gpr"
    fileObject = open(file_Name, 'wb')
    pickle.dump(gpr_comp, fileObject)
    fileObject.close()

    # print("Compressive GPR: %s" % (gpr_comp))

    print('Generating 2D plots...')
    straw_points = [mix[2] for mix in all_mixes]
    clay_points = [mix[1] for mix in all_mixes]
    # mix locations rectangular grid
    print('1 of 5')
    settings = PlotSettings()
    settings.title = 'Test Mix Locations'
    settings.x_label = 'Clay Soil Fraction [Weight %]'
    settings.y_label = 'Straw Fraction [Weight %]'
    plot_scatter(clay_points, straw_points, settings=settings)

    # straw test plots
    print('2 of 5')
    settings = PlotSettings()
    settings.title = 'Compressive Strength Straw'
    settings.x_label = 'Straw Mass Fraction'
    settings.y_label = 'Compression Strength [MPa]'
    plot_2d(
        plot_straw_mixes_comp,
        plot_straw_comp_strengths,
        [gpr_comp, compressive_cob],
        stddevs=plot_straw_comp_stddevs, settings=settings)

    """
    print('3 of 5')
    settings = PlotSettings()
    settings.title = 'Bending Strength Straw'
    plot_2d(
        plot_straw_mixes_bend, plot_straw_bend_strengths,
        [gpr_bend, bending_cob],
        stddevs=plot_straw_bend_stddevs, settings=settings)
    """

    # Clay test plots
    print('4 of 5')
    settings = PlotSettings()
    settings.title = 'Compressive Strength Clay'
    settings.x_label = 'Clay Mass Fraction'
    settings.y_label = 'Compression Strength [MPa]'
    plot_2d(
        plot_clay_mixes_comp, plot_clay_comp_strengths,
        [gpr_comp, compressive_cob],
        stddevs=plot_clay_comp_stddevs, x_axis="clay", settings=settings)

    """
    print('5 of 5')
    settings = PlotSettings()
    settings.title = 'Bending Strength Clay'
    plot_2d(
        plot_clay_mixes_bend, plot_clay_bend_strengths,
        [gpr_bend, bending_cob],
        stddevs=plot_clay_bend_stddevs, settings=settings)
    """
    """
    print('Generating ternary plots...')
    # Ternary plots
    print('1 of 4')
    ternary_scale = 100
    settings = PlotSettings()
    settings.title = 'Compressive Strength GPR'
    settings.scale = ternary_scale
    ternary_gpr(
        all_mixes, all_comp_strengths, gpr_comp,
        stddevs=all_comp_stddevs, settings=settings)
    settings = PlotSettings()
    settings.title = 'Bending Strength GPR'
    settings.scale = ternary_scale
    print('2 of 4')
    ternary_gpr(
        all_mixes, all_bend_strengths, gpr_bend,
        stddevs=all_bend_stddevs, settings=settings)
    """
    """
    settings = PlotSettings()
    settings.title = 'Compressive Matrix Model'
    settings.scale = ternary_scale
    print('3 of 7')
    ternary_analytical(cob.compressive_strength_matrix, settings=settings)
    """
    """
    settings = PlotSettings()
    settings.title = 'Compressive Strength Model'
    settings.scale = ternary_scale
    print('3 of 4')
    ternary_analytical(cob.compressive_strength, settings=settings)

    settings = PlotSettings()
    settings.title = 'Bending Strength Model'
    settings.scale = ternary_scale
    print('4 of 4')
    ternary_analytical(cob.flexure_strength, settings=settings)
    """
    """
    settings = PlotSettings()
    settings.title = 'Modulus Model'
    settings.scale = ternary_scale
    print('6 of 7')
    ternary_analytical(cob.modulus, settings=settings)

    settings = PlotSettings()
    settings.title = 'Shrinkage Cracking'
    settings.scale = ternary_scale
    print('7 of 7')
    ternary_analytical(cob.shrinkage_cracking_derating, settings=settings)
    """

    print('Generating contour plots...')
    print('1 of 2')
    settings = PlotSettings()
    # settings.show = True
    settings.n = 100
    settings.title = 'Compressive Strength GPR Contour'
    settings.x_label = 'Clay Mass Fraction'
    settings.y_label = 'Straw Mass Fraction'
    plot_contour(gpr_comp, xrange=[0.01, 0.4], yrange=[0, 0.015],
                 settings=settings, standard=300)

    print('2 of 2')
    settings = PlotSettings()
    # settings.show = True
    settings.n = 100
    settings.title = 'Compressive Strength Model Contour'
    settings.x_label = 'Clay Mass Fraction'
    settings.y_label = 'Straw Mass Fraction'
    plot_contour(compressive_cob, xrange=[0.01, 0.4], yrange=[0, 0.015],
                 settings=settings,
                 standard=300)

    print('Generating surface plots...')
    print('1 of 4')
    settings = PlotSettings()
    # settings.show = True
    settings.title = 'Compressive Strength GPR'
    settings.y_label = 'Compression Strength [MPa]'
    plot_spinning_surface(all_mixes, all_comp_strengths, gpr_comp,
                          settings=settings)

    """
    print('2 of 4')
    settings = PlotSettings()
    # settings.show = True
    settings.title = 'Bending Strength GPR'
    plot_surface(all_mixes, all_bend_strengths, gpr_bend, settings=settings)
    """

    print('3 of 4')
    settings = PlotSettings()
    # settings.show = True
    settings.title = 'Compressive Strength Model'
    # settings.show = True
    settings.y_label = 'Compression Strength [MPa]'
    plot_spinning_surface(all_mixes, all_comp_strengths, compressive_cob,
                          settings=settings)

    """
    print('4 of 4')
    settings = PlotSettings()
    # settings.show = True
    settings.title = 'Bending Strength Model'
    plot_surface(all_mixes, all_bend_strengths, bending_cob, settings=settings)
    """


if __name__ == "__main__":
    main()
