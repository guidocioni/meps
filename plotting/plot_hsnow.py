import seaborn as sns
from matplotlib.colors import from_levels_and_colors
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from functools import partial
import utils
import sys
from computations import compute_snow_change

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')


# The one employed for the figure name when exported
variable_name = 'hsnow'

utils.print_message('Starting script to plot '+variable_name)

# Get the projection as system argument from the call so that we can
# span multiple instances of this script outside
if not sys.argv[1:]:
    utils.print_message(
        'Projection not defined, falling back to default (scandinavia)')
    projection = 'scandinavia'
else:
    projection = sys.argv[1]


def main():
    """In the main function we basically read the files and prepare the variables to be plotted.
    This is not included in utils.py as it can change from case to case."""
    dset = utils.read_dataset(variables=['liquid_water_content_of_surface_snow',
                                         'altitude_of_0_degree_isotherm'],
                              projection=projection)
    dset['liquid_water_content_of_surface_snow'] = dset['liquid_water_content_of_surface_snow'] / 10.

    dset = compute_snow_change(dset, snowvar='liquid_water_content_of_surface_snow')

    levels_hsnow = (-50, -40, -30, -20, -10, -5, -2.5, -2, -1, -0.5,
                    0, 0.5, 1, 2, 2.5, 5, 10, 20, 30, 40, 50)
    levels_snowlmt = np.arange(0., 3000., 500.)

    cmap, norm = from_levels_and_colors(levels_hsnow,
                                        sns.color_palette("PuOr",
                                                          n_colors=len(levels_hsnow) + 1),
                                        extend='both')

    _ = plt.figure(figsize=(utils.figsize_x, utils.figsize_y))

    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = utils.get_projection(dset, projection, labels=True)
    m.fillcontinents(color='lightgray',lake_color='whitesmoke', zorder=0)

    dset = dset.drop(['longitude', 'latitude', 'liquid_water_content_of_surface_snow']).load()

    # All the arguments that need to be passed to the plotting function
    args = dict(m=m, x=x, y=y, ax=ax, cmap=cmap, norm=norm,
                levels_hsnow=levels_hsnow,
                levels_snowlmt=levels_snowlmt, time=dset.time)

    utils.print_message('Pre-processing finished, launching plotting scripts')
    if debug:
        plot_files(dset.isel(time=slice(-2, -1)), **args)
    else:
        # Parallelize the plotting by dividing into chunks and utils.processes
        dss = utils.chunks_dataset(dset, utils.chunks_size)
        plot_files_param = partial(plot_files, **args)
        p = Pool(utils.processes)
        p.map(plot_files_param, dss)


def plot_files(dss, **args):
    # Using args we don't have to change the prototype function if we want to add other parameters!
    first = True
    for time_sel in dss.time:
        data = dss.sel(time=time_sel)
        time, run, cum_hour = utils.get_time_run_cum(data)
        # Build the name of the output image
        filename = utils.subfolder_images[projection] + \
            '/' + variable_name + '_%s.png' % cum_hour

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['snow_increment'],
                                 extend='both',
                                 cmap=args['cmap'],
                                 norm=args['norm'],
                                 levels=args['levels_hsnow'])

        css = args['ax'].contour(args['x'], args['y'],
                                 data['snow_increment'],
                                 levels=args['levels_hsnow'],
                                 colors='gray',
                                 linewidths=0.2)

        labels2 = args['ax'].clabel(css, css.levels,
                                    inline=True, fmt='%4.0f', fontsize=6)

        c = args['ax'].contour(args['x'], args['y'],
                               data['altitude_of_0_degree_isotherm'],
                               levels=args['levels_snowlmt'],
                               colors='red', linewidths=0.5)

        labels = args['ax'].clabel(
            c, c.levels, inline=True, fmt='%4.0f', fontsize=5)

        an_fc = utils.annotation_forecast(args['ax'], time)
        an_var = utils.annotation(args['ax'],
                                  'Snow depth change [cm] since run beginning and snow limit [m]',
                                  loc='lower left', fontsize=6)
        an_run = utils.annotation_run(args['ax'], run)

        if first:
            cb = plt.colorbar(cs, orientation='horizontal', label='Snow depth change [m]',
                              pad=0.038, fraction=0.035, ticks=args['levels_hsnow'][::2])
            cb.ax.tick_params(labelsize=7)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **utils.options_savefig)

        utils.remove_collections(
            [c, cs, css, labels, labels2, an_fc, an_var, an_run])

        first = False


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed_time = time.time()-start_time
    utils.print_message(
        "script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
