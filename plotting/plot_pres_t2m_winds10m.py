import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from functools import partial
import utils
import sys
import metpy.calc as mpcalc


# The one employed for the figure name when exported
variable_name = 't_v_pres'

utils.print_message('Starting script to plot '+variable_name)

# Get the projection as system argument from the call so that we can
# span multiple instances of this script outside
if not sys.argv[1:]:
    projection = 'scandinavia'
else:
    projection = sys.argv[1]


def main():
    """In the main function we basically read the files and prepare the variables to be plotted.
    This is not included in utils.py as it can change from case to case."""
    dset = utils.read_dataset(variables=['x_wind_10m',
                                         'y_wind_10m',
                                         'air_temperature_2m',
                                         'air_pressure_at_sea_level'],
                              projection=projection)

    dset['air_temperature_2m'] = dset['air_temperature_2m'].metpy.convert_units('degC').metpy.dequantify()
    dset['air_pressure_at_sea_level'] = dset['air_pressure_at_sea_level'].metpy.convert_units('hPa').metpy.dequantify()

    levels_t2m = np.arange(-25, 45, 1)

    cmap = utils.get_colormap("temp")
    _ = plt.figure(figsize=(utils.figsize_x, utils.figsize_y))

    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = utils.get_projection(dset, projection, labels=True)

    dset = dset.drop(['longitude', 'latitude']).load()

    levels_mslp = np.arange(dset['air_pressure_at_sea_level'].min().astype("int"),
                            dset['air_pressure_at_sea_level'].max().astype("int"), 3.)

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, cmap=cmap,
                levels_t2m=levels_t2m, levels_mslp=levels_mslp,
                time=dset.time)

    utils.print_message('Pre-processing finished, launching plotting scripts')
    # Parallelize the plotting by dividing into chunks and utils.processes
    dss = utils.chunks_dataset(dset, utils.chunks_size)
    plot_files_param = partial(plot_files, **args)
    p = Pool(utils.processes)
    p.map(plot_files_param, dss)


def plot_files(dss, **args):
    first = True
    for time_sel in dss.time:
        data = dss.sel(time=time_sel)
        data['air_pressure_at_sea_level'].values = mpcalc.smooth_n_point(
            data['air_pressure_at_sea_level'].values, n=9, passes=10)
        time, run, cum_hour = utils.get_time_run_cum(data)
        # Build the name of the output image
        filename = utils.subfolder_images[projection] + \
            '/' + variable_name + '_%s.png' % cum_hour

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['air_temperature_2m'],
                                 extend='both',
                                 cmap=args['cmap'],
                                 levels=args['levels_t2m'])

        cs2 = args['ax'].contour(args['x'], args['y'],
                                 data['air_temperature_2m'],
                                 extend='both',
                                 levels=args['levels_t2m'][::5],
                                 linewidths=0.3,
                                 colors='gray', alpha=0.7)

        c = args['ax'].contour(args['x'], args['y'],
                               data['air_pressure_at_sea_level'],
                               levels=args['levels_mslp'],
                               colors='white', linewidths=1.)

        labels = args['ax'].clabel(
            c, c.levels, inline=True, fmt='%4.0f', fontsize=6)
        labels2 = args['ax'].clabel(
            cs2, cs2.levels, inline=True, fmt='%2.0f', fontsize=7)

        maxlabels = utils.plot_maxmin_points(args['ax'], args['x'], args['y'], data['air_pressure_at_sea_level'],
                                       'max', 170, symbol='H', color='royalblue', random=True)
        minlabels = utils.plot_maxmin_points(args['ax'], args['x'], args['y'], data['air_pressure_at_sea_level'],
                                       'min', 170, symbol='L', color='coral', random=True)

        # We need to reduce the number of points before plotting the vectors,
        # these values work pretty well
        density = 17
        cv = args['ax'].quiver(args['x'][::density, ::density],
                               args['y'][::density, ::density],
                               data['x_wind_10m'][::density, ::density],
                               data['y_wind_10m'][::density, ::density],
                               scale=None,
                               alpha=0.8, color='gray')

        an_fc = utils.annotation_forecast(args['ax'], time)
        an_var = utils.annotation(args['ax'],
                            'MSLP [hPa], Winds@10m and Temperature@2m', loc='lower left', fontsize=6)
        an_run = utils.annotation_run(args['ax'], run)


        if first:
            plt.colorbar(cs, orientation='horizontal',
                         label='Temperature [C]', pad=0.03, fraction=0.04)

        plt.savefig(filename, **utils.options_savefig)

        utils.remove_collections([cs, cs2, c, labels, labels2, an_fc,
                           an_var, an_run, cv, maxlabels, minlabels])

        first = False


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed_time = time.time()-start_time
    utils.print_message("script took " + time.strftime("%H:%M:%S",
                  time.gmtime(elapsed_time)))
