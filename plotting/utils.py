import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.colors as colors
import pandas as pd
from matplotlib.colors import from_levels_and_colors
import seaborn as sns
import os
import matplotlib.patheffects as path_effects
import matplotlib.cm as mplcm
import sys
import xarray as xr
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import metpy
from matplotlib.image import imread as read_png
import requests
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import warnings
warnings.filterwarnings(
    action='ignore',
    message='The unit of the quantity is stripped.'
)

apiKey = os.environ['MAPBOX_KEY']
apiURL_places = "https://api.mapbox.com/geocoding/v5/mapbox.places"

if 'MODEL_DATA_FOLDER' in os.environ:
    folder = os.environ['MODEL_DATA_FOLDER']
else:
    folder = '/home/ekman/ssd/guido/meps/'
folder_images = folder
chunks_size = 10
processes = 4
figsize_x = 11
figsize_y = 9

if "HOME_FOLDER" in os.environ:
    home_folder = os.environ['HOME_FOLDER']
else:
    home_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Options for savefig
options_savefig = {
    'dpi': 100,
    'bbox_inches': 'tight',
    'transparent': False
}

# Dictionary to map the output folder based on the projection employed
subfolder_images = {
    'scandinavia': folder_images,
    'nord': folder_images + "/nord",
}

proj_defs = {
    'scandinavia':
    {
        'projection': 'lcc',
        'lat_0': 63.3,
        'lon_0': 15,
        'lat_1': 63.3,
        'lat_2': 63.3,
        'rsphere': 6371000.0,
        'resolution': 'i',
        'llcrnrlon': 2,
        'llcrnrlat': 52.5,
        'urcrnrlon': 49,
        'urcrnrlat': 71.5
    },
    'nord':
    {
        'projection': 'lcc',
        'lat_0': 63.3,
        'lon_0': 15,
        'lat_1': 63.3,
        'lat_2': 63.3,
        'rsphere': 6371000.0,
        'resolution': 'i',
        'llcrnrlon': 11.39,
        'llcrnrlat': 64.5,
        'urcrnrlon': 36.5,
        'urcrnrlat': 71.8
    },
}


def get_run():
    date = pd.to_datetime('now', utc=True)
    hour = date.hour

    if (hour >= 3 ) & (hour < 6):
        run = "00"
    elif (hour >= 6 ) & (hour < 9):
        run = "03"
    elif (hour >= 9 ) & (hour < 12):
        run = "06"
    elif (hour >= 12 ) & (hour < 15):
        run = "09"
    elif (hour >= 15 ) & (hour < 18):
        run = "12"
    elif (hour >= 18 ) & (hour < 21):
        run = "15"
    elif (hour >= 21 ) & (hour <= 23):
        run = "18"
    elif (hour > 0 ) & (hour < 3):
        date = (pd.to_datetime('now', utc=True) - pd.Timedelta('1 day'))
        run = "21"

    return f"{date.strftime('%Y%m%d')}T{run}Z"


def read_dataset(variables=['air_temperature_2m'], level=None, projection=None,
                 engine='netcdf4'):
    """Wrapper to initialize the dataset"""
    # Get Run
    run = os.environ.get('DATE_RUN', get_run())
    dset = xr.open_dataset(f"https://thredds.met.no/thredds/dodsC/mepslatest/meps_det_2_5km_{run}.ncml",
                           engine=engine)
    dset = dset[variables].squeeze()
    dset = dset.metpy.parse_cf()
    if level:
        dset = dset.sel(pressure=level, method='nearest')
    if projection != 'scandinavia':
        # This may be slow as it needs to load the data
        proj_options = proj_defs[projection]
        mask = ((dset['longitude'] <= proj_options['urcrnrlon']) &
                (dset['longitude'] >= proj_options['llcrnrlon']) &
                (dset['latitude'] <= proj_options['urcrnrlat']) &
                (dset['latitude'] >= proj_options['llcrnrlat']))
        dset = dset.where(mask, drop=True)

    dset['run'] = dset['time'].isel(time=0).to_pandas()

    # chunk now based on the dimension of the dataset after the subsetting
    # dset = dset.chunk({'time': round(len(dset.time) / 10),
    #                    'latitude': round(len(dset.latitude) / 4),
    #                    'longitude': round(len(dset.longitude) / 4)})

    return dset


def get_time_run_cum(dset):
    time = dset['time'].to_pandas()
    run = dset['run'].to_pandas()
    cum_hour = np.array((time - run) / pd.Timedelta('1 hour')).astype(int)

    return time, run, cum_hour


def print_message(message):
    """Formatted print"""
    print(os.path.basename(sys.argv[0])+' : '+message)


def get_city_coordinates(city):
    # First read the local cache and see if we already downloaded the city coordinates
    if os.path.isfile(home_folder + '/plotting/cities_coordinates.csv'):
        cities_coords = pd.read_csv(home_folder + '/plotting/cities_coordinates.csv',
                                    index_col=[0])
        if city in cities_coords.index:
            return cities_coords.loc[city].lon, cities_coords.loc[city].lat
        else:
            # make the request and append to the file
            url = "%s/%s.json?&access_token=%s" % (apiURL_places, city, apiKey)
            response = requests.get(url)
            json_data = json.loads(response.text)
            lon, lat = json_data['features'][0]['center']
            to_append = pd.DataFrame(index=[city],
                                     data={'lon': lon, 'lat': lat})
            to_append.to_csv(home_folder + '/plotting/cities_coordinates.csv',
                             mode='a', header=False)

            return lon, lat
    else:
        # Make request and create the file for the first time
        url = "%s/%s.json?&access_token=%s" % (apiURL_places, city, apiKey)
        response = requests.get(url)
        json_data = json.loads(response.text)
        lon, lat = json_data['features'][0]['center']
        cities_coords = pd.DataFrame(index=[city],
                                     data={'lon': lon, 'lat': lat})
        cities_coords.to_csv(home_folder + '/plotting/cities_coordinates.csv')

        return lon, lat


def get_projection(dset, projection="de", countries=True, regions=True, labels=False, color_borders='black'):
    from mpl_toolkits.basemap import Basemap
    proj_options = proj_defs[projection]
    m = Basemap(**proj_options)

    m.drawcoastlines(linewidth=0.5, linestyle='solid',
                     color=color_borders, zorder=7)
    if countries:
        m.drawcountries(linewidth=1, linestyle='solid',
                        color=color_borders, zorder=7)

    x, y = m(dset['longitude'], dset['latitude'])

    return (m, x, y)


def plot_background_mapbox(m, xpixels=800):
    ypixels = round(m.aspect * xpixels)
    bbox = '[%s,%s,%s,%s]' % (m.llcrnrlon, m.llcrnrlat,
                              m.urcrnrlon, m.urcrnrlat)
    url = 'https://api.mapbox.com/styles/v1/mapbox/dark-v10/static/%s/%sx%s?access_token=%s&logo=false' % (
        bbox, xpixels, ypixels, apiKey)

    img = plt.imread(url)

    m.imshow(img, origin='upper')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def chunks_dataset(ds, n):
    """Same as 'chunks' but for the time dimension in
    a dataset"""
    for i in range(0, len(ds.time), n):
        yield ds.isel(time=slice(i, i + n))


# Annotation run, models
def annotation_run(ax, time, loc='upper right', fontsize=8):
    """Put annotation of the run obtaining it from the
    time array passed to the function."""
    time = pd.to_datetime(time)
    at = AnchoredText('MEPS Run %s' % time.strftime('%Y%m%d %H UTC'),
                      prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)
    return (at)


def annotation_forecast(ax, time, loc='upper left', fontsize=8, local=True):
    """Put annotation of the forecast time."""
    time = pd.to_datetime(time)
    if local:  # convert to local time
        time = convert_timezone(time)
        at = AnchoredText('Valid %s' % time.strftime('%A %d %b %Y at %H:%M (Berlin)'),
                          prop=dict(size=fontsize), frameon=True, loc=loc)
    else:
        at = AnchoredText('Forecast for %s' % time.strftime('%A %d %b %Y at %H:%M UTC'),
                          prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)
    return (at)


def add_logo_on_map(ax=home_folder+'/plotting/meteoindiretta_logo.png', zoom=0.15, pos=(0.92, 0.1)):
    '''Add a logo on the map given a pnd image, a zoom and a position
    relative to the axis ax.'''
    img_logo = OffsetImage(read_png(logo), zoom=zoom)
    logo_ann = AnnotationBbox(
        img_logo, pos, xycoords='axes fraction', frameon=False)
    logo_ann.set_zorder(10)
    at = ax.add_artist(logo_ann)
    return at


def convert_timezone(dt_from, from_tz='utc', to_tz='Europe/Berlin'):
    """Convert between two timezones. dt_from needs to be a Timestamp 
    object, don't know if it works otherwise."""
    dt_to = dt_from.tz_localize(from_tz).tz_convert(to_tz)
    # remove again the timezone information

    return dt_to.tz_localize(None)


def annotation(ax, text, loc='upper right', fontsize=8):
    """Put a general annotation in the plot."""
    at = AnchoredText('%s' % text, prop=dict(
        size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)

    return (at)


def annotation_forecast_radar(ax, time, loc='upper left', fontsize=8, local=True):
    """Put annotation of the forecast time."""
    if local:  # convert to local time
        time = convert_timezone(time)
        at = AnchoredText('Valid %s' % time.strftime('%A %d %b %Y at %H:%M (Berlin)'),
                          prop=dict(size=fontsize), frameon=True, loc=loc)
    else:
        at = AnchoredText('Valid %s' % time.strftime('%A %d %b %Y at %H:%M UTC'),
                          prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)

    return (at)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Truncate a colormap by specifying the start and endpoint."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return (new_cmap)


def get_colormap(cmap_type):
    """Create a custom colormap."""
    colors_tuple = pd.read_csv(
        home_folder + '/plotting/cmap_%s.rgba' % cmap_type).values

    cmap = colors.LinearSegmentedColormap.from_list(
        cmap_type, colors_tuple, colors_tuple.shape[0])
    return (cmap)


def get_colormap_norm(cmap_type, levels):
    """Create a custom colormap."""
    if cmap_type == "rain":
        cmap, norm = from_levels_and_colors(levels, sns.color_palette("Blues", n_colors=len(levels)),
                                            extend='max')
    elif cmap_type == "snow":
        cmap, norm = from_levels_and_colors(levels, sns.color_palette("PuRd", n_colors=len(levels)),
                                            extend='max')
    elif cmap_type == "snow_discrete":
        colors = ["#DBF069", "#5AE463", "#E3BE45", "#65F8CA", "#32B8EB",
                  "#1D64DE", "#E97BE4", "#F4F476", "#E78340", "#D73782", "#702072"]
        cmap, norm = from_levels_and_colors(levels, colors, extend='max')
    elif cmap_type == "rain_acc":
        cmap, norm = from_levels_and_colors(levels, sns.color_palette('gist_stern_r', n_colors=len(levels)),
                                            extend='max')
    elif cmap_type == "rain_new":
        colors_tuple = pd.read_csv(
            home_folder + '/plotting/cmap_prec.rgba').values
        cmap, norm = from_levels_and_colors(levels, sns.color_palette(colors_tuple, n_colors=len(levels)),
                                            extend='max')
    elif cmap_type == "winds":
        colors_tuple = pd.read_csv(
            home_folder + '/plotting/cmap_winds.rgba').values
        cmap, norm = from_levels_and_colors(levels, sns.color_palette(colors_tuple, n_colors=len(levels)),
                                            extend='max')
    elif cmap_type == "rain_acc_wxcharts":
        colors_tuple = pd.read_csv(
            home_folder + '/plotting/cmap_rain_acc_wxcharts.rgba').values
        cmap, norm = from_levels_and_colors(levels, sns.color_palette(colors_tuple, n_colors=len(levels)),
                                            extend='max')
    elif cmap_type == "snow_wxcharts":
        colors_tuple = pd.read_csv(
            home_folder + '/plotting/cmap_snow_wxcharts.rgba').values
        cmap, norm = from_levels_and_colors(levels, sns.color_palette(colors_tuple, n_colors=len(levels)),
                                            extend='max')
    elif cmap_type == "cape_wxcharts":
        colors_tuple = pd.read_csv(
            home_folder + '/plotting/cmap_cape_wxcharts.rgba').values
        cmap, norm = from_levels_and_colors(levels, sns.color_palette(colors_tuple, n_colors=len(levels)),
                                            extend='max')
    elif cmap_type == "winds_wxcharts":
        colors_tuple = pd.read_csv(
            home_folder + '/plotting/cmap_winds_wxcharts.rgba').values
        cmap, norm = from_levels_and_colors(levels, sns.color_palette(colors_tuple, n_colors=len(levels)),
                                            extend='max')

    return (cmap, norm)


def remove_collections(elements):
    """Remove the collections of an artist to clear the plot without
    touching the background, which can then be used afterwards."""
    for element in elements:
        try:
            for coll in element.collections:
                coll.remove()
        except AttributeError:
            try:
                for coll in element:
                    coll.remove()
            except ValueError:
                print_message('WARNING: Element is empty')
            except TypeError:
                element.remove()
        except ValueError:
            print_message('WARNING: Collection is empty')


def plot_maxmin_points(ax, lon, lat, data, extrema, nsize, symbol, color='k',
                       random=False):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True)
    """
    from scipy.ndimage.filters import maximum_filter, minimum_filter

    # We have to first add some random noise to the field, otherwise it will find many maxima
    # close to each other. This is not the best solution, though...
    if random:
        data = np.random.normal(data, 0.2)

    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)
    # Filter out points on the border
    mxx, mxy = mxx[(mxy != 0) & (mxx != 0)], mxy[(mxy != 0) & (mxx != 0)]

    texts = []
    for i in range(len(mxy)):
        texts.append(ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], symbol, color=color, size=15,
                             clip_on=True, horizontalalignment='center', verticalalignment='center',
                             path_effects=[path_effects.withStroke(linewidth=1, foreground="black")], zorder=8))
        texts.append(ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], '\n' + str(data[mxy[i], mxx[i]].astype('int')),
                             color="gray", size=10, clip_on=True, fontweight='bold',
                             horizontalalignment='center', verticalalignment='top',
                             zorder=8))
    return (texts)


def add_vals_on_map(ax, bmap, var, levels, density=50,
                    cmap='rainbow', norm=None, shift_x=0., shift_y=0., fontsize=7.5, lcolors=True):
    '''Given an input projection, a variable containing the values and a plot put
    the values on a map exlcuing NaNs and taking care of not going
    outside of the map boundaries, which can happen.
    - shift_x and shift_y apply a shifting offset to all text labels
    - colors indicate whether the colorscale cmap should be used to map the values of the array'''

    if norm is None:
        norm = colors.Normalize(vmin=np.min(levels), vmax=np.max(levels))

    m = mplcm.ScalarMappable(norm=norm, cmap=cmap)

    # Remove values outside of the extents
    var = var[::density, ::density]
    lons = var.longitude.values
    lats = var.latitude.values

    at = []
    for ilat, ilon in np.ndindex(var.shape):
        if not var[ilat, ilon].isnull():
            if lcolors:
                at.append(ax.annotate(('%d' % var[ilat, ilon]), bmap(lons[ilat, ilon] + shift_x, lats[ilat, ilon] + shift_y),
                                      color=m.to_rgba(float(var[ilat, ilon])), weight='bold', fontsize=fontsize,
                                      path_effects=[path_effects.withStroke(linewidth=1, foreground="white")], zorder=5))
            else:
                at.append(ax.annotate(('%d' % var[ilat, ilon]), bmap(lons[ilat, ilon] + shift_x, lats[ilat, ilon] + shift_y),
                                      color='white', weight='bold', fontsize=fontsize,
                                      path_effects=[path_effects.withStroke(linewidth=1, foreground="white")], zorder=5))

    return at

def divide_axis_for_cbar(ax, width="45%", height="2%", pad=-2, adjust=0.03):
    '''Using inset_axes, divides axis in two to place the colorbars side to side.
    Note that we use the bbox explicitlly with padding to adjust the position of the colorbars
    otherwise they'll come out of the axis (don't really know why)'''
    ax_cbar = inset_axes(ax,
                         width=width,
                         height=height,
                         loc='lower left',
                         borderpad=pad,
                         bbox_to_anchor=(adjust, 0., 1, 1),
                         bbox_transform=ax.transAxes
                         )
    ax_cbar_2 = inset_axes(ax,
                           width=width,
                           height=height,
                           loc='lower right',
                           borderpad=pad,
                           bbox_to_anchor=(-adjust, 0., 1, 1),
                           bbox_transform=ax.transAxes
                           )

    return ax_cbar, ax_cbar_2
