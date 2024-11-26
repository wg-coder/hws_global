#%%
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from datetime import datetime
import pandas as pd
import pickle
import pymannkendall as mk

from sklearn.linear_model import LinearRegression


# plot libaray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.io.shapereader import Reader
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# import cmaps
# import seaborn as sns
import matplotlib.transforms as mtransforms
from matplotlib.ticker import ScalarFormatter

# 定义拟合线性方程，获取趋势和p值的函数
def mk_trend_ve(x):
    if np.isnan(x).sum() > 35:
        return (np.nan ,np.nan)
    else :
        mk_result = mk.original_test(x)
        slope = mk_result.slope
        p = mk_result.p
        return (slope ,p)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = '15'

cf_coastial_land_200km = cfeature.ShapelyFeature(
    Reader(r'/home/zq2/wg/code/global_landfalling_heatwaves/shp/coastial_land_100km_final.shp').geometries(),
    ccrs.PlateCarree(),
    edgecolor='gray',
    facecolor='none'
)

def add_map_feature(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='#626063')
    # ax.add_feature(cf_coastial_land_200km)
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-60, 90, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent([-180, 180, -65, 90], crs=ccrs.PlateCarree())
    ax.axes.set_xlabel('')
    ax.axes.set_ylabel('')


def xarray_weighted_mean(data_array):

    weights = np.cos(np.deg2rad(data_array.lat))
    weights.name = "weights"
    data_weighted = data_array.weighted(weights)
    weighted_mean = data_weighted.mean(("lon", "lat"))

    return weighted_mean



# FIG S land hws metrics map (完成 12.26 15:04)
fig = plt.figure(figsize=(20,18))
grid = plt.GridSpec(18,18, wspace=0.12, hspace = 0.2)

# set sub figure
ax1 = fig.add_subplot(grid[0:5,0:8],projection = ccrs.PlateCarree())
ax2 = fig.add_subplot(grid[0:5,9:17],projection = ccrs.PlateCarree())
ax3 = fig.add_subplot(grid[6:11,0:8],projection = ccrs.PlateCarree())
ax4 = fig.add_subplot(grid[6:11,9:17],projection = ccrs.PlateCarree())
ax5 = fig.add_subplot(grid[12:17,0:8],projection = ccrs.PlateCarree())


# ax1
ax1.cla()
ds_land_hws['exposure'].mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax1,levels=np.arange(0,11,1),cmap='MPL_OrRd',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days per year)')
)
add_map_feature(ax1)
fig.show()
# ax2
ax2.cla()
ds_land_hws['frequency'].mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax2,levels=np.arange(0,4.2,0.2),cmap='MPL_OrRd',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Frequency (events per year)',ticks=np.arange(0,4.2,1))
)
add_map_feature(ax2)

# ax3
ax3.cla()
ds_land_hws['extent'].mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax3,levels=np.arange(0,3200000,200000),cmap='MPL_OrRd',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Extent (km$^2$)')
)
add_map_feature(ax3)

# ax4
ax4.cla()
ds_land_hws['cum_heat_grid'].mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax4,levels=np.arange(0,11,1),cmap='MPL_OrRd',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Cumulative Heat (°C)')
)
add_map_feature(ax4)

# ax5
ax5.cla()
(ds_land_hws['cum_heat_grid'].mean(dim='time',skipna=True).where(da_land_mask, drop=True)/\
ds_land_hws['exposure'].mean(dim='time',skipna=True).where(da_land_mask, drop=True)).plot(
    ax=ax5,levels=np.arange(0,2.2,0.2),cmap='MPL_OrRd',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Intensity (°C/day)')
)
add_map_feature(ax5)

ax1.text(0,1.03,'(a)',transform=ax1.transAxes,fontsize=22)
ax2.text(0,1.03,'(b)',transform=ax2.transAxes,fontsize=22)
ax3.text(0,1.03,'(c)',transform=ax3.transAxes,fontsize=22)
ax4.text(0,1.03,'(d)',transform=ax4.transAxes,fontsize=22)
ax5.text(0,1.03,'(e)',transform=ax5.transAxes,fontsize=22)


ax1.text(0.02,0.04,'Mean:2.77',transform=ax1.transAxes,fontsize=22)
ax2.text(0.02,0.04,'Mean:1.41',transform=ax2.transAxes,fontsize=22)
ax3.text(0.02,0.04,'Mean:1.09',transform=ax3.transAxes,fontsize=22)
ax4.text(0.02,0.04,'Mean:2.30',transform=ax4.transAxes,fontsize=22)
ax5.text(0.02,0.04,'Mean:0.83',transform=ax5.transAxes,fontsize=22)
fig.show()

fig.savefig(fig_savepath+'figS_land_hws_metrics_map.png',dpi=400)

# FIG S all hws metrics map (完成 12.24 22:44)
fig = plt.figure(figsize=(18,12))
grid = plt.GridSpec(12,18, wspace=0.12, hspace = 0.1)

# set sub figure
ax1 = fig.add_subplot(grid[0:5,0:8],projection = ccrs.PlateCarree())
ax2 = fig.add_subplot(grid[0:5,9:17],projection = ccrs.PlateCarree())
ax3 = fig.add_subplot(grid[6:11,0:8],projection = ccrs.PlateCarree())
ax4 = fig.add_subplot(grid[6:11,9:17],projection = ccrs.PlateCarree())


# ax1
ax1.cla()
ds_all_hws['exposure'].mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax1,levels=np.arange(0,11,1),cmap='MPL_OrRd',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days per year)')
)
add_map_feature(ax1)
fig.show()
# ax2
ax2.cla()
ds_all_hws['frequency'].mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax2,levels=np.arange(0,4.2,0.2),cmap='MPL_Oranges',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Frequency (events per year)',ticks=np.arange(0,4.2,1))
)
add_map_feature(ax2)

# ax3
ax3.cla()
ds_all_hws['cum_heat_grid'].mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax3,levels=np.arange(0,11,1),cmap='cmocean_matter',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Cumulative Heat (°C)')
)
add_map_feature(ax3)

# ax4
ax4.cla()
(ds_all_hws['cum_heat_grid'].mean(dim='time',skipna=True).where(da_land_mask, drop=True)/\
ds_all_hws['exposure'].mean(dim='time',skipna=True).where(da_land_mask, drop=True)).plot(
    ax=ax4,levels=np.arange(0,2.2,0.2),cmap='cmocean_matter',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Intensity (°C/day)')
)
add_map_feature(ax4)

ax1.text(0,1.03,'(a)',transform=ax1.transAxes,fontsize=22)
ax2.text(0,1.03,'(b)',transform=ax2.transAxes,fontsize=22)
ax3.text(0,1.03,'(c)',transform=ax3.transAxes,fontsize=22)
ax4.text(0,1.03,'(d)',transform=ax4.transAxes,fontsize=22)


ax1.text(0.02,0.04,'Mean:6.83',transform=ax1.transAxes,fontsize=22)
ax2.text(0.02,0.04,'Mean:3.14',transform=ax2.transAxes,fontsize=22)
ax3.text(0.02,0.04,'Mean:6.49',transform=ax3.transAxes,fontsize=22)
ax4.text(0.02,0.04,'Mean:0.93',transform=ax4.transAxes,fontsize=22)
fig.show()

fig.savefig(fig_savepath+'figS_all_hws_metrics_map.png',dpi=400)


# Observed ratio of metrics of landfalling HWs to all HWs （完成 12.24 22:41）
hws_metrics = 'frequency'
da_hws_frequency_ratio = ds_ocean_onto_land_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')/(
    ds_land_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')+
    ds_land_onto_ocean_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')+
    ds_miscellaneous_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')+
    ds_ocean_onto_land_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')
)
hws_metrics = 'exposure'
da_hws_exposure_ratio = ds_ocean_onto_land_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')/(
    ds_land_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')+
    ds_land_onto_ocean_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')+
    ds_miscellaneous_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')+
    ds_ocean_onto_land_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')
)
hws_metrics = 'cum_heat_grid'
da_hws_cum_intensity_ratio = ds_ocean_onto_land_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')/(
    ds_land_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')+
    ds_land_onto_ocean_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')+
    ds_miscellaneous_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')+
    ds_ocean_onto_land_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')
)

fig = plt.figure(figsize=(18,12))
grid = plt.GridSpec(12,18, wspace=0.12, hspace = 0.1)

# set sub figure
ax1 = fig.add_subplot(grid[0:5,0:8],projection = ccrs.PlateCarree())
ax2 = fig.add_subplot(grid[0:5,9:17],projection = ccrs.PlateCarree())
ax3 = fig.add_subplot(grid[6:11,0:8],projection = ccrs.PlateCarree())
# ax4 = fig.add_subplot(grid[6:11,9:17])
ax4 = fig.add_axes([0.52,0.2,0.34,0.26])

ax1.cla()
(da_hws_frequency_ratio*100).plot(
    ax=ax1,levels=np.arange(0,55,5),cmap='PiYG_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Ratio (%)')
)
add_map_feature(ax1)

ax2.cla()
(da_hws_exposure_ratio*100).plot(
    ax=ax2,levels=np.arange(0,55,5),cmap='PiYG_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Ratio (%)')
)
add_map_feature(ax2)

ax3.cla()
(da_hws_cum_intensity_ratio*100).plot(
    ax=ax3,levels=np.arange(0,55,5),cmap='PiYG_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Ratio (%)')
)
add_map_feature(ax3)

# ax4 增加沿海100km，200km的比例平均与全球平均
# xarray_weighted_mean((da_hws_cum_intensity_ratio*100).where(da_coastial_land_100km_mask,drop=True))
list_ratio_100km = [40.38,42.86,43.53]
list_ratio_200km = [35.96,37.89,87.06]
list_ratio_all = [21.06,22.30,48.07]
x = np.arange(len(list_ratio_100km))
ax4.cla()
ax4.bar(x - 0.25, list_ratio_100km, width=0.25, label='Coastal land within 100 km of the coast')
ax4.bar(x, list_ratio_200km, width=0.25, label='Coastal land within 200 km of the coast')
ax4.bar(x + 0.25, list_ratio_all, width=0.25, label='Global land')
# 添加X轴标题
ax4.set_xticks(x, ['Frequency', 'Exposure', 'Cumulative Heat'])
ax4.set_ylim(0,100)
# 添加图例
ax4.legend(frameon=False)
ax4.set_ylabel('Ratio (%)')

ax1.text(0,1.03,'(a) Frequency',transform=ax1.transAxes,fontsize=22)
ax2.text(0,1.03,'(b) Exposure',transform=ax2.transAxes,fontsize=22)
ax3.text(0,1.03,'(c) Cumulative Heat',transform=ax3.transAxes,fontsize=22)

ax1.text(0,1.03,'(a) Frequency',transform=ax1.transAxes,fontsize=22)
ax2.text(0,1.03,'(b) Exposure',transform=ax2.transAxes,fontsize=22)
ax3.text(0,1.03,'(c) Cumulative Heat',transform=ax3.transAxes,fontsize=22)

fig.show()
fig.savefig(fig_savepath + 'FigS_Observed_ratio_of_metrics_of_landfalling_HWs_to_all_HWs.png',dpi=400)


#
# bounds = [0,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,1]
# cmap = plt.get_cmap('RdBu_r')
# norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
#
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
# c = ax.contourf(lons, lats, da_cum_heat_grid_ratio, extend='neither', levels=bounds,cmap=cmap, norm=norm,transform = ccrs.PlateCarree())
# #levels=np.arange(0.0000001,1.1,0.1)
# ax.add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
# # ax.add_feature(cf_coastial_land_200km)
# ax.set_xticks(np.arange(-180,181,60), crs=ccrs.PlateCarree())
# ax.set_yticks(np.arange(-60,91,30), crs=ccrs.PlateCarree())
# ax.xaxis.set_major_formatter(LongitudeFormatter())
# ax.yaxis.set_major_formatter(LatitudeFormatter())
# ax.set_extent([-180,180,-60,90],crs=ccrs.PlateCarree())
# cb_pos = fig.add_axes([0.2,0.1,0.6,0.03])
# # cb = plt.colorbar(c,shrink=0.7)
# cb = plt.colorbar(c,cax=cb_pos,orientation='horizontal',ticks=bounds)
# fig.show()

# Oceanic observed ratio of exposure/frequency of landfalling HWs to all HWs (完成 12.25 00:40)
# 这是在海洋上空的！
hws_metrics = 'frequency'
da_hws_frequency_ratio_ocean = ds_ocean_onto_land_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')/(
    ds_land_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')+
    ds_land_onto_ocean_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')+
    ds_miscellaneous_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')+
    ds_ocean_onto_land_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')
)
hws_metrics = 'exposure'
da_hws_exposure_ratio_ocean = ds_ocean_onto_land_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')/(
    ds_land_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')+
    ds_land_onto_ocean_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')+
    ds_miscellaneous_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')+
    ds_ocean_onto_land_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')
)
hws_metrics = 'cum_heat_grid'
da_hws_cum_intensity_ratio_ocean = ds_ocean_onto_land_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')/(
    ds_land_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')+
    ds_land_onto_ocean_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')+
    ds_miscellaneous_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')+
    ds_ocean_onto_land_hws[hws_metrics].where(da_sea_mask, drop=True).sum(dim='time')
)

fig = plt.figure(figsize=(18,12))
grid = plt.GridSpec(12,18, wspace=0.12, hspace = 0.1)

# set sub figure
ax1 = fig.add_subplot(grid[0:5,0:8],projection = ccrs.PlateCarree())
ax2 = fig.add_subplot(grid[0:5,9:17],projection = ccrs.PlateCarree())
ax3 = fig.add_subplot(grid[6:11,0:8],projection = ccrs.PlateCarree())
# ax4 = fig.add_subplot(grid[6:11,9:17],projection = ccrs.PlateCarree())

ax1.cla()
da_hws_frequency_ratio_ocean.plot(
    ax=ax1,levels=np.arange(0,1.05,0.05),cmap='PiYG_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Ratio (%)',ticks=np.arange(0,1.05,0.2))
)
add_map_feature(ax1)

ax2.cla()
da_hws_exposure_ratio_ocean.plot(
    ax=ax2,levels=np.arange(0,1.05,0.05),cmap='PiYG_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Ratio (%)',ticks=np.arange(0,1.05,0.2))
)
add_map_feature(ax2)

ax3.cla()
da_hws_cum_intensity_ratio_ocean.plot(
    ax=ax3,levels=np.arange(0,1.05,0.05),cmap='PiYG_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Ratio (%)',ticks=np.arange(0,1.05,0.2))
)
add_map_feature(ax3)

ax1.text(0,1.03,'(a) Frequency',transform=ax1.transAxes,fontsize=22)
ax2.text(0,1.03,'(b) Exposure',transform=ax2.transAxes,fontsize=22)
ax3.text(0,1.03,'(c) Cumulative Heat',transform=ax3.transAxes,fontsize=22)

ax1.text(0.68,1.03,'Mean: 75.91%',transform=ax1.transAxes,fontsize=22)
ax2.text(0.68,1.03,'Mean: 78.76%',transform=ax2.transAxes,fontsize=22)
ax3.text(0.68,1.03,'Mean: 79.44%',transform=ax3.transAxes,fontsize=22)

fig.show()
fig.savefig(fig_savepath + 'FigS_Oceanic_observed_ratio_of_metrics_of_landfalling_HWs_to_all_HWs.png',dpi=400)

# landfalling hws metrics map （完成 12.24 22:50）
fig = plt.figure(figsize=(18,12))
grid = plt.GridSpec(12,18, wspace=0.12, hspace = 0.1)

# set sub figure
ax1 = fig.add_subplot(grid[0:5,0:8],projection = ccrs.PlateCarree())
ax2 = fig.add_subplot(grid[0:5,9:17],projection = ccrs.PlateCarree())
ax3 = fig.add_subplot(grid[6:11,0:8],projection = ccrs.PlateCarree())
ax4 = fig.add_subplot(grid[6:11,9:17],projection = ccrs.PlateCarree())
# ax1
ax1.cla()
ds_ocean_onto_land_hws['frequency'].mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax1,levels=np.arange(0,2.2,0.2),cmap='MPL_Oranges',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Frequency (events per year)')
)
add_map_feature(ax1)

# ax2
ax2.cla()
ds_ocean_onto_land_hws['extent'].mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax2,levels=np.arange(0,4000000,200000),cmap='MPL_copper_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Extent (km$^2$)')
)
add_map_feature(ax2)

# ax3
ax3.cla()
ds_ocean_onto_land_hws['cum_heat_grid'].mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax3,levels=np.arange(0,8.1,0.8),cmap='cmocean_matter',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Cumulative Heat (°C)')
)
add_map_feature(ax3)

# ax4
ax4.cla()
(ds_ocean_onto_land_hws['cum_heat_grid'].mean(dim='time',skipna=True).where(da_land_mask, drop=True)/\
ds_ocean_onto_land_hws['exposure'].mean(dim='time',skipna=True).where(da_land_mask, drop=True)).plot(
    ax=ax4,levels=np.arange(0,2.2,0.2),cmap='cmocean_matter',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Intensity (°C/day)')
)
add_map_feature(ax4)

ax1.text(0,1.03,'(a)',transform=ax1.transAxes,fontsize=22)
ax2.text(0,1.03,'(b)',transform=ax2.transAxes,fontsize=22)
ax3.text(0,1.03,'(c)',transform=ax3.transAxes,fontsize=22)
ax4.text(0,1.03,'(d)',transform=ax4.transAxes,fontsize=22)

ax1.text(0.02,0.04,'Mean:0.66',transform=ax1.transAxes,fontsize=22)
ax2.text(0.02,0.04,'Mean:4.07',transform=ax2.transAxes,fontsize=22)
ax3.text(0.02,0.04,'Mean:1.79',transform=ax3.transAxes,fontsize=22)
ax4.text(0.02,0.04,'Mean:1.11',transform=ax4.transAxes,fontsize=22)
fig.show()
fig.savefig(fig_savepath + 'FigS_landfalling_hws_metrics_map.png',dpi=400)

# landfalling exposure 不同年份演变图 （完成 12.24 23:05）
fig = plt.figure(figsize=(18,12))
grid = plt.GridSpec(12,18, wspace=0.12, hspace = 0.1)

# set sub figure
ax1 = fig.add_subplot(grid[0:5,0:8],projection = ccrs.PlateCarree())
ax2 = fig.add_subplot(grid[0:5,9:17],projection = ccrs.PlateCarree())
ax3 = fig.add_subplot(grid[6:11,0:8],projection = ccrs.PlateCarree())
ax4 = fig.add_subplot(grid[6:11,9:17],projection = ccrs.PlateCarree())

ax1.cla()
ds_ocean_onto_land_hws['exposure'].sel(time=slice(1979,1990)).mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax1,levels=np.arange(0,6.1,0.5),cmap='RdBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days per year)')
)
add_map_feature(ax1)

ax2.cla()
ds_ocean_onto_land_hws['exposure'].sel(time=slice(1991,2000)).mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax2,levels=np.arange(0,6.1,0.5),cmap='RdBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days per year)')
)
add_map_feature(ax2)

ax3.cla()
ds_ocean_onto_land_hws['exposure'].sel(time=slice(2001,2010)).mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax3,levels=np.arange(0,6.1,0.5),cmap='RdBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days per year)')
)
add_map_feature(ax3)

ax4.cla()
ds_ocean_onto_land_hws['exposure'].sel(time=slice(2011,2020)).mean(dim='time',skipna=True).where(da_land_mask, drop=True).plot(
    ax=ax4,levels=np.arange(0,6.1,0.5),cmap='RdBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days per year)')
)
add_map_feature(ax4)

ax1.text(0,1.03,'(a) 1979-1990',transform=ax1.transAxes,fontsize=22)
ax2.text(0,1.03,'(b) 1991-2000',transform=ax2.transAxes,fontsize=22)
ax3.text(0,1.03,'(c) 2001-2010',transform=ax3.transAxes,fontsize=22)
ax4.text(0,1.03,'(d) 2011-2020',transform=ax4.transAxes,fontsize=22)

ax1.text(0.02,0.04,'Mean:0.56',transform=ax1.transAxes,fontsize=22)
ax2.text(0.02,0.04,'Mean:1.01',transform=ax2.transAxes,fontsize=22)
ax3.text(0.02,0.04,'Mean:1.24',transform=ax3.transAxes,fontsize=22)
ax4.text(0.02,0.04,'Mean:3.27',transform=ax4.transAxes,fontsize=22)

fig.show()
fig.savefig(fig_savepath + 'FigS_landfalling_exposure_four_timeslice_map.png',dpi=400)

# Trend of landfalling hws （完成 12.26 11:53）
data_landfalling_exposure_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_ocean_onto_land_hws['exposure'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_landfalling_frequency_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_ocean_onto_land_hws['frequency'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_landfalling_extent_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_ocean_onto_land_hws['extent'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_landfalling_cum_heat_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_ocean_onto_land_hws['cum_heat_grid'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_landfalling_mean_intensity_trend = xr.apply_ufunc(
    mk_trend_ve,
    (ds_ocean_onto_land_hws['cum_heat_grid'].where(da_land_mask, drop=True) / \
     ds_ocean_onto_land_hws['exposure'].where(da_land_mask, drop=True)).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)

fig = plt.figure(figsize=(20,18))
grid = plt.GridSpec(18,18, wspace=0.12, hspace = 0.2)

# set sub figure
ax1 = fig.add_subplot(grid[0:5,0:8],projection = ccrs.PlateCarree())
ax2 = fig.add_subplot(grid[0:5,9:17],projection = ccrs.PlateCarree())
ax3 = fig.add_subplot(grid[6:11,0:8],projection = ccrs.PlateCarree())
ax4 = fig.add_subplot(grid[6:11,9:17],projection = ccrs.PlateCarree())
ax5 = fig.add_subplot(grid[12:17,0:8],projection = ccrs.PlateCarree())

# ax1
ax1.cla()
(data_landfalling_exposure_trend[0]*10).plot(
    ax=ax1,extend='both', levels=np.arange(-1,1.1,0.1),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days/10yr)',ticks=np.arange(-1,1.1,0.4))
)
add_map_feature(ax1)

# ax2
ax2.cla()
(data_landfalling_frequency_trend[0]*10).plot(
    ax=ax2,levels=np.arange(-0.5,0.55,0.05),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Frequency (events/10yr)',ticks=np.arange(-0.4,0.44,0.2))
)
add_map_feature(ax2)

# ax3
ax3.cla()
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))
(data_landfalling_extent_trend[0]*10).plot(
    ax=ax3,levels=np.arange(-1000000,1100000,100000),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Extent (km$^2$/10yr)',format=formatter)
)
add_map_feature(ax3)

# ax4
ax4.cla()
(data_landfalling_cum_heat_trend[0]*10).plot(
    ax=ax4,levels=np.arange(-1,1.1,0.1),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Cumulative Heat (°C/10yr)',ticks=np.arange(-1,1.1,0.4))
)
add_map_feature(ax4)

ax5.cla()
(data_landfalling_mean_intensity_trend[0]*10).plot(
    ax=ax5,levels=np.arange(-0.5,0.55,0.05),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Intensity (°C*day$^-1$/10yr)',ticks=np.arange(-0.4,0.44,0.2))
)
add_map_feature(ax5)

ax1.text(0,1.03,'(a)',transform=ax1.transAxes,fontsize=22)
ax2.text(0,1.03,'(b)',transform=ax2.transAxes,fontsize=22)
ax3.text(0,1.03,'(c)',transform=ax3.transAxes,fontsize=22)
ax4.text(0,1.03,'(d)',transform=ax4.transAxes,fontsize=22)
ax5.text(0,1.03,'(e)',transform=ax5.transAxes,fontsize=22)

ax1.text(0.02,0.04,'Mean:0.63*',transform=ax1.transAxes,fontsize=22)
ax2.text(0.02,0.04,'Mean:0.25*',transform=ax2.transAxes,fontsize=22)
ax3.text(0.02,0.04,'Mean:5.78*',transform=ax3.transAxes,fontsize=22)
ax4.text(0.02,0.04,'Mean:0.54*',transform=ax4.transAxes,fontsize=22)
ax5.text(0.02,0.04,'Mean:0.00',transform=ax5.transAxes,fontsize=22)
fig.show()
fig.savefig(fig_savepath+'FigS_Trend_of_landfalling_hws.png',dpi=400)

# Trend of land hws （完成 12.16 11:52）
data_land_exposure_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_land_hws['exposure'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_land_frequency_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_land_hws['frequency'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_land_extent_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_land_hws['extent'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_land_extend_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_land_hws['extent'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_land_cum_heat_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_land_hws['cum_heat_grid'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_land_mean_intensity_trend = xr.apply_ufunc(
    mk_trend_ve,
    (ds_land_hws['cum_heat_grid'].where(da_land_mask, drop=True) / \
     ds_land_hws['exposure'].where(da_land_mask, drop=True)).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)

fig = plt.figure(figsize=(20,18))
grid = plt.GridSpec(18,18, wspace=0.12, hspace = 0.2)

# set sub figure
ax1 = fig.add_subplot(grid[0:5,0:8],projection = ccrs.PlateCarree())
ax2 = fig.add_subplot(grid[0:5,9:17],projection = ccrs.PlateCarree())
ax3 = fig.add_subplot(grid[6:11,0:8],projection = ccrs.PlateCarree())
ax4 = fig.add_subplot(grid[6:11,9:17],projection = ccrs.PlateCarree())
ax5 = fig.add_subplot(grid[12:17,0:8],projection = ccrs.PlateCarree())

# ax1
ax1.cla()
(data_land_exposure_trend[0]*10).plot(
    ax=ax1,extend='both', levels=np.arange(-1,1.1,0.1),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days/10yr)',ticks=np.arange(-1,1.1,0.4))
)
add_map_feature(ax1)

# ax2
ax2.cla()
(data_land_frequency_trend[0]*10).plot(
    ax=ax2,levels=np.arange(-0.5,0.55,0.05),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Frequency (events/10yr)',ticks=np.arange(-0.4,0.44,0.2))
)
add_map_feature(ax2)

# ax3
ax3.cla()
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))
(data_land_extent_trend[0]*10).plot(
    ax=ax3,levels=np.arange(-1000000,1100000,100000),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Extent (km$^2$/10yr)',format=formatter)
)
add_map_feature(ax3)

# ax4
ax4.cla()
(data_land_cum_heat_trend[0]*10).plot(
    ax=ax4,levels=np.arange(-1,1.1,0.1),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Cumulative Heat (°C/10yr)',ticks=np.arange(-1,1.1,0.4))
)
add_map_feature(ax4)

ax5.cla()
(data_land_mean_intensity_trend[0]*10).plot(
    ax=ax5,levels=np.arange(-0.5,0.55,0.05),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Intensity (°C*day$^-1$/10yr)',ticks=np.arange(-0.4,0.44,0.2))
)
add_map_feature(ax5)

ax1.text(0,1.03,'(a)',transform=ax1.transAxes,fontsize=22)
ax2.text(0,1.03,'(b)',transform=ax2.transAxes,fontsize=22)
ax3.text(0,1.03,'(c)',transform=ax3.transAxes,fontsize=22)
ax4.text(0,1.03,'(d)',transform=ax4.transAxes,fontsize=22)
ax5.text(0,1.03,'(e)',transform=ax5.transAxes,fontsize=22)

ax1.text(0.02,0.04,'Mean:0.52*',transform=ax1.transAxes,fontsize=22)
ax2.text(0.02,0.04,'Mean:0.25*',transform=ax2.transAxes,fontsize=22)
ax3.text(0.02,0.04,'Mean:1.22*',transform=ax3.transAxes,fontsize=22)
ax4.text(0.02,0.04,'Mean:0.33*',transform=ax4.transAxes,fontsize=22)
ax5.text(0.02,0.04,'Mean:-0.01*',transform=ax5.transAxes,fontsize=22)
fig.show()
fig.savefig(fig_savepath+'FigS_Trend_of_land_hws.png',dpi=400)


# Trend of all hws （完成 12.26 11:15）
data_all_exposure_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_all_hws['exposure'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_all_frequency_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_all_hws['frequency'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_all_extent_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_all_hws['extent'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
# data_all_extend_trend = xr.apply_ufunc(
#     mk_trend_ve,
#     ds_all_hws['extent'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
#     input_core_dims = [['time']],
#     output_core_dims = [[],[]],
#     vectorize=True
# )
data_all_cum_heat_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_all_hws['cum_heat_grid'].where(da_land_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_all_mean_intensity_trend = xr.apply_ufunc(
    mk_trend_ve,
    (ds_all_hws['cum_heat_grid'].where(da_land_mask, drop=True) / \
     ds_all_hws['exposure'].where(da_land_mask, drop=True)).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)

fig = plt.figure(figsize=(18,12))
grid = plt.GridSpec(12,18, wspace=0.12, hspace = 0.1)

# set sub figure
ax1 = fig.add_subplot(grid[0:5,0:8],projection = ccrs.PlateCarree())
ax2 = fig.add_subplot(grid[0:5,9:17],projection = ccrs.PlateCarree())
ax3 = fig.add_subplot(grid[6:11,0:8],projection = ccrs.PlateCarree())
# ax4 = fig.add_subplot(grid[6:11,9:17],projection = ccrs.PlateCarree())

# ax1
ax1.cla()
(data_all_exposure_trend[0]*10).plot(
    ax=ax1,extend='both', levels=np.arange(-1,1.1,0.1),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Exposure (days/10yr)',ticks=np.arange(-1,1.1,0.4))
)
add_map_feature(ax1)

# ax2
ax2.cla()
(data_all_frequency_trend[0]*10).plot(
    ax=ax2,levels=np.arange(-0.5,0.55,0.05),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Frequency (events/10yr)',ticks=np.arange(-0.4,0.44,0.2))
)
add_map_feature(ax2)

# # ax3
# ax3.cla()
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((0, 0))
# (data_all_extent_trend[0]*10).plot(
#     ax=ax3,levels=np.arange(-1000000,1100000,100000),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
#     cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Extent (km$^2$/10yr)',format=formatter)
# )
# add_map_feature(ax3)

# ax4
ax3.cla()
(data_all_cum_heat_trend[0]*10).plot(
    ax=ax3,levels=np.arange(-1,1.1,0.1),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Cumulative Heat (°C/10yr)',ticks=np.arange(-1,1.1,0.4))
)
add_map_feature(ax3)

ax4.cla()
(data_all_mean_intensity_trend[0]*10).plot(
    ax=ax4,levels=np.arange(-0.5,0.55,0.05),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.6,orientation='horizontal',label='Intensity (°C*day$^-1$/10yr)',ticks=np.arange(-0.4,0.44,0.2))
)
add_map_feature(ax4)

ax1.text(0,1.03,'(a)',transform=ax1.transAxes,fontsize=22)
ax2.text(0,1.03,'(b)',transform=ax2.transAxes,fontsize=22)
ax3.text(0,1.03,'(c)',transform=ax3.transAxes,fontsize=22)
ax4.text(0,1.03,'(d)',transform=ax4.transAxes,fontsize=22)
# ax5.text(0,1.03,'(e)',transform=ax5.transAxes,fontsize=22)

ax1.text(0.02,0.04,'Mean:2.03*',transform=ax1.transAxes,fontsize=22)
ax2.text(0.02,0.04,'Mean:0.98*',transform=ax2.transAxes,fontsize=22)
ax3.text(0.02,0.04,'Mean:2.03*',transform=ax3.transAxes,fontsize=22)
ax4.text(0.02,0.04,'Mean:0.01',transform=ax4.transAxes,fontsize=22)
# ax5.text(0.02,0.04,'Mean:0.02',transform=ax5.transAxes,fontsize=22)
fig.show()
fig.savefig(fig_savepath+'FigS_Trend_of_all_hws.png',dpi=400)




#

fig = plt.figure(figsize=(18,16))
grid = plt.GridSpec(10,14, wspace=0.8, hspace = 0.3)

#proj = ccrs.LambertConformal(central_longitude=105,standard_parallels=(25,47))
proj = ccrs.PlateCarree()

# set sub figure
sub1 = fig.add_subplot(grid[0:2,0:5])
sub2 = fig.add_subplot(grid[0:2,5:9])
sub3 = fig.add_subplot(grid[0:2,9:14])
sub4 = fig.add_subplot(grid[3:6,0:7],projection = proj)
sub5 = fig.add_subplot(grid[3:6,7:14],projection = proj)
sub6 = fig.add_subplot(grid[7:10,0:7],projection = proj)
sub7 = fig.add_subplot(grid[7:10,7:14],projection = proj)

# landfalling HWs exposure ratio
bounds = [0,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,1]
cmap = plt.get_cmap('PiYG_r')
norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
c = ax.contourf(lons, lats, da_exposure_ratio, extend='neither', levels=bounds,cmap=cmap, norm=norm,transform = ccrs.PlateCarree())
#levels=np.arange(0.0000001,1.1,0.1)
ax.add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
# ax.add_feature(cf_coastial_land_200km)
ax.set_xticks(np.arange(-180,181,60), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-60,91,30), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_extent([-180,180,-60,90],crs=ccrs.PlateCarree())
cb_pos = fig.add_axes([0.2,0.1,0.6,0.03])
# cb = plt.colorbar(c,shrink=0.7)
cb = plt.colorbar(c,cax=cb_pos,orientation='horizontal',ticks=bounds)
fig.show()

#cum
bounds = [0,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,1]
cmap = plt.get_cmap('RdBu_r')
norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
c = ax.contourf(lons, lats, da_cum_heat_grid_ratio, extend='neither', levels=bounds,cmap=cmap, norm=norm,transform = ccrs.PlateCarree())
#levels=np.arange(0.0000001,1.1,0.1)
ax.add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
# ax.add_feature(cf_coastial_land_200km)
ax.set_xticks(np.arange(-180,181,60), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-60,91,30), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_extent([-180,180,-60,90],crs=ccrs.PlateCarree())
cb_pos = fig.add_axes([0.2,0.1,0.6,0.03])
# cb = plt.colorbar(c,shrink=0.7)
cb = plt.colorbar(c,cax=cb_pos,orientation='horizontal',ticks=bounds)
fig.show()

# trend
hws_metrics = 'cum_heat_grid'
data_lm_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_ocean_onto_land_hws[hws_metrics].where(da_land_mask, drop=True),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
c = ax.contourf(lons, lats, data_lm_trend[0].data*10,
                extend='both', levels=np.arange(-4,4.5,0.5),
                cmap='RdYlBu_r', transform = ccrs.PlateCarree())
#levels=np.arange(0.0000001,1.1,0.1)
ax.add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
# ax.add_feature(cf_coastial_land_200km)
ax.set_xticks(np.arange(-180,181,60), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-60,91,30), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_extent([-180,180,-60,90],crs=ccrs.PlateCarree())
cb_pos = fig.add_axes([0.2,0.1,0.6,0.03])
# cb = plt.colorbar(c,shrink=0.7)
cb = plt.colorbar(c,cax=cb_pos,orientation='horizontal')
fig.show()
plt.cla()


fig.savefig('/home/zq2/wg/code/global_landfalling_heatwaves/src/plot/part1/fig1-exposure_ratio-加混合.pdf')


ds_ocean_onto_land_hws[hws_metrics].where(da_land_mask, drop=True).sum(dim='time')


fig = plt.figure(figsize=[15,16])
ax1 = fig.add_axes([0.075, 0.45, 0.7, 0.4], frameon=True,projection=ccrs.PlateCarree())
ax2 = fig.add_axes([0.075, 0.80, 0.7, 0.15], frameon=True) #图上
ax3 = fig.add_axes([0.80, 0.513, 0.15, 0.273], frameon=True) #图右
# ax4 = fig.add_axes([0.755, 0.08, 0.2, 0.35], frameon=True) #右下

bbox = ax1.get_position()

bounds = [0,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2]
cmap = plt.get_cmap('RdBu_r')
norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

ax1.cla()
c = ax1.contourf(lons, lats, da_cum_heat_grid_ratio.where(da_cum_heat_grid_ratio>0.5), extend='neither', levels=bounds,cmap=cmap, norm=norm,transform = ccrs.PlateCarree())
#levels=np.arange(0.0000001,1.1,0.1)
ax1.add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
# ax.add_feature(cf_coastial_land_200km)
ax1.set_xticks(np.arange(-180,181,60), crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(-60,91,30), crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(LongitudeFormatter())
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.set_extent([-180,180,-60,90],crs=ccrs.PlateCarree())
cb_pos = fig.add_axes([0.2,0.1,0.6,0.03])
# cb = plt.colorbar(c,shrink=0.7)
cb = plt.colorbar(c,cax=cb_pos,orientation='horizontal',ticks=bounds)
fig.show()



# 将二维数组转换为一维数组
data_1d = da_cum_heat_grid_ratio.data[~np.isnan(da_cum_heat_grid_ratio.data)].flatten()
# 绘制直方图来展示概率分布，density=True使得直方图总面积等于1
hist, bin_edges = np.histogram(data_1d, bins=20, density=True)
cdf = np.cumsum(hist * np.diff(bin_edges))
# 绘制CDF
plt.plot(bin_edges[1:], cdf, marker='.', linestyle='none')
# 设置图表标题和坐标轴标签
plt.title('Cumulative Probability Distribution')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
# 显示图形
plt.show()


# ax1
ax1.cla()
ds_ocean_onto_land_hws[hws_metrics].where(da_land_mask, drop=True).plot(
    ax=ax1,cmap=cmap, norm=norm,cbar_kwargs=dict(fraction=0.05, shrink=0.6,orientation='horizontal',label='Moisture Contribution (mm)'),
    levels = bounds)
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
# ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.2)

ax1.set_xticks(np.arange(-180, 180,30), crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(-90,90,30), crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(LongitudeFormatter())
ax1.yaxis.set_major_formatter(LatitudeFormatter())
# ax1.set_xlim(-95, 140)
# ax1.set_ylim(-20, 85)
ax1.axes.set_xlabel('')
ax1.axes.set_ylabel('')


fig.show()