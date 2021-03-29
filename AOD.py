### This file contains fuctions to read AOD data from MOD08 and MYD08 datasets for given region and generate time averaged maps, difference maps and metrics

from netCDF4 import Dataset
import os
import pyresample
import numpy as np
from matplotlib import pyplot as plt
import datetime
import numpy.ma as ma
from extras import get_mask
import pandas as pd
import shapefile as shp


plt.rcParams.update({'xtick.labelsize': 16})
plt.rcParams.update({'ytick.labelsize': 16})
plt.rcParams.update({'font.size': 16})


### This function reads AOD data from MOD08_D3 from Terra & MYD08_D3 from Aqua 
### between start_date and end_date for the given years and for the given region coordinates(x,y)
### Inputs:
###     f_path: path to save daily AOD mean values for the given region and dates in csv format.
###     path:  MOD08_D3/MYD08_D3 data location
###     start_date: Date (in python datetime format) from which data needs to be considered. 
###                The year in this data will not be considered. The years in the input parameter "years" will be considered
###     end_date:  Date (in python datetime format) upto which data needs to be considered. 
###                The year in this data will not be considered. The years in the input parameter "years" will be considered
###     years: List of years to be considered for processing data
###     (x,y): longitude and latitude pairs for the required region
###     con: suffix to store the csv file (eg: "India", "IGP", "Global")
### Outputs: 
###     AOD_mean_<start_date>_<end_date>_<con>.csv: daily AOD mean values for he given region from start_date to end_date in csv format in f_path location
###     AOD_mean_all: returns Average pixel wise AOD values for the given region (each pixel represents AOD mean value for all the dates within date range given) 
###     daten: dates for which data is processed. 
def get_AOD_years(f_path,path,start_date,end_date,years,x,y,con):
	files=os.listdir(path);
	date1=[]
	AOD_list=[]
	AOD_mean=[]
	daten=[]
	my_lats,my_lons,mask,swath = get_mask(x,y)
	for file in files:
		if not ( '._' in file ):
			#print(file)
			year=file[10:14]
			#print(year)
			days=file[14:17]
			#print(days)
			#print(year,days)
			date=datetime.datetime(int(year),1,1)+datetime.timedelta(days=int(days))
			#print(date)
			start_daten=datetime.datetime(date.year,start_date.month,start_date.day)
			end_daten=datetime.datetime(date.year,end_date.month,end_date.day)
			if ((date.year in years) and (date>=start_daten) and (date<=end_daten)):
				#print(file)
				fh = Dataset(path+'\\'+file, mode='r');
				#print(file)	
				AOD = fh.variables['Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean'][:]
				#print(AOD)
				lon=fh.variables['XDim'][:]
				lat=fh.variables['YDim'][:]
				date1.append(date)
				daten.append(datetime.datetime.strftime(date,'%d.%m.%y'))
				AOD.data[AOD.mask==True]=np.nan
				lon_grid, lat_grid = np.meshgrid(lon, lat)
				grid = pyresample.geometry.GridDefinition(lats=lat_grid, lons=lon_grid)
				AOD_n = pyresample.kd_tree.resample_nearest(source_geo_def=grid, target_geo_def=swath, data=AOD.data,radius_of_influence=100000)
				AOD_n = ma.masked_array(AOD_n, mask=mask)
				AOD_list.append(AOD_n)
				AOD_mean.append(np.nanmean(AOD_n))

	AOD_df = pd.DataFrame()
	AOD_df['date']=date1
	AOD_df['AOD_mean']=AOD_mean
	#print(AOD_n)
	AOD_df.to_csv(f_path+'\\AOD_mean_'+daten[0] + '_' + daten[-1]+'_'+con+'.csv');
	
	AOD_mean_all=np.nanmean(AOD_list,axis=0)
	AOD_mean_all = ma.masked_array(AOD_mean_all, mask=mask)
    
	

	return AOD_mean_all,daten


### This function reads AOD data from MOD08_D3 from Terra & MYD08_D3 from Aqua 
### between start_date and end_date for the given years and for the given point coordinates
### Inputs:
###     f_path: path to save daily AOD mean values for the given region and dates in csv format.
###     path:  MOD08_D3/MYD08_D3 data location
###     start_date: Date (in python datetime format) from which data needs to be considered. 
###                The year in this data will not be considered. The years in the input parameter "years" will be considered
###     end_date:  Date (in python datetime format) upto which data needs to be considered. 
###                The year in this data will not be considered. The years in the input parameter "years" will be considered
###     years: List of years to be considered for processing data
###     coordinate_file: file containing point coordinates
###     sufix: suffix to store the csv file (eg: "global", "Indian")
### Outputs: 
###     AOD_mean_<start_date>_<end_date>_<suffix>_cities.csv: daily AOD mean values for he given region from start_date to end_date in csv format in f_path location
def get_AOD_by_coordinates(f_path,path,start_date,end_date,years,coordinate_file,suffix):
	files=os.listdir(path);
	date1=[]
	AOD_list=[]
	AOD_mean=[]
	daten=[]
	#my_lats,my_lons,mask,swath = get_mask(x,y)
	df = pd.read_csv(coordinate_file)
	my_lats=df['Lat'].values
	my_lons=df['Long'].values
	#my_lon_grid, my_lat_grid = np.meshgrid(my_lons, my_lats)
	swath = pyresample.geometry.SwathDefinition(lons=my_lons, lats=my_lats)
	df_out=pd.DataFrame()
	for file in files:
		if not ( '._' in file ):
			year=file[10:14]
			days=file[14:17]
			date=datetime.datetime(int(year),1,1)+datetime.timedelta(days=int(days))
			start_daten=datetime.datetime(date.year,start_date.month,start_date.day)
			end_daten=datetime.datetime(date.year,end_date.month,end_date.day)
			if ((date.year in years) and (date>=start_daten) and (date<=end_daten)):
				val = {}
				fh = Dataset(path+'\\'+file, mode='r');
				#print(file)	
				date1.append(date)
				daten.append(datetime.datetime.strftime(date,'%d.%m.%y'))
				AOD = fh.variables['Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean'][:]
				lon=fh.variables['XDim'][:]
				lat=fh.variables['YDim'][:]
				date1.append(date)
				daten.append(datetime.datetime.strftime(date,'%d-%m'))
				AOD.data[AOD.mask==True]=np.nan
				lon_grid, lat_grid = np.meshgrid(lon, lat)
				grid = pyresample.geometry.GridDefinition(lats=lat_grid, lons=lon_grid)
				AOD_n = pyresample.kd_tree.resample_nearest(source_geo_def=grid, target_geo_def=swath, data=AOD.data,radius_of_influence=120000)
				#AOD_n = ma.masked_array(AOD_n, mask=mask)
				#AOD_list.append(AOD_n)
				#AOD_mean.append(np.nanmean(AOD_n))
				val['date']=datetime.datetime.strftime(date,'%d.%m.%y')
				for i in range(0,len(df)):
					val[df['Cities'][i]]=AOD_n[i]
				#AOD_n = ma.masked_array(AOD_n, mask=mask)
				AOD_list.append(val)
				#AOD_mean.append(np.nanmean(AOD_n))


	df_out=pd.DataFrame(AOD_list)
	#AOD_df = pd.DataFrame()
	#AOD_df['date']=date1
	#AOD_df['AOD_mean']=AOD_mean
	
	df_out.to_csv(f_path+'\\AOD_mean_'+daten[0] + '_' + daten[-1]+'_'+suffix+'_cities.csv');



## function to change colormap to suit postitive and negative differences so that the zero point always comes in the middle.
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

orig_cmap = matplotlib.cm.RdBu.reversed()
cmap2 = matplotlib.cm.jet


def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]



##genreates mask based on longitudes and latidutes
# X = longitude which can be obtained from shapefile
# Y = latitude which can be obtained from shapefile
def get_mask(x,y):
	#my_lats = np.linspace(np.floor(np.min(y))-0.5, np.ceil(np.max(y))+0.5, int((np.max(y)-np.min(y)))+3)
	my_lats = np.arange(np.floor(np.min(y))-0.5, np.ceil(np.max(y))+0.5, 1)
	#my_lons = np.linspace(np.floor(np.min(x))-0.5, np.ceil(np.max(x))+0.5, int((np.max(x)-np.min(x)))+3)
	my_lons = np.arange(np.floor(np.min(x))-0.5, np.ceil(np.max(x))+0.5, 1)
	#my_lats = np.linspace(np.floor(np.min(y)), np.ceil(np.max(y)), int(4*(np.max(y)-np.min(y))))
	#my_lons = np.linspace(np.floor(np.min(x)), np.ceil(np.max(x)), int(4*(np.max(x)-np.min(x))))
	x_1, y_1 = np.meshgrid(my_lons, my_lats)
	x_1, y_1 = x_1.flatten(), y_1.flatten()
	points = np.vstack((x_1,y_1)).T 
	polygon=np.vstack((x,y)).T
	p = Path(polygon)
	grid = p.contains_points(points)
	mask = grid.reshape(len(my_lons),len(my_lats)) 
	#mask = grid.reshape(int(4*(np.max(y)-np.min(y))),int(4*(np.max(x)-np.min(x)))) 
	mask = ~mask
	my_lon_grid, my_lat_grid = np.meshgrid(my_lons, my_lats)
	swath = pyresample.geometry.SwathDefinition(lons=my_lon_grid, lats=my_lat_grid)
	#print(my_lons)
	#print(my_lats)
	return(my_lats,my_lons,mask,swath)



#### This function generates time averaged maps and different maps for the given region using values in var_1 and var_2 matrices.
### Inputs:
###     f_path: path to save outputs.
###     prefix: name that is used to save the output files
###     var_1: AOD mean values for the given region 
###     var_2: AOD mean values for the given region
###     (x,y): longitude and latitude pairs for the required region
###     con: suffix to store the csv file (eg: "India", "IGP", "Global")
###     date_1: date matrix for var_1
###     date_2: date matrix for var_2
def compare_values(f_path,prefix,var_1,var_2,date_1,date_2,x,y,con):

	my_lats,my_lons,mask,swath = get_mask(x,y)
	fig = plt.figure()
	plt.plot(x,y,color="black")
	plt.title("Time Averaged " + prefix +  " from "+ date_1[0] + ' to ' + date_1[-1] + " (" + '%.2f' % np.nanmean(var_1) +")" ,fontsize=8)
	plt.imshow(var_1, aspect='auto', interpolation='none',
           extent=extents(my_lons) + extents(my_lats), origin='lower',cmap=cmap2)
	plt.colorbar()
	plt.xlabel('Longitude (E)')
	plt.ylabel('Latitude (N)')
	plt.axes().set_aspect('equal')
	plt.tight_layout()
	plt.clim(0,1.4)
	#plt.clim(np.nanmin([np.nanmin(var_1),np.nanmin(var_2)]),np.nanmax([np.nanmax(var_1),np.nanmax(var_2)]))
	plt.savefig(f_path+"\\TimeAveraged"+prefix+"_"+date_1[0] + '_' + date_1[-1] + '_' + con + '.png')
	plt.show(block=False)
	plt.close()

	fig = plt.figure()
	plt.plot(x,y,color="black")
	plt.title("Time Averaged " + prefix +  " from "+ date_2[0] + ' to ' + date_2[-1] + " (" + '%.2f' % np.nanmean(var_2) +")" ,fontsize=7)
	plt.xlabel('Longitude (E)')
	plt.ylabel('Latitude (N)')
	plt.imshow(var_2, aspect='auto', interpolation='none',
           extent=extents(my_lons) + extents(my_lats), origin='lower',cmap=cmap2)
	plt.colorbar()
	plt.axes().set_aspect('equal')
	plt.tight_layout()
	plt.clim(0,1.4)
	#plt.clim(np.nanmin([np.nanmin(var_1),np.nanmin(var_2)]),np.nanmax([np.nanmax(var_1),np.nanmax(var_2)]))
	plt.savefig(f_path+"\\TimeAveraged"+prefix+"_"+date_2[0] + '_' + date_2[-1] + '_' + con + '.png')
	plt.show(block=False)
	plt.close()
	
	fig = plt.figure()
	val=var_2-var_1
	midpoint = 1-(np.nanmax(val)/(np.nanmax(val)-np.nanmin(val)))

	plt.plot(x,y,color="black")
	plt.title("Change in mean " + prefix +  " between "+ date_2[0] + '_' + date_2[-1]+" and "+date_1[0] + '_' + date_1[-1]+" (" + '%.2f' % np.nanmean(val) +")" ,fontsize=7 )
	plt.xlabel('Longitude (E)')
	plt.ylabel('Latitude (N)')
	if (np.nanmin(val)<=0) & (np.nanmax(val)>=0):
		shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name='shifted')
	else:
		if (np.nanmin(val)<0) & (np.nanmax(val)<0):
			shifted_cmap = matplotlib.cm.Blues
		else:
			shifted_cmap = matplotlib.cm.Reds

	plt.imshow(val, aspect='auto', interpolation='none',
           extent=extents(my_lons) + extents(my_lats), origin='lower',cmap=shifted_cmap)
	plt.colorbar()
	plt.axes().set_aspect('equal')
	plt.tight_layout()
	#plt.clim(1*pow(10,18),3.5*pow(10,18))
	plt.savefig(f_path+"\\DifferenceOfMean"+prefix+"_"+date_2[0] + '_' + date_2[-1]+"and"+date_1[0] + '_' + date_1[-1]+'_' + con + '.png')
	plt.show(block=False)
	plt.close()


#### This function compares matrices var_1 and var_2 and generates mean, min, max and standard deviation values of two matrices; 
#### mean, min, max and standard deviation of difference matric of these metrices; percentage of pixels greater than 1 std from difference and
#### percentage of pixels less than 1 std from difference matrix.
### Inputs:
###     var_1: matrix 1
###     var_2: matrix 2
### Outputs:
###     metrix1: mean, min, max and standard deviation of var_1 matrix
###     metrix2: mean, min, max and standard deviation of var_2 matrix
###     metrix: mean, min, max and standard deviation of (var_2 - var_1) matrix
###     pos_val: percentage of pixels greater than 1 std from difference matrix.
###     neg_val: percentage of pixels less than 1 std from difference matrix.
def calculate_metrix(var_1,var_2):

	val=var_2-var_1
	metrix1=[np.nanmean(var_1),np.nanmin(var_1),np.nanmax(var_1),np.nanstd(var_1)]
	metrix2=[np.nanmean(var_2),np.nanmin(var_2),np.nanmax(var_2),np.nanstd(var_2)]
	metrix=[np.nanmean(val),np.nanmin(val),np.nanmax(val),np.nanstd(val)]
	#print((np.size(val)-np.sum(np.isnan(val))-np.sum(val.mask)),np.sum(val>=0),np.sum(val<0))
	#print(val)
	pos_val = np.nansum(val>=np.nanstd(val))/(np.size(val)-np.sum(np.isnan(val))-np.sum(val.mask))*100
	neg_val = np.nansum(val<=-np.nanstd(val))/(np.size(val)-np.sum(np.isnan(val))-np.sum(val.mask))*100
	#pos_val = np.nansum(val>=0)/(np.size(val)-np.sum(np.isnan(val))-np.sum(val.mask))*100
	#neg_val = np.nansum(val<0)/(np.size(val)-np.sum(np.isnan(val))-np.sum(val.mask))*100
	return(metrix1,metrix2,metrix, pos_val, neg_val)



#### This function generates time averaged maps and different maps for global region using values in var_1 and var_2 matrices.
### Inputs:
###     f_path: path to save outputs.
###     prefix: name that is used to save the output files
###     var_1: AOD mean values for the given region 
###     var_2: AOD mean values for the given region
###     date_1: date matrix for var_1
###     date_2: date matrix for var_2
def compare_values_global(f_path,prefix,var_1,var_2,date_1,date_2):
	sf = shp.Reader("F:\\Mahesh\\mopitt\\world_final\\world_final\\world_final.shp")
	my_lats = np.linspace(-90, 90, 720)
	my_lons = np.linspace(-180, 180, 1440)
	#my_lats,my_lons,mask,swath = get_mask(x,y)
	fig = plt.figure()
	for shape in sf.shapeRecords():
		x=[i[0] for i in shape.shape.points[:]]
		y = [i[1] for i in shape.shape.points[:]]
		plt.plot(x,y,color="black",linewidth=2)
	plt.title("Time Averaged " + prefix +  " from "+ date_1[0] + ' to ' + date_1[-1],fontsize=8)
	plt.imshow(var_1, aspect='auto', interpolation='none',
           extent=extents(my_lons) + extents(my_lats), origin='lower',cmap=cmap2)
	plt.colorbar()
	plt.xlabel('Longitude (E)')
	plt.ylabel('Latitude (N)')
	plt.axes().set_aspect('equal')
	#plt.tight_layout()
	plt.clim(np.nanmin([np.nanmin(var_1),np.nanmin(var_2)]),np.nanmax([np.nanmax(var_1),np.nanmax(var_2)]))
	plt.savefig(f_path+"\\TimeAveraged"+prefix+"_"+date_1[0] + '_' + date_1[-1] + '_global' + '.pdf')
	plt.show(block=False)
	plt.close()

	fig = plt.figure()
	for shape in sf.shapeRecords():
		x=[i[0] for i in shape.shape.points[:]]
		y = [i[1] for i in shape.shape.points[:]]
		plt.plot(x,y,color="black",linewidth=2)
	#plt.plot(x,y,color="black")
	plt.title("Time Averaged " + prefix +  " from "+ date_2[0] + ' to ' + date_2[-1] ,fontsize=8)
	plt.xlabel('Longitude (E)')
	plt.ylabel('Latitude (N)')
	plt.imshow(var_2, aspect='auto', interpolation='none',
           extent=extents(my_lons) + extents(my_lats), origin='lower',cmap=cmap2)
	plt.colorbar()
	plt.axes().set_aspect('equal')
	#plt.tight_layout()
	plt.clim(np.nanmin([np.nanmin(var_1),np.nanmin(var_2)]),np.nanmax([np.nanmax(var_1),np.nanmax(var_2)]))
	plt.savefig(f_path+"\\TimeAveraged"+prefix+"_"+date_2[0] + '_' + date_2[-1] + '_global'  + '.pdf')
	plt.show(block=False)
	plt.close()
	
	fig = plt.figure()
	val=var_2-var_1
	midpoint = 1-(np.nanmax(val)/(np.nanmax(val)-np.nanmin(val)))
	#print(midpoint)
	for shape in sf.shapeRecords():
		x=[i[0] for i in shape.shape.points[:]]
		y = [i[1] for i in shape.shape.points[:]]
		plt.plot(x,y,color="black",linewidth=2)
	plt.title("Change in mean " + prefix +  " between "+ date_2[0] + '_' + date_2[-1]+" and "+date_1[0] + '_' + date_1[-1],fontsize=7 )
	plt.xlabel('Longitude (E)')
	plt.ylabel('Latitude (N)')

	shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name='shifted')
	plt.imshow(val, aspect='auto', interpolation='none',
           extent=extents(my_lons) + extents(my_lats), origin='lower',cmap=shifted_cmap)
	plt.colorbar()
	plt.axes().set_aspect('equal')
	#plt.tight_layout()
	#plt.clim(1*pow(10,18),3.5*pow(10,18))
	plt.savefig(f_path+"\\DifferenceOfMean"+prefix+"_"+date_2[0] + '_' + date_2[-1]+"and"+date_1[0] + '_' + date_1[-1]+'_global' + '.pdf')
	plt.show(block=False)
	plt.close()
	
	# pos=var_2-var_1
	# pos[pos<=0]=np.nan
	# fig = plt.figure()
	