
## This python script is a sample code on how to use functions defined in AOD.py and extras.py to generate time average maps, difference maps, metrix using AOD data

from AOD import compare_values, compare_values_global, calculate_metrix
from AOD import get_AOD_years, get_AOD_by_coordinates
import shapefile as shp
import datetime
import numpy.ma as ma


## initialize paths and locations to shapefiles
## uncomment the below lines and give proper inputs for initializing required variables


# f_path = <location to generate output maps and csv files>
# path1=<AOD data path1>
# path2=<AOD data path2>
# india_shapefile=<India shape file location>
# world_shapefile=<World shape file location>
# igp_shapefile = <IGP shape file location>
# northeast_shapefile=<North East shape file location>
# east_shapefile=<Eastren India shape file location>
# ci_shapefile=<Central India shape file location>




## sample code to generate time averaged maps, difference maps for central India Region
# This can be used for any region

start_date1=datetime.datetime(2020, 3, 1);
end_date1=datetime.datetime(2020, 3, 21);
years1=[2020]


start_date2=datetime.datetime(2020, 3, 25);
end_date2=datetime.datetime(2020, 4, 14);
years2=[2020]


sf = shp.Reader(ci_shapefile)
shape=sf.shapeRecords()[0]
#plt.figure()
x = [i[0] for i in shape.shape.points[:]]
y = [i[1] for i in shape.shape.points[:]]
var_1,date_1=get_AOD_years(f_path,path1,start_date1,end_date1,years1,x,y,'ci')
var_2,date_2=get_AOD_years(f_path,path2,start_date2,end_date2,years2,x,y,'ci')
compare_values_india(f_path,"AOD",var_1,var_2,date_1,date_2,x,y,'ci')



## sample code for generating Weekly metrix for 30 weeks from January-1st
sf = shp.Reader(india_shapefile)
shape=sf.shapeRecords()[13]
#plt.figure()
x = [i[0] for i in shape.shape.points[:]]
y = [i[1] for i in shape.shape.points[:]]

data=[]
for i in range(0,31):
    start_date=datetime.datetime(2020, 1, 1)+i*datetime.timedelta(days=7);
    end_date=datetime.datetime(2020, 1, 1)+(i+1)*datetime.timedelta(days=7)-datetime.timedelta(days=1);
    years1=[2014,2015,2016,2017,2018,2019]
    years2=[2020]

    var_1,date_1=get_AOD_years(f_path,path1,start_date,end_date,years1,x,y,'India')
    var_2,date_2=get_AOD_years(f_path,path2,start_date,end_date,years2,x,y,'India')
    #compare_values_india(f_path,"AOD",var_1,var_2,date_1,date_2,x,y,'India')
    metrix1,metrix2,metrix, pos_val, neg_val=calculate_metrix(var_1,var_2)
    metrix1.extend(metrix2)
    metrix1.extend(metrix)
    metrix1.append(pos_val)
    metrix1.append(neg_val)
    metrix1.append(i+1)
    data.append(metrix1)
import pandas as pd
df = pd.DataFrame(data)
df.to_csv(f_path+"\\AOD_metrix_IGP.csv")


data=[]
for i in range(0,31):
    start_date=datetime.datetime(2020, 1, 1)+i*datetime.timedelta(days=7);
    end_date=datetime.datetime(2020, 1, 1)+(i+1)*datetime.timedelta(days=7)-datetime.timedelta(days=1);
    years1=[2019]
    years2=[2020]

    var_1,date_1=get_AOD_years(f_path,path1,start_date,end_date,years1,x,y,'India')
    var_2,date_2=get_AOD_years(f_path,path2,start_date,end_date,years2,x,y,'India')
    #compare_values_india(f_path,"AOD",var_1,var_2,date_1,date_2,x,y,'India')
    metrix1,metrix2,metrix, pos_val, neg_val=calculate_metrix(var_1,var_2)
    metrix1.extend(metrix2)
    metrix1.extend(metrix)
    metrix1.append(pos_val)
    metrix1.append(neg_val)
    metrix1.append(i+1)
    data.append(metrix1)
import pandas as pd
df = pd.DataFrame(data)
df.to_csv(f_path+"\\AOD_metrix_2019_2020_IGP.csv")



## sample code for generating average AOD values for known coordinates

coordinate_file='F:\\Mahesh\\mopitt\\Indian_cities.csv'
get_AOD_by_coordinates(f_path,path1,start_date1,end_date1,years1,coordinate_file,'India')

coordinate_file='F:\\Mahesh\\mopitt\\new_cities.csv'
get_AOD_by_coordinates(f_path,path1,start_date1,end_date1,years1,coordinate_file,'Global')

