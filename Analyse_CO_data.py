

from extras_CO import compare_values_India, get_CO_years, get_CO_by_coordinates
import shapefile as shp
import datetime
import numpy.ma as ma
import pandas as pd

res=0.1
## initialize paths and locations to shapefiles
## uncomment the below lines and give proper inputs for initializing required variables


# f_path = <location to generate output maps and csv files>
# path1=<CO data path1>
# path2=<CO data path2>
# india_shapefile=<India shape file location>
# world_shapefile=<World shape file location>
# igp_shapefile = <IGP shape file location>
# northeast_shapefile=<North East shape file location>
# east_shapefile=<Eastren India shape file location>
# ci_shapefile=<Central India shape file location>




## sample code to generate time averaged maps, difference maps for Indian Region
# This can be used for any region

start_date1=datetime.datetime(2020, 3, 1);
end_date1=datetime.datetime(2020, 3, 21);
years1=[2020]


start_date2=datetime.datetime(2020, 3, 25);
end_date2=datetime.datetime(2020, 4,14);
years2=[2020]


sf = shp.Reader(india_shapefile)
shape=sf.shapeRecords()[13]
#plt.figure()
x = [i[0] for i in shape.shape.points[:]]
y = [i[1] for i in shape.shape.points[:]]
var_1,date_1=get_CO_years(f_path,path1,start_date1,end_date1,years1,x,y,'India')
var_2,date_2=get_CO_years(f_path,path2,start_date2,end_date2,years2,x,y,'India')
compare_values_India(f_path,"CO",var_1,var_2,date_1,date_2,x,y,res,'India')



## sample code for generating average CO values for known coordinates
start_date1=datetime.datetime(2020, 1, 1);
end_date1=datetime.datetime(2020, 5,3);
years1=[2020]


coordinate_file='F:\\Mahesh\\mopitt\\Indian_cities.csv'
get_CO_by_coordinates(f_path,path1,start_date1,end_date1,years1,coordinate_file,'India')

coordinate_file='F:\\Mahesh\\mopitt\\new_cities.csv'
get_CO_by_coordinates(f_path,path1,start_date1,end_date1,years1,coordinate_file,'Global')


## sample code for generating Weekly metrix for 30 weeks from January-1st
sf = shp.Reader(india_shapefile)
shape=sf.shapeRecords()[13]
#plt.figure()
x = [i[0] for i in shape.shape.points[:]]
y = [i[1] for i in shape.shape.points[:]]

data=[]
for i in range(0,32):
   start_date=datetime.datetime(2020, 1, 1)+i*datetime.timedelta(days=7);
   end_date=datetime.datetime(2020, 1, 1)+(i+1)*datetime.timedelta(days=7)-datetime.timedelta(days=1);
   years1=[2019]
   years2=[2020]
   #var_1,date_1=get_CO_years(f_path,path1,start_date,end_date,years1,x,y,'India')
   #var_2,date_2=get_CO_years(f_path,path2,start_date,end_date,years2,x,y,'India')
   try:
       var_1,date_1=get_CO_years(f_path,path1,start_date,end_date,years1,x,y,'India')
       var_2,date_2=get_CO_years(f_path,path2,start_date,end_date,years2,x,y,'India')
       compare_values_India(f_path,"CO",var_1,var_2,date_1,date_2,x,y,res,'India')
       metrix1,metrix2,metrix, pos_val, neg_val=calculate_metrix(var_1,var_2)
       print(metrix1[3])
       metrix1.extend(metrix2)
       metrix1.extend(metrix)
       metrix1.append(pos_val)
       metrix1.append(neg_val)
       metrix1.append(i+1)
       data.append(metrix1)
   except:
       print(i)
       data.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0])

df = pd.DataFrame(data)
df.to_csv(f_path+"\\CO_metrix2.csv")