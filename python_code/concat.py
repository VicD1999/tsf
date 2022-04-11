# Script to concat 2 netcdf files
# Author: Victor Dachet

from scipy.io import netcdf

netCDF_file_path = "data/MAR/concat.nc"
netCDF_file_path2 = "data/MAR/concat2.nc"

new_netcdf = {}
new_netcdf2 = {}

units = {}

with netcdf.netcdf_file(netCDF_file_path, 'r') as f:
    print(f.variables)
    
    for name, var in f.variables.items():
        new_netcdf[name] = var[:].copy()
    
    f.close()

        
with netcdf.netcdf_file(netCDF_file_path2, 'r') as f2:
    print(f2.variables)
    
    for name, var in f2.variables.items():
        new_netcdf2[name] = var[:].copy()
        units[name] = var.units
        
    f2.close()


print([(key, new_netcdf[key].shape) for key in new_netcdf])
print([new_netcdf2[key].shape for key in new_netcdf2])
print([units[key] for key in units])


with netcdf.netcdf_file("data/MAR/concat_20210331_20210430.nc", 'w') as f:
    f.createDimension('TIME', new_netcdf['TIME'].shape[0] + new_netcdf2['TIME1'].shape[0] )
    f.createDimension('X21_100', 80)
    f.createDimension('Y21_70', 50)
    f.createDimension('ZULEV', 2)
    
    time = f.createVariable('TIME', 'int32', ('TIME',))
    time[:new_netcdf['TIME'].shape[0]] = new_netcdf['TIME']
    time[new_netcdf['TIME'].shape[0]:] = new_netcdf2['TIME1']
    time.units = b'minutes since 2016-09-01 00:00:00'
    
    UUZ = f.createVariable("UUZ", "float", ('TIME', 'ZULEV', 'Y21_70', 'X21_100',) )
    
    UUZ[:new_netcdf['TIME'].shape[0], :, :, :] = new_netcdf['UUZ']
    UUZ[new_netcdf['TIME'].shape[0]:, :, :, :] = new_netcdf2['UUZ']
    
    VVZ = f.createVariable("VVZ", "float", ('TIME', 'ZULEV', 'Y21_70', 'X21_100',) )
    
    VVZ[:new_netcdf['TIME'].shape[0], :, :, :] = new_netcdf['VVZ']
    VVZ[new_netcdf['TIME'].shape[0]:, :, :, :] = new_netcdf2['VVZ']
    
    VVZ.units = b'm/s'
    
    f.close()

