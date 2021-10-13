import os
import sys
import pandas as pd
import csv
import PySAM.ResourceTools as tools
import ssl
def download_meta_data(city):
    editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    save_path = os.path.join(sys.path[0],str(city))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # replace with key and email address from https://developer.nrel.gov/signup/
    sam_api_key = editable_data['SAM API key']
    sam_email = editable_data['your_email']
    #Location Coordinates
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    # Declare all variables as strings. Spaces must be replaced with '+', i.e., change 'John Smith' to 'John+Smith'.
    # Define the lat, long of the location and the year
    # You must request an NSRDB api key from the link above
    api_key = sam_api_key
    # Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
    attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
    # Choose year of data
    # Set leap year to true or false. True will return leap day data if present, false will not.
    leap_year = 'false'
    # Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
    interval = '60'
    # Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
    # NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
    # local time zone.
    utc = 'false'
    # Your full name, use '+' instead of spaces.
    your_name = editable_data['your_name']
    # Your reason for using the NSRDB.
    reason_for_use = editable_data['reason_for_use']
    # Your affiliation
    your_affiliation = editable_data['your_affiliation']
    # Your email address
    your_email = sam_email
    # Please join our mailing list so we can keep you up-to-date on new developments.
    mailing_list = editable_data['mailing_list']
    if mailing_list=='yes':
        mailing_list='true'
    else:
        mailing_list='false'
    for year in range(int(editable_data['starting_year']),int(editable_data['ending_year'])+1):
        # Declare url string
        url = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context
        # Return just the first 2 lines to get metadata:
        #try:
        info = pd.read_csv(url)
        info_name =city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year)+'.csv'
        save_path = os.path.join(sys.path[0],str(city))
        info.to_csv(os.path.join(save_path,info_name), index = False)
        print('Downlaoding meteorlogical data of '+city+' in '+str(year))
        #except:
            #print('ERROR bad request: Data cannnot be downloaded from NSRDB')
            #print('Please, check values of 16 ("Longitude") to 23 ("SAM API key") rows in EditableFile.csv file')
            #sys.exit()
