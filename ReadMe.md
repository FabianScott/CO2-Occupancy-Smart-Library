This is code for implementing the methods presented in Davide Cali's 2015 Paper: 
"CO2 based occupancy detection algorithm: Experimental analysis and validation for office and residential buildings".
The goal of the paper was to verify a mass-balance (MB) approach to estimating the occupancy level of different buildings.
To get started using this repository, the data files must be of the correct format as explained in the following.
The code takes .csv files of CO2 of a format compatible with the Azure SQL set-up at DTU
Library. CO2 files require:
- Three columns; a timestamp for creation time (YYYY-MM-DDThh:mm:ss.0000000), a device ID and co2 measurement.
- Title of the file must be co2_YYYY_DD_MM.csv, prefix is a variable in code and can thus be changed 
- In the case of multiple periods on the same day, numbering the files is possible ie. co2_YYYY_DD_MM1.csv
For occupancy files:
- n_zones + 1 columns: first one contains timestamps (hh.mm.ss), rest are zones. 
- Title of the file must be N_YYYY_DD_MM.csv, prefix is a variable in code and can thus be changed 
- One empty row at the bottom of the file, stems from the method of creating the file
- Colon delimited, but this is again a parameter which can be changed
If done in Excel, the file must be exported using the ".csv (comma delimited)" option. despite its name

With correctly formatted data, there are several methods in the file 'Functions.py' to obtain parameters for the MB.
For a tour of the different methods, see 'Showcase.ipynb'. 
The main method for a quick application is 'hold_out', which takes a list of dates as input along with the timestep (dt)
and the zone volumes to perform the holdout method on each of the periods specified as dates.

