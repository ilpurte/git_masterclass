Usage of analysis code:

python analysis.py sky_map.csv

The values in the calibration.csv file determine the smearing of events.
Three CSV files are actually available in the config folder: one with totally incorrect calibration, one with almost correct calibration, and one with perfect calibration.

If you want to generate a different sky_map.csv file, you can use:
python generate_sky_map.py
In principle this code should not be used during the Masterclass, we should already provide a pre-generated sky_map.csv file.
The number of atm_nu and atm_mu is hardcoded.