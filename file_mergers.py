import numpy as np
import pandas as pd
from glob import glob # TODO: add in directory of files to loop through...
#import sys 
from collections import defaultdict
import csv
import math

class Calendar:

	def __init__(self):
		self.month_max_days = {
	        1: 31,
	        2: 28,
	        3: 31,
	        4: 30,
	        5: 31,
	        6: 30,
	        7: 31,
	        8: 31,
	        9: 30,
	        10: 31,
	        11: 30,
	        12: 31
	    }

	def max_days_in_month(self, month_num):
	    return self.month_max_days[month_num]

	def is_leap_yr(self, year_num):
		if year_num % 4 > 0:
			return False

	def max_days_yr(self):
		if self.is_leap_yr(self.year_num) == True:
			return 366
		else:
			return 365

	def split_dates(self, date):
		date_els = date.split('/')
		month = int(date_els[0])
		day = int(date_els[1])
		year = int(date_els[2])
		return day, month, year

	def create_dates(self, day, month, year, days_in_month, periodicity):
		date_array = []
		day = int(day)
		month = int(month)
		year = int(year)
		# put the first date into the array
		date = str(month) +'/'+ str(day) + '/' + str(year)
		date_array.append(date)
		# now make the remainder of the days
		for i in range(1, periodicity):
			day += 1
			if day > days_in_month:
				day = 1
				month += 1
				if month > 12:
					month = 1
					year += 1
			date = str(month) +'/'+ str(day) + '/' + str(year)
			#date = date.strip() # was just to make sure. probably don't need. 
			date_array.append(date)
		return date_array

class Source:

	def __init__(self):
		self.calendar = Calendar() # we'll use this generic calendar as a look-up helper class
		self.get_params()
		self.read_file(self.base_file, 1) # base file
		self.initial_data_drop() # i.e. drop IDs from df
		# in drop, we create self.dataframe which is our final dataframe
		self.final_columns = list(self.dataframe.columns.values)

		""" BULK OF WORK HERE  
			# LOOP THROUGH EACH MERGE file
			# READ IT, MATCH IT (AND CREATE ALT DATES AS NEEDED)
		"""
		for merge_f in self.merge_files:
			self.read_file(merge_f, 2)
			merge_periodicity = self.merge_f_dict[merge_f]
			if merge_periodicity > 1:
				# create all the alternate dates
				merge_dataframe = self.expand_raw_merge_data(merge_periodicity)
			else:
				merge_dataframe = self.raw_data_merge
			# with a "cleanish" dataframe in hand, merge the data
			self.merge_data(merge_dataframe)

			print self.dataframe.shape

		self.write_finished_file()

	def get_params(self):
		# self.base_files = [] TO DO... (faster / less-error pron than manually one-at-a-time)
		self.base_file = raw_input("BASE file:   ") # base file
		#self.base_file_period = int(raw_input("period of original file: 1 == daily, 2 == weekly, 3 == monthly, 4 == quarterly")) # base file
		
		#financials_file = raw_input("MERGE file name, NO directory:   ") # financials file (earnings, etc) 
		company_ticker = raw_input("Company Ticker:  ")
		self.financials_file = "financials/financials-"+company_ticker+".csv"

		self.merge_files = [
			'10-yr-tres.csv',
			'brent.csv',
			'civil-labor-part-rate.csv',
			'civil-unemployment-rate.csv',
			'cpi.csv',
			'fed-debt-to-GDP.csv',
			'housing-start.csv',
			'initial-claims.csv',
			'libor.csv',
			'personal-consumption-expend.csv',
			'personal-savings-rate.csv',
			'real-gdp.csv',
			'real-median-hh-income.csv',
			#'sp500.csv',
			'usd-euro.csv'
		]
		self.merge_files.append(self.financials_file)
		self.merge_f_dict = {
			# 1: daily, 7: weekly, 31: monthly, 91: quarterly, 365: annual
			# we assign the numbers like this because we're going to use them later...saves a conversion step
			'10-yr-tres.csv': 1,
			'brent.csv': 1,
			'civil-labor-part-rate.csv': 31,
			'civil-unemployment-rate.csv': 31,
			'cpi.csv':31,
			'fed-debt-to-GDP.csv':91,
			'housing-start.csv': 31,
			'initial-claims.csv': 7,
			'libor.csv':1,
			'personal-consumption-expend.csv': 31,
			'personal-savings-rate.csv': 31,
			'real-gdp.csv': 91,
			'real-median-hh-income.csv': 365,
			'sp500.csv': 1,
			'usd-euro.csv': 1,
			self.financials_file: 91
		}
		

		self.root_base = 'data/transformed/v4/' # this is where we'll version things
		self.root_merge = 'data/fundamentals/v4/'

		self.fin_file_name = raw_input("Name for Final File:    ")
		self.fin_file_name = 'data/working/v4/' + self.fin_file_name

	def read_file(self, file_name, f_type):
    	# TYPE: base (1) vs merge (2)
    	# Read data
		print "reading file data for % r" % file_name
		if f_type == 1:
			self.raw_data_base = pd.read_csv(self.root_base+file_name, dtype = str)
		elif f_type == 2:
			self.raw_data_merge = pd.read_csv(self.root_merge+file_name, dtype = str)
    
	def initial_data_drop(self):
		columns = list(self.raw_data_base.columns.values)
		if 'id' in columns:
			self.dataframe = self.raw_data_base.drop(['id'], axis=1)
		else:
			self.dataframe = self.raw_data_base
	
	def merge_data(self, merge_dataframe): 
		self.dataframe = self.dataframe.merge(merge_dataframe, how='inner', on='Date')	
		self.dataframe.drop_duplicates(subset='Date', keep='last', inplace=True)

	def expand_raw_merge_data(self, periodicity): 
		dates = self.raw_data_merge['Date']
		# two empty arrays that we'll append values to
		# then, later, make into np.arrays and then create a dataframe with
		all_dates = []
		all_date_vals = []
		for date in dates:
			#print date
			day, month, year = self.calendar.split_dates(date)  # <-- need a way to get the "previous day" so that you can fix the information leak 
			# get the original date from the merge data frame (to get its "cell" value)
			tmp_df = self.raw_data_merge.loc[self.raw_data_merge['Date'] == date] # dataframe row
			tmp_df_cols = list(tmp_df.columns.values) # column headers, also used below
			#print tmp_df_cols
			date_value = tmp_df.iloc[0][tmp_df_cols[1:]] # the value of that specific date  # <- THIS IS WHERE YOU CAN NAB THE FULL DATA SET...RATHER THAN JUST ONE COL
			#print "%r %r %r" % (day, month, year)
			days_in_month = self.calendar.max_days_in_month(month)
			if (month == 2 and self.calendar.is_leap_yr(year) == True): # annoying control for leap years
				days_in_month += 1
			new_dates = self.calendar.create_dates(day, month, year, days_in_month, periodicity)
			# new_dates is an array
			# we want to get single value entries so we have to loop through them (unfortunately)
			for nd in new_dates:
				all_date_vals.append(date_value)
				all_dates.append(nd)
		# then make a dataframe with all this stuff 
		all_dates = np.array([all_dates]) # this will make a row vector 
		# we transpose to get the column vector
		all_dates = np.transpose(all_dates)
		all_date_vals = np.array(all_date_vals)
		#print all_dates.shape
		#print all_date_vals.shape
		merged_dates_date_vals = np.hstack((all_dates, all_date_vals))
		new_merge_df = pd.DataFrame(merged_dates_date_vals, columns = tmp_df_cols)
		#print new_merge_df
		# return it to the calling function
		return new_merge_df

	def write_finished_file(self):
		final_file = self.dataframe.to_csv(self.fin_file_name,index_label='id')

if __name__ == "__main__":
	
	transformation = Source() 
