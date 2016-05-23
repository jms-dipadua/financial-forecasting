import numpy as np
import pandas as pd
#from glob import glob
#import sys 
from collections import defaultdict
import csv
import math

class Source:

	def __init__(self):
		self.get_params()
		self.read_file(self.o_file) # original file
		self.initial_data_drop()
		self.prep_numeric()
		self.calc_SMA(total_periods = self.dataframe.shape[0],
			step_size = self.sma_num_periods, # less to type; reduced by one due to zero-indexing
			eval_dataframe = self.dataframe['Close'],  # get the approriate data   
			write_col_bool = True)
		self.calc_WMA(total_periods = self.dataframe.shape[0],
			step_size = self.wma_num_periods, # less to type; reduced by one due to zero-indexing
			eval_dataframe = self.dataframe['Close'],  # get the approriate data   
			write_col_bool = True)
		self.calc_CCI(step_size = 20) # for now we'll hardcode 20 in for size of period to examine
		self.calc_RSI(step_size = 14) # just for now
		self.write_finished_file()

	def get_params(self):
		self.o_file = raw_input("File to Transform:   ") # original file
		self.o_file = 'data/raw/v3/' + self.o_file # versioning happens here too
		# can even ask HOW MANY of each of these i want - for now, KISS
		self.sma_num_periods = int(raw_input("Number of Periods for SMA:   ")) # 8, for last run
		self.wma_num_periods = float(raw_input("Number of Periods for WMA:   ")) # 12, for last runs
		self.wma_weights = [] # hardcoded for now. can make this a prompt-loop in future iterations
		equi_weights = 1 / self.wma_num_periods
		self.wma_num_periods = int(self.wma_num_periods) # converting back to int 
		for i in range(1,self.wma_num_periods+1): # doesn't currently add up to 1 but that's not a true constraint
			weight = equi_weights * i
			self.wma_weights.append(weight)
		#self.rsi_num_periods = int(raw_input("Number of Periods for RSI:    "))
		self.fin_file_name = raw_input("Name for Final File:    ")
		self.fin_file_name = 'data/transformed/v3/' + self.fin_file_name

	def read_file(self, file_name):
    	# Read data
	    print "reading file data"
	    #samp1 = list(range(73,220)) # FOR TESTING
	    #data = pd.read_csv(file_name, dtype = str, usecols = samp1) # TESTING
	    self.raw_data = pd.read_csv(file_name, dtype = str)
    
	def initial_data_drop(self):
		# columns loop below was being called at initization of object
		# added following if clause to prevent...
		if not self.raw_data.empty:   
			columns = list(self.raw_data.columns.values)
	    	print "Decide whether to Keep (1) or Drop (0) a Column"
	    	drop_cols = []
	    	for col in columns:
	    		print "Column:  %r"  % col
	    		decision = int(raw_input("Keep(1) or Drop(0) column:   "))

	    		if decision == 1:
	    			pass
	    		elif decision == 0:
	    			drop_cols.append(col)

	    	#drop unwanted cols
	    	self.dataframe = self.raw_data.drop(drop_cols, axis=1)
	    	self.final_columns = list(self.dataframe.columns.values)
	    	#print self.final_columns
    	
	def prep_numeric(self): 
		numeric_cols = ['Open', 'Close', 'High', 'Low']
		for col in numeric_cols:
			self.dataframe[col] = self.dataframe[col].convert_objects(convert_numeric=True)
			self.dataframe[col] = np.around(self.dataframe[col], 3) # 3 significant digits

	def calc_SMA(self, total_periods, step_size, eval_dataframe, write_col_bool): # calcluate the simple moving average
    	# we'll setup a loop here that "stepwises" through the data
    	# SMA (and all MA) use historical data to calculate the current value
    	# so we'll get the previous stepwise values to calculate the current period
		if not self.dataframe.empty: 
			smas = [] # holder array for all smas
	    	for i in range(0, total_periods):
	    		eval_data = []
	    		for j in range(1, step_size): #MAY NEED ADJUSTMENT -- is this starting in the right place?
		    		# FYI, we do it this way
		    		# because if it's a NEGATIVE value using iloc
		    		# it will PULL from the LAST value, which we DON'T WANT! 
		    		if not i - j >= 0: 
		    			eval_data.append(eval_dataframe.iloc[i]) # this is debatable because it includes itself as the M.A.
		    		else:
		    			eval_data.append(eval_dataframe.iloc[i-j])
		    	mean = 0
		    	for value in eval_data:
		    		mean += value
		    	mean = round(mean / len(eval_data), 2)
		    	smas.append(mean)
		    # end main foor loop
		if write_col_bool == True: # either append SMA to final dataframe OR return the SMA vector to the calling function
			# append cols... 
			self.dataframe['SMA-'+str(self.sma_num_periods)] = smas
			self.final_columns.append('SMA-'+str(self.sma_num_periods))
		else:
			return smas 

	def calc_WMA(self, total_periods, step_size, eval_dataframe, write_col_bool):
    	# calculate the weighted moving average
		if not self.dataframe.empty: 
			wmas = [] # holder array for all wmas
			for i in range(0, total_periods):
				eval_data = []
				step_size_tot = 0
				for j in range(1, step_size): #MAY NEED ADJUSTMENT -- is this starting in the right place?
					if not i - j >= 0:
						eval_data.append(eval_dataframe.iloc[i] * j)
						step_size_tot +=j
					else:
						eval_data.append(eval_dataframe.iloc[i-j] * j)
						step_size_tot +=j
				weighted_mean = 0
				k = 0
				weight_sum = 0
				eval_data = np.array(eval_data)
				eval_sum = np.sum(eval_data)
				weighted_mean = round(eval_sum / step_size_tot, 2)
				"""
				for value in eval_data:
					weighted_mean += value * self.wma_weights[k]
					weight_sum += self.wma_weights[k]
					k += 1
				weighted_mean = round(weighted_mean / weight_sum, 2)
				"""
				wmas.append(weighted_mean)
		    # end main foor loop
		if write_col_bool == True: # use this in the generalization method
			# append cols... 
			self.dataframe['WMA-'+str(self.wma_num_periods)] = wmas
			self.final_columns.append('WMA-'+str(self.wma_num_periods))
		else:
			return wmas
    
	def calc_CCI(self, step_size):
		# calclate the commodity channel index (CCI)
		lamb_const = 0.15
		sum_list = list(['High', 'Low', 'Close'])
		for item in sum_list:
			self.dataframe[item] = self.dataframe[item]
		self.typical_price = self.dataframe[sum_list].sum(axis = 1) / 3
		tp_sma = self.calc_SMA(self.typical_price.shape[0], step_size, self.typical_price, False)
		tp_sma_mean = np.mean(tp_sma) 
		tp_sma_deviation = abs(tp_sma - tp_sma_mean)
		cci = (self.typical_price - tp_sma) / (.015 * tp_sma_deviation)
		cci = np.round(cci, 4) # to constrain from being a super long number...
		self.dataframe['cci-'+str(step_size)] = cci
		self.final_columns.append('cci-'+str(step_size))
		
	def calc_RSI(self, step_size):
		# calculate the relative strength indicator
		avg_gain, avg_loss = self.calc_gain_loss_rates(self.dataframe.shape[0], step_size, self.dataframe['Close'])
		rs = np.divide(avg_gain, avg_loss)
		rsi = 100 - (100 /(1+ rs))
		self.dataframe['rsi-'+str(step_size)] = rsi
		self.dataframe['rsi-'+str(step_size)].iloc[0] = 0
		self.final_columns.append('RSI-'+str(step_size))

	def calc_gain_loss_rates(self, total_periods, step_size, eval_dataframe):
		if not self.dataframe.empty: 
			avg_gain = []
			avg_loss = []
	    	for i in range(0, total_periods):
	    		num_gains = 0
	    		num_losses = 0
	    		for j in range(1, step_size): # -- is this starting in the right place?
		    		current_close = eval_dataframe.iloc[i] # will get overwritten in applicable situations
		    		if not i - j >= 0:
		    			prev_close = eval_dataframe.iloc[i]  # MIGHT BE SOME ISSUES WITH DIRECTION / NAMING HERE... 
		    		else:
		    			prev_close = eval_dataframe.iloc[i - j]
		    			current_close = eval_dataframe.iloc[i - j +1]
		    		# increment number of gains or losses for periods
		    		if current_close > prev_close:
		    			num_gains += 1
		    		elif current_close < prev_close:
		    			num_losses += 1
			    # END J-FOR: calc avg gain / loss for period and append to arrays
	    		av_g = float(num_gains) / step_size
	    		av_l = float(num_losses) / step_size
	    		avg_gain.append(av_g)
	    		avg_loss.append(av_l)
		return avg_gain, avg_loss

	def write_finished_file(self):
		final_file = self.dataframe.to_csv(self.fin_file_name,index_label='id')

if __name__ == "__main__":
	transformation = Source() 
