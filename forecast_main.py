import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
#import h5py

from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping


#from keras.utils.visualize_util import plot
#import pydot
#import graphviz

class Company: 
	def __init__(self):
		self.get_params()
		self.read_file()
		self.initial_data_drop()
		self.gen_train_test()
		
	def get_params(self):
		print "welcome to the jungle."
		self.base_file = raw_input("RAW COMPANY file:   ") # base file
		self.root_dir = 'data/working/v5/'  # version directory
		self.fin_dir = 'data/outputs/v5/' # version directory
		self.experiment_version = raw_input("Experiment Version:    ")
		self.fin_file_name = self.fin_dir + self.experiment_version +'.csv' # --> USE THE ROOT EXP FOR FILES? OR JUST ONE OUPUT?
		self.pl_file_name = self.fin_dir + self.experiment_version +'_pl.csv'

	def read_file(self):
		print "reading file"
		self.raw_data = pd.read_csv(self.root_dir+self.base_file)

	def initial_data_drop(self):
		self.raw_data2 = self.raw_data

		print "initial_data_drop (IDs & Dates)"
		columns = list(self.raw_data.columns.values)
		if 'id' in columns:
			self.raw_data = self.raw_data.drop(['id'], axis=1)

		if 'Volume' in columns:
			self.raw_data = self.raw_data.drop(['Volume'], axis=1)
		
		if 'Date' in columns:
			self.raw_dates = self.raw_data['Date']
			#print self.raw_dates
			self.raw_data = self.raw_data.drop(['Date'], axis=1)

		# the following section is for experiment customization: ie selection of which inputs to keep or drop
		columns = list(self.raw_data.columns.values)
		#drop_cols = []
		#drop_col_nums =[] # use this so i can make it more reliable for future experiments (i.e. when dropping the same columns across different companies)
		counter = 0
		# get the columns to keep (manual version)
		"""
		drop_cols = []
		drop_col_nums = []
		for column in columns:
			print "Keep (1) or DROP (0):  %r"  % column
			if int(raw_input()) == 0:
				drop_cols.append(column)
				drop_col_nums.append(counter)
			counter += 1
		print drop_cols # so i can keep track of this for experiment documentation purposes 
		print drop_col_nums
		"""
		# v5-1
		#drop_cols = ['DGS10', 'DCOILBRENTEU', 'xCIVPART', 'UNRATE', 'CPIAUCSL', 'GFDEGDQ188S', 'HOUST', 'IC4WSA', 'USD3MTD156N', 'PCE', 'PSAVERT', 'xA191RL1Q225SBEA', 'spClose', 'DEXUSEU', 'EPS', '12mo-EPS', 'net_income', 'total_assets', 'total_revenue', 'free_cash_flow', 'total_liabilities', 'profit_margin']
		#drop_col_nums = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
		# v5-2
		#drop_cols = ['Open', 'High', 'Low', 'SMA-5', 'SMA-15', 'SMA-50', 'SMA-200', 'WMA-10', 'WMA-30', 'WMA-100', 'WMA-200', 'cci-20', 'rsi-14']
		#drop_col_nums = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
		#self.raw_data.drop(self.raw_data.columns[drop_col_nums], axis = 1, inplace=True)
		
		# v5-3 ("all params")
		print list(self.raw_data.columns.values) # again for documentation purposes 
		
	def gen_train_test(self):
		# split timeseries
		print "generating x_train, y_train, x_test"
		data_shape = self.raw_data.shape[0]
		print "data_shape of raw_data:  %r"  % data_shape
		train_len = int(round(data_shape * .9)) # get 90% of data for train
		print "train_len of raw_data:  %r"  % train_len
		
		# get rid of any NaN that may have appeared in there
		self.raw_data.replace(to_replace = np.nan, value = 0, inplace=True)		

		X_train = self.raw_data.ix[0:train_len-1, :]  # one less than train_len; train_len will be start of test
		X_train2 = self.raw_data2.ix[0:train_len-1, :]
		# last row of data set won't have a prior-day but that's okay, we can just drop it (since there's no way to validate it)
		X_test =  self.raw_data.ix[train_len:data_shape-2, :] # ones less than data_shape because of 0-index + row dropping
		X_test2 = self.raw_data2.ix[train_len:data_shape-2, :]
		# generate / extract y_vals : y_train & y_valid
		# first row has no prior-day information but day 2 is its y-val
		y_vals_raw = self.raw_data.loc[:,['Close']]
		# RENAME AXIS
		y_vals_raw.rename(columns={'Close': 'nxtDayClose'}, inplace=True)

		# zero indexing takes care of needing to manipulate by one here
		# drop first day because need "day +1" close price as the "train target" based on "day's feature inputs"
		y_train = y_vals_raw.ix[1:train_len] 
		y_test = y_vals_raw.ix[train_len+1:data_shape-1, :] # but as with X_test, we will later drop the last row
		# &&  drop last row from y_valid
		#y_valid = y_valid.iloc[0:y_valid.shape[0]-1] # as above. awkward. 
		
		self.X_train = X_train
		self.X_train2 = X_train2
		# also do checks on head / tail   
		# to test head, swap head for tail
		# commented out when not testing
		#print self.X_train.tail(5)
		self.y_train = y_train.as_matrix() # make sure they're matrix/vectors
		#print self.y_train[-5:-1]
		self.X_test = X_test
		self.X_test2 = X_test2
		#print self.X_test.tail(5)
		self.y_test = y_test.as_matrix()
		#print self.y_valid[-1]
		self.y_dates = self.raw_dates.ix[train_len+1:data_shape-1].as_matrix()

		# last step is to generate a cross-validation set
		# since we're in time series, we can't randomize (hence this process and not sci-kit...)
		# we'll dedicate 90% of data set to train, 10% to cross-validation
		data_shape = self.X_train.shape[0]
		train_len = int(round(data_shape * .9))
		X_train = self.X_train[0: train_len - 1]
		X_cv = self.X_train[train_len: data_shape]
		self.X_train = X_train
		self.X_cv = X_cv
		y_train = self.y_train[0: train_len-1]
		y_cv = self.y_train[train_len: data_shape]
		self.y_train = y_train
		self.y_cv = y_cv
		print "shapes of final train/tests: \n x_train: %r    \n y_train: %r   \n x_cv: %r  \n y_cv: %r   \n x_test: %r  \n y_test: %r" % (X_train.shape, y_train.shape, X_cv.shape, y_cv.shape, X_test.shape, y_test.shape)

		return

class Forecast:
	def __init__(self):
		self.company = Company()
		self.basic_vis()
		self.pre_process_data() #v1.x-ish: scaling, PCA, etc
		self.svm() # uses self.company.X_train/test, etc
		self.ann() # uses self.company.X_train/test, etc
		# self.ensemble()  # v1.x
		self.svm_decisions, self.svm_gain_loss = self.decisions(self.svm_preds) # this has to ouptut // generate a notion of "shares held"
		self.ann_decisions, self.ann_gain_loss = self.decisions(self.ann_preds) # this has to ouptut // generate a notion of "shares held"
		self.buy_hold_prof_loss()		
		self.profit_loss_rollup()
		self.write_final_file()

	def pre_process_data(self):
		# some STRUCTURE and CLEAN UP 
		# convert to numpy and numbers
		self.company.y_train = np.array(self.company.y_train[0:,0]) # need to recast for some reason...
		self.company.y_train = self.company.y_train.astype(float)
		# so do y_valid too
		self.company.y_test = np.array(self.company.y_test[0:,0])
		# company.y_valid is an object..not sure...but this converts it
		self.company.y_test = self.company.y_test.astype(float)
		#print self.company.y_valid.dtype

		#  SCALE input values ...not sure if i should do the target...
		scaler = StandardScaler()
		self.daily_highs = self.company.X_test2['High']
		self.daily_lows = self.company.X_test2['Low']
		self.company.X_train = scaler.fit_transform(self.company.X_train)
		self.company.X_test = scaler.fit_transform(self.company.X_test)
		self.company.X_cv = scaler.fit_transform(self.company.X_cv)

		# make true train and CV split
		#self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.company.X_train, self.company.y_train, test_size=0.33, random_state=42)
		return

	def basic_vis(self):
		# TODO :: shift so that its not a correlation with the OPEN but with the CLOSE (since that's the dependent var)
		correlations = self.company.X_train.corr() # uses pandas built in correlation 
		# Generate a mask for the upper triangle (cuz it's just distracting)
		mask = np.zeros_like(correlations, dtype=np.bool)
		mask[np.triu_indices_from(mask)] = True
		# Set up the matplotlib figure
		f, ax = plt.subplots(figsize=(11, 9))
		plt.title("Feature Correlations")
		# Generate a custom diverging colormap
		cmap = sns.diverging_palette(220, 10, as_cmap=True)

		# Draw the heatmap with the mask and correct aspect ratio
		sns.heatmap(correlations, mask=mask, cmap=cmap, vmax=.3,
            square=False, xticklabels=3, yticklabels=True,
            linewidths=.6, cbar_kws={"shrink": .5}, ax=ax)
		plt.yticks(rotation=0) 
		#plt.show()
		f.savefig(self.company.fin_dir + '/correlation-images/' + self.company.experiment_version+'.png')
		
	def svm(self):
		# for regression problems, scikitlearn uses SVR: support vector regression
		C_range = np.logspace(0, 4, 6) # normally 12; doing 10 for now due to run-time length
		#print C_range
		gamma_range = np.logspace(-5, 1, 6)  # normally 12; doing 10 for now due to run-time length
		#print gamma_range
		param_grid = dict(gamma=gamma_range, C=C_range)
		# based on LONG test with the gridsearch (see notes) for v4b-5
		# below is rounded numbers
		#param_grid = dict(C=[432876], gamma=[1.8738])
		## probably want to introduce max iterations...
		grid = GridSearchCV(svm.SVR(kernel='rbf', verbose=True), param_grid=param_grid, cv=2, scoring = 'mean_squared_error')
		grid.fit(self.company.X_train, self.company.y_train)

		print("The best parameters are %s with a score of %0.2f"
			% (grid.best_params_, grid.best_score_))

		self.svm_preds = grid.predict(self.company.X_test)
		# this is for repeating or one-off specific experiments
		#self.svm_C = float(raw_input("input C val:   "))
		#self.svm_gamma = float(raw_input("input gamma val:   "))
		#regression = svm.SVR(kernel='rbf', C=self.svm_C, gamma=self.svm_gamma, verbose=True)
		#regression.fit(self.X_train, self.y_train)
		#self.svm_preds = regression.predict(self.company.X_test)
		
		#print self.svm_preds
		
		self.svm_mse_cv = grid.score(self.company.X_cv, self.company.y_cv)
		print "(cv) Mean Squared Error: %f" % self.svm_mse_cv
		self.svm_mse_test = grid.score(self.company.X_cv, self.company.y_cv)
		print "(test) Mean Squared Error: %f" % self.svm_mse_test
		
		# save the parameters to a file
		joblib.dump(grid.best_estimator_,  self.company.fin_dir + '/svm-models/' + self.company.experiment_version +'_svm_model.pkl')
		
		# visualize results 
		plt.figure()
		plt.title("SVM Learning Curve: " + self.company.experiment_version)
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		train_sizes, train_scores, test_scores = learning_curve(
			grid.best_estimator_, self.company.X_train, self.company.y_train, cv=5, train_sizes=[50, 100, 200, 300, 400, 500, 600])
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()
		plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
			alpha=0.1, color="r")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, 
			alpha=0.1, color="g")
		plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
		plt.legend(loc="best")
		plt.savefig(self.company.fin_dir + '/svm-learning-curves/' + self.company.experiment_version+'.png')

	def ann(self):
		#print self.company.X_train.shape[1]
		
		model = Sequential()
		model.add(Dense(input_dim=self.company.X_train.shape[1], output_dim=50, init="glorot_uniform"))
		#model.add(Activation('tanh'))
		model.add(Dropout(0.1))
		model.add(Dense(input_dim=50, output_dim=10, init="uniform"))
		model.add(Activation('tanh'))
		#model.add(Dropout(0.5))
		model.add(Dense(input_dim=10, output_dim=1, init="glorot_uniform"))
		model.add(Activation("linear"))

		sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mean_squared_error', optimizer='rmsprop')
		early_stopping = EarlyStopping(monitor='val_loss', patience=110)

		model.fit(self.company.X_train, self.company.y_train, nb_epoch=1000, validation_split=.1, batch_size=16, verbose = 1, show_accuracy = True, shuffle = False, callbacks=[early_stopping])
		self.ann_mse = model.evaluate(self.company.X_cv, self.company.y_cv, show_accuracy=True, batch_size=16)
		print self.ann_mse
		self.ann_preds = model.predict(self.company.X_test)

		yaml_string = model.to_yaml()
		with open(self.company.fin_dir + '/ann-models/' + self.company.experiment_version +'_ann_model.yml', 'w+') as outfile:
			outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
		#model.save_weights(self.company.fin_dir + '/ann-models/' + self.company.experiment_version +'_ann_weights')
		"""
		nb_features = self.company.X_train.shape[1]
		X_train = self.company.X_train.reshape(self.company.X_train.shape + (1, ))
		X_test = self.company.X_test.reshape(self.company.X_test.shape + (1, ))
		print X_train.shape

		model = Sequential()
		model.add(Convolution1D(nb_filter = 24, filter_length = 1, input_shape =(nb_features,1) ))
		model.add(Activation("tanh"))
		model.add(Dropout(0.2)) # some dropout to help w/ overfitting
		model.add(Convolution1D(nb_filter = 48, filter_length= 1, subsample_length= 1))
		model.add(Activation("tanh"))
		model.add(Convolution1D(nb_filter = 96, filter_length= 1, subsample_length=1))
		model.add(Activation("tanh"))
		model.add(Dropout(0.3))
		model.add(Convolution1D(nb_filter = 192, filter_length= 1, subsample_length=1))
		model.add(Activation("tanh"))
		model.add(Dropout(0.6))
		model.add(MaxPooling1D(pool_length=2))
		# flatten to add dense layers
		model.add(Flatten())
		#model.add(Dense(input_dim=nb_features, output_dim=50))
		model.add(Dense(nb_features * 2))
		model.add(Activation("tanh"))
		#model.add(Dropout(0.5))
		model.add(Dense(1))
		model.add(Activation("linear"))
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mean_squared_error', optimizer='sgd')
		early_stopping = EarlyStopping(monitor='val_loss', patience=5)

		model.fit(X_train, self.company.y_train, nb_epoch=50, validation_split=0.25, verbose = 1, callbacks=[early_stopping])
		self.ann_preds = model.predict(X_test)
		"""
		#print self.ann_preds
		#print "Trained ANN Score: %r" % score
		# visualize
		#plot(model, to_file= '/ann-training/' + self.company.fin_file_name + '.png')

		return

	def ensemble(self):
		return

	def decisions(self, predictions):
		# intializations: self.shares_held = 0  & buy_price = 0 
		self.shares_held = 0
		self.buy_price = 0
		decisions = []
		gain_loss = []
		num_preds = predictions.shape[0]

		#print "total number of predictions: %f"  % num_preds
		#print "shape of y_test: %f  "  % self.company.y_test.shape

		# loop through each prediction and make a purchase decision
		# uses for-i loop because i want to use the int for indexing within
		for i in range(0,num_preds):
			# SETUP
			# the actual close value
			actual_close = round(self.company.y_test[i],3)
			day_high = self.daily_highs.iloc[i]
			day_low = self.daily_lows.iloc[i]
			# the previous close, pulled from y_train (for first row of x) and y_test
			if i == 0:
				prv_close = round(self.company.y_train[-1],3)
			else:
				prv_close = round(self.company.y_test[i-1],3)
			#print "%r ::   %r"  % (prv_close, predictions[i])

			# *have* to liquidate on the last day	
			if (i == num_preds -1) and (self.shares_held > 0): 
				sell_price = (day_high + day_low) / 2 # mean of prv & actual..."market-ish price"
				gain_loss.append(sell_price * self.shares_held - self.buy_price * self.shares_held )
				decisions.append("final_day_liquidation")
				break 

			# ACTUAL DECISIONS 
			# buy
			if predictions[i] > prv_close and self.shares_held == 0:
				# have to fabricate a buy price: using mean of prv close & actual close...seems sort of realistic...could do mean of high, low, open too...
				self.buy_price = round((day_high + day_low) / 2, 3)
				self.shares_held = int(round(1000 / self.buy_price))
				#print "shares purchased: %r at %r"  % (self.shares_held, self.buy_price)
				#print "actual close: %r   ::  predicted close: %r    ::   previous close: %r " % (actual_close, predictions[i], prv_close)
				decisions.append("purchase")
			# sells (stop loss)
			elif (self.buy_price > prv_close) and (self.shares_held > 0):
				# stop loss check; if not > 3% loss, then no change
				if (prv_close / self.buy_price) < .97:
					sell_price = (day_high + day_low) / 2 # mean of prv & actual..."market-ish price"
					gain_loss.append(sell_price * self.shares_held - self.buy_price * self.shares_held)
					# reset holdings
					self.shares_held = 0
					self.buy_price = 0
					decisions.append("stop_loss_sell")
				else: # could do dollar cost averaging here (if wanted to get fancy)
					decisions.append("Hold") 
			# sells (stop gain)
			elif (self.buy_price < prv_close) and (self.shares_held > 0):
				# stop gain check; if not > 10% gain, then no change
				if (prv_close / self.buy_price) > 1.09:
					sell_price = (day_high + day_low) / 2 # mean of prv & actual..."market-ish price"
					gain_loss.append(sell_price * self.shares_held - self.buy_price * self.shares_held )
					self.shares_held = 0
					self.buy_price = 0
					decisions.append("stop_gain_sell")
				else:
					decisions.append("Hold")
			else:
				decisions.append("Hold")	
		
		#print decisions

		return decisions, gain_loss

	def profit_loss_rollup(self):
		# could output something like shares purchased / sold, cost basis & exit-price
		# for now just a single line
		
		columns = ["Profit/Loss"]
		index = ["BUY-HOLD", "SVM", "ANN", "SVM-MSE-CV", "SVM-MSE-TEST", "ANN-MSE"]
		self.profit_df = [self.bh_pl, np.sum(self.svm_gain_loss), np.sum(self.ann_gain_loss), self.svm_mse_cv, self.svm_mse_test, self.ann_mse]
		self.profit_df = pd.DataFrame(self.profit_df, index=index, columns=columns)
		print "Buy & Hold profit/loss %r" % self.bh_pl
		#print self.svm_decisions
		print "SVM profit/loss %r" % np.sum(self.svm_gain_loss)
		#print self.ann_decisions
		print "ANN profit/loss %r" % np.sum(self.ann_gain_loss)
		return

	def buy_hold_prof_loss(self):
		# buy price somewhere (mean) between the previous two period close prices
		buy_price = round((self.company.y_test[0] + self.company.y_train[-1]) / 2,3)
		shares_purchased = int(round(1000/ buy_price, 0))
		# sell price somewhere (mean) between the previous two perioud close prices
		sell_price = round((self.company.y_test[-2] + self.company.y_test[-1]) /2 ,3)
		self.bh_pl = sell_price * shares_purchased - buy_price * shares_purchased

		return

	def write_final_file(self):
		columns = ['Actual', 'SVM', 'ANN', 'SVM-decisons', 'ANN-decisions']
		# going to make a data frame to print to a csv
		# but preds were not all in the same shape
		# this helps with that and merges them all up
		self.final_df = np.vstack((self.company.y_test, self.svm_preds))
		self.final_df = np.transpose(self.final_df)
		self.final_df = np.hstack((self.final_df, self.ann_preds))
		#print self.final_df.shape
		#print np.array( [self.svm_decisions] ).shape
		self.final_df = np.hstack((self.final_df, np.transpose(np.array( [self.svm_decisions] ))  ))
		self.final_df = np.hstack((self.final_df, np.transpose(np.array( [self.ann_decisions] ))  ))
		self.final_df = pd.DataFrame(self.final_df, columns=columns)
		self.final_df['Date'] = self.company.y_dates
		
		final_file = self.final_df.to_csv(self.company.fin_file_name,index_label='id')
		pl_fin_file = self.profit_df.to_csv(self.company.pl_file_name, index=True)
		return

if __name__ == "__main__":
	forecast = Forecast() 