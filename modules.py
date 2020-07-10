import classes

import re
import sys
import copy
import math
import string
import random
import matplotlib.colors

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

def chunks(l, n):
	"""Yield successive n-sized chunks from l.

	Notes: from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

	Args:
		l (iterable): source of chunks
		n (int): chunk size

	Yield:
		chunks from l

	"""
	for i in range(0, len(l), n):
		yield l[i:i + n]

def label_point(x, y, val, col='tab:blue'):
	"""Creates graph annotations.

	Notes: modified from https://stackoverflow.com/questions/46027653/adding-labels-in-x-y-scatter-plot-with-seaborn

	Args:
		(x,y) (int, int): location in plt for annotation
		val (str): annotation to be displayed
		col (str): color of annotation

	"""
	for i in range(len(x)):
		plt.text(x[i],y[i],str(val[i]),color=col)

def analyzeLoadings(cell_union,cell_lines,matchedLoadings1,matchedLoadings2,by='rank'):
	"""Compares loading data between multiple PCAs

	Args:
		cell_union (df): contains all peptides that occur in at least one comparison, used to generate index
		cell_lines (list): cell lines to name columns
		matchedLoadings (list of df): loading scores for PC1 and PC2 for each peptide
		by (str): optional parameter to specify how to determine averages
			'rank': avg-1 will be average of relative rank of peptide for PC1
			'value': avg-1 will be average of raw loading values of peptide for PC1

	Returns:
		rankedLoadings (df): contains ranks of peptide impacts on PC1 and PC2, as well as average ranking across all others to reference comparisons and n for averaging
			...contains all peptides found in at least one intersectioon between others and reference, so not all peptides are expressed in all comparisons
			...any entries equal to 0 contain no peptide information and are not included in the avg or n
		ranked1 (df): peptides ordered by decreasing impact on PC1
		ranked2 (df): peptides ordered by decreasing impact on PC2

	Notes:
		'rank' is probably a more robust way to compare loadings between PCAs

	"""
	rankedLoadings = pd.DataFrame(data = np.zeros((cell_union.shape[0],(len(cell_lines)+2)*2)),index=cell_union.index)

	#renames columns to be more understandable
	names = [str(each) for each in cell_lines]*2
	for i in range(len(names)):
		names[i] = names[i]+"-"+str(math.floor(i/len(cell_lines))+1)
	names.insert(int(len(names)/2),'n-1')
	names.insert(int(len(names)/2),'avg-1')
	names.append('avg-2')
	names.append('n-2')
	rankedLoadings.columns = names
	print(names)
	print(rankedLoadings)

	#populates rankedLoadings with individual ranks for PC1 and PC2 for each others to reference comparison
	for i in range(len(matchedLoadings1)):
		#PC1
		for j in range(len(matchedLoadings1[i])):
			try:
				peptide = matchedLoadings1[i].iloc[j].name
				cell_line = names[i]
				if by == 'rank':
					#j is integer index in peptide's impact on PC1 in descending order (most impactful peptide will at j = 1)
					rankedLoadings.loc[peptide][cell_line] = j+1
				elif by == 'value':
					#value is loading value for peptide's impact on PC1
					rankedLoadings.loc[peptide][cell_line] = matchedLoadings1[i].iloc[j][0]
			except Exception as e:
				#no readout confirms all peptides are in rankedLoadings index
				print(e)
		#PC2
		for j in range(len(matchedLoadings2[i])):
			try:
				peptide = matchedLoadings2[i].iloc[j].name
				cell_line = names[i+int(len(names)/2)]
				if by == 'rank':
					#j is integer index in peptide's impact on PC2 in descending order (most impactful peptide will at j = 1)
					rankedLoadings.loc[peptide][cell_line] = j+1
				elif by == 'value':
					#value is loading value for peptide's impact on PC2
					rankedLoadings.loc[peptide][cell_line] = matchedLoadings2[i].iloc[j][1]
			except Exception as e:
				print(e,'for ',peptide)

	#averages data that does not contain a 0 (no peptide in that others to reference comparison)
	#also stores n for averaging
	#PC1
	for each in names[0:int(len(names)/2-2)]:
		val = rankedLoadings[each]
		rankedLoadings['avg-1'] += abs(val)
		#if peptide rank is 0, val = False = 0; if peptide rank is not 0, val = True = 1
		rankedLoadings['n-1'] += val != 0
	rankedLoadings['avg-1'] /= rankedLoadings['n-1']
	#PC2
	for each in names[int(len(names)/2):len(names)-2]:
		val = rankedLoadings[each]
		rankedLoadings['avg-2'] += abs(val)
		rankedLoadings['n-2'] += val != 0
	rankedLoadings['avg-2'] /= rankedLoadings['n-2']

	if by == 'rank':
		#sorts using sort_values because do not need to worry about absolute value
		ranked1 = rankedLoadings.sort_values(by="avg-1")
		ranked2 = rankedLoadings.sort_values(by="avg-2")
	elif by == 'value':
		#sort using np.argsort to sort by absolute value
		ranked1 = rankedLoadings.iloc[np.argsort(-1*abs(rankedLoadings["avg-1"])).values]
		ranked2 = rankedLoadings.iloc[np.argsort(-1*abs(rankedLoadings["avg-2"])).values]

	return rankedLoadings,ranked1,ranked2


def fullIntersection(cellData, caseInsensitive):
	"""Performs full intersection of cell line data from one replicate.
	
	Args:
		cellData (list of df): replicate data to be intersected
		caseInsensitive (bool): case insensitivity of intersection

	Returns:
		intersection (df): only peptides that appear in all dfs in cellData

	"""

	cell_data = copy.deepcopy(cellData)
	# print(cell_data)

	if caseInsensitive:
		for i in range(len(cell_data)):
			cell_data[i]['upper-peptide'] = cell_data[i]["peptide-phosphosite"].str.upper()
			cell_data[i] = cell_data[i].drop('peptide-phosphosite',axis=1)

	if caseInsensitive:
		#performs initial intersection of 2934 and 2935 based on upper-peptide and Master Protein Descriptions
		intersection = pd.merge(cell_data[0],cell_data[1],how="inner",on=["upper-peptide","Master Protein Descriptions", "Overflow Protein Descriptions"])

	else:
		#performs initial intersection of 2934 and 2935 based on peptide-phosphosite and Master Protein Descriptions
		intersection = pd.merge(cell_data[0],cell_data[1],how="inner",on=["peptide-phosphosite","Master Protein Descriptions", "Overflow Protein Descriptions"])
	#iterates through rest of cell lines and finds intersection with growing DataFrame of intersecting peptides
	for i in range(2,len(cell_data)):
		if caseInsensitive:
			intersection = pd.merge(intersection,cell_data[i],how="inner",on=["upper-peptide","Master Protein Descriptions", "Overflow Protein Descriptions"])
		else:
			intersection = pd.merge(intersection,cell_data[i],how="inner",on=["peptide-phosphosite","Master Protein Descriptions", "Overflow Protein Descriptions"])

	if caseInsensitive:
		intersection.rename(columns = {'upper-peptide': 'peptide-phosphosite'},inplace = True)

	# print(intersection)
	intersection = intersection.set_index(["peptide-phosphosite",'Master Protein Descriptions', 'Overflow Protein Descriptions'])

	return intersection


def separateIntersections(cellData, commonData, caseInsensitive, combine = False):
	"""Performs separate intersections of cell line data with reference.
	
	Args:
		cellData (list of df): others data to be intersected
		commonData (df): reference data to be intersected with
		caseInsensitive (bool): case insensitivity of intersection
		combine (bool): whether to include reference at the end of the output

	Returns:
		cell_data (list of df): list of others intersections with reference, maybe original reference as cell_data[-1]

	"""

	cell_data = copy.deepcopy(cellData)
	common_data = copy.deepcopy(commonData)

	if caseInsensitive:
		for i in range(len(cell_data)):
			cell_data[i]['upper-peptide'] = cell_data[i]["peptide-phosphosite"].str.upper()
			cell_data[i] = cell_data[i].drop('peptide-phosphosite',axis=1)
		common_data['upper-peptide'] = common_data["peptide-phosphosite"].str.upper()
		common_data = common_data.drop('peptide-phosphosite',axis=1)

	#iterates through rest of cell lines and finds intersection with growing DataFrame of intersecting peptides
	for i in range(len(cell_data)):
		if caseInsensitive:
			cell_data[i] = pd.merge(cell_data[i],common_data,how="inner",on=["upper-peptide","Master Protein Descriptions", "Overflow Protein Descriptions"])
			cell_data[i].rename(columns = {'upper-peptide':'peptide-phosphosite'},inplace = True)
		else:
			cell_data[i] = pd.merge(cell_data[i],common_data,how="inner",on=["peptide-phosphosite","Master Protein Descriptions", "Overflow Protein Descriptions"])

		cell_data[i] = cell_data[i].set_index(["peptide-phosphosite",'Master Protein Descriptions', 'Overflow Protein Descriptions'])

	if combine:
		if caseInsensitive:
			common_data = common_data.rename(columns = {'upper-peptide':'peptide-phosphosite'},inplace = True)
		cell_data.append(common_data)

	return cell_data

def combineReplicates(experiment, n_cutoff, std_cutoff):
	"""Combines replicates using outer pd.merge().

	Args:
		experiment (Experiment): experiment to draw replicates from
		n_cutoff (int): minimum number of replicates a peptide shows up in to be included
		std_cutoff (int): minimum std dev across replicates for a peptide to be included

	Returns:
		stat_data (list of df): combined peptide data, including mean, n, std dev, min, and max for each peptide
		merged_data (list of df): combined peptide data
		full_replicate_data (list of df): combined peptide data filtered by n or std but with n replicates listed individually

	"""
	replicates = []
	for each in experiment.experimentalReplicates:
		replicates.append(copy.deepcopy(each))
	
	names = {}
	finalNames = []
	for each in experiment.cellLines:
		line = []
		for every in experiment.timePoints:
			names[str(each)+"-"+str(every)] = str(each)+"-"+str(every)+"_1"
			line.append(str(each)+"-"+str(every))
		finalNames.append(line)

	stat_data = [0]*len(experiment.cellLines)
	merged_data = [0]*len(experiment.cellLines)
	full_replicate_data = [0]*len(experiment.cellLines)

	for i in range(len(experiment.cellLines)):
		stat_data[i] = pd.merge(replicates[0][i],replicates[1][i],how="outer",on=["peptide-phosphosite","Master Protein Descriptions", "Overflow Protein Descriptions"],suffixes=('','_2'))

	for i in range(len(experiment.cellLines)):
		toDrop = []
		full_replicate_data[i] = pd.DataFrame()

		for k in range(2,len(replicates)):
			stat_data[i] = pd.merge(stat_data[i],replicates[k][i],how="outer",on=["peptide-phosphosite","Master Protein Descriptions", "Overflow Protein Descriptions"],suffixes=('','_'+str(k+1)))
		
		stat_data[i] = stat_data[i].rename(columns = names)
		stat_data[i] = stat_data[i].set_index(["peptide-phosphosite",'Master Protein Descriptions', 'Overflow Protein Descriptions'])

		for time in experiment.timePoints:
			cellLine = experiment.cellLines[i]
			#selects only columns for specified cell line and time
			consideredReplicates = [str(cellLine)+"-"+str(time)+"_"+str(n) for n in range(1,len(replicates)+1)]
			#adds mean, std, n, min, and max
			stat_data[i][str(cellLine)+"-"+str(time)] = stat_data[i][consideredReplicates].mean(axis=1)
			stat_data[i][str(cellLine)+"-"+str(time)+"_std"] = stat_data[i][consideredReplicates].std(axis=1)
			stat_data[i][str(cellLine)+"-"+str(time)+"_std"] = stat_data[i][str(cellLine)+"-"+str(time)+"_std"].fillna(0)
			stat_data[i][str(cellLine)+"-"+str(time)+"_n"] = stat_data[i][consideredReplicates].count(axis=1)
			stat_data[i][str(cellLine)+"-"+str(time)+"_min"] = stat_data[i][consideredReplicates].min(axis=1)
			stat_data[i][str(cellLine)+"-"+str(time)+"_max"] = stat_data[i][consideredReplicates].max(axis=1)

			#adds individual replicate data to full_replicate_data df
			for each in consideredReplicates:
				full_replicate_data[i][each] = stat_data[i][each]

			#removes columns from individual replicates
			stat_data[i] = stat_data[i].drop(columns = consideredReplicates)

			#cutoffs
			#bool series of whether peptide row meets thresholds
			#n
			n_true = stat_data[i][str(cellLine)+"-"+str(time)+"_n"] >= n_cutoff
			stat_data[i] = stat_data[i][n_true.values]
			full_replicate_data[i] = full_replicate_data[i][n_true.values]

			#std
			std_true = stat_data[i][str(cellLine)+"-"+str(time)+"_std"] <= std_cutoff
			stat_data[i] = stat_data[i][std_true.values]
			full_replicate_data[i] = full_replicate_data[i][std_true.values]
			
			toDrop.append(str(cellLine)+"-"+str(time)+"_n")
			toDrop.append(str(cellLine)+"-"+str(time)+"_std")
			toDrop.append(str(cellLine)+"-"+str(time)+"_min")
			toDrop.append(str(cellLine)+"-"+str(time)+"_max")

		merged_data[i] = stat_data[i].drop(columns = toDrop)

		stat_data[i].reset_index(inplace = True)
		merged_data[i].reset_index(inplace = True)
		full_replicate_data[i].reset_index(inplace = True)
	return stat_data,merged_data,full_replicate_data

def background_gradient(s, m, M, cmap='bwr', low=0, high=0):
	#entire df-wise gradient generation, modified from https://stackoverflow.com/questions/38931566/pandas-style-background-gradient-both-rows-and-columns
    # rng = M - m
    #centers to 0
    rng = max(M-0, 0-m)
    norm = matplotlib.colors.Normalize(0-rng, 0+rng)
    # norm = matplotlib.colors.Normalize(m - (rng * low), M + (rng * high))
    normed = norm(s.values)
    c = [matplotlib.colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]

def heatmap(intersection, cell_lines, time_points, name, display, saveFile, saveFig, fileLocation, fullscreen, normalization):
	#splits into reference and other cell lines
	reference = intersection.iloc[:,-1*len(time_points):]
	others = intersection.iloc[:,:-1*len(time_points)]

	if normalization == 'refbasal':
		#normalize all cell line time points to last cell line's first time point
		intersection = np.log2(intersection).subtract(np.log2(reference.iloc[:,0]), axis='rows')
		title = ' Peptide Abundances Normalized to {}-{} Time Point'.format(cell_lines[-1], time_points[0])
	elif normalization == 'reftime':
		for i in range(len(time_points)):
			#all other results at a given time point
			indices = [i+len(time_points)*j for j in range(len(cell_lines))]
			#normalize by log2(other time point) - log2(reference time point)
			intersection.iloc[:,indices] = np.log2(intersection.iloc[:,indices]).subtract(np.log2(reference.iloc[:,i]), axis='rows')
		title = ' Peptide Abundances Normalized to {} Time Points'.format(cell_lines[-1])
	elif normalization == 'ownbasal':
		for i in range(len(cell_lines)):
			#all time points for one cell line
			indices = [j+i*len(time_points) for j in range(len(time_points))]
			#normalize cell line's time points to first time point
			intersection.iloc[:,indices] = np.log2(intersection.iloc[:,indices]).subtract(np.log2(intersection.iloc[:,indices[0]]), axis='rows')
		title = ' Peptide Abundances Normalized to Own {} Time Points'.format(time_points[0])
	else:
		print("ERROR: select refbasal, reftime, ownbasal for normalization.")

	sns.set()
	#plots heatmap with clusters for peptides
	g = sns.clustermap(intersection, yticklabels=False, xticklabels=True, center=0, row_cluster = True, col_cluster=False, cmap = 'bwr')
	g.ax_heatmap.set_title(name+title)
	#adds dividers between cell lines
	dividers = [i*len(time_points) for i in range(0,len(cell_lines)+1)]
	#adjusts edge dividers slightly to ensure proper thickness
	dividers[0] += .05
	dividers[-1] -= .05
	g.ax_heatmap.vlines(dividers, *g.ax_heatmap.get_ylim(), colors='k')

	if saveFig:
		print('writing to:'+fileLocation+title+' heatmap.png')
		plt.savefig(fileLocation+title+' heatmap.png')
		plt.close()

	if saveFile:
		r_ind = g.dendrogram_row.reordered_ind
		with pd.ExcelWriter("{}{} heatmap.xlsx".format(fileLocation, title)) as writer:
			intersection.iloc[r_ind,:].style.apply(background_gradient, cmap = 'bwr', m = intersection.min().min(), M = intersection.max().max()).to_excel(writer, engine = 'openpyxl')

	if not display:
		plt.close()

def heatmapToReference(cellData, cell_lines, time_points, name, display, saveFile, saveFig, fileLocation, fullscreen, normalization):
	collection = []
	titles = []
	min_val = float("inf")
	max_val = -1*float("inf")
	saveTitle = 'heatmaps to {} normalized to {}'.format(cell_lines[-1], normalization)

	for ind, intersection in enumerate(cellData):
		others = intersection.iloc[:,:len(time_points)]
		reference = intersection.iloc[:,len(time_points):]

		if normalization == 'refbasal':
			#normalize all cell line time points to last cell line's first time point
			intersection = np.log2(intersection).subtract(np.log2(reference.iloc[:,0]), axis='rows')
			title = ' {} Peptide Abundances Normalized to {}-{} Time Point'.format(cell_lines[ind], cell_lines[-1], time_points[0])
		elif normalization == 'reftime':
			for i in range(len(time_points)):
				#all other results at a given time point
				indices = [i+len(time_points)*j for j in range(2)]
				#normalize by log2(other time point) - log2(reference time point)
				intersection.iloc[:,indices] = np.log2(intersection.iloc[:,indices]).subtract(np.log2(reference.iloc[:,i]), axis='rows')
			title = ' {} Peptide Abundances Normalized to {} Time Points'.format(cell_lines[ind], cell_lines[-1])
		elif normalization == 'ownbasal':
			for i in range(2):
				#all time points for one cell line
				indices = [j+i*len(time_points) for j in range(len(time_points))]
				#normalize cell line's time points to first time point
				intersection.iloc[:,indices] = np.log2(intersection.iloc[:,indices]).subtract(np.log2(intersection.iloc[:,indices[0]]), axis='rows')
			title = ' {} Peptide Abundances Normalized to Own {} Time Points'.format(cell_lines[ind], time_points[0])
		else:
			print("ERROR: select refbasal, reftime, ownbasal for normalization.")

		collection.append(others)
		titles.append(title)
		min_val = min(min_val,others.values.min())
		max_val = max(max_val,others.values.max())

	sns.set()

	def looper():
		for i, others in enumerate(collection):
			#plots heatmap with clusters for peptides
			g = sns.clustermap(others, yticklabels=False, xticklabels=True, center=0, row_cluster = True, col_cluster=False, cmap = 'bwr', vmin = min_val, vmax = max_val)
			g.ax_heatmap.set_title(name+titles[i])
			r_ind = g.dendrogram_row.reordered_ind
			
			if saveFig:
				print('writing to:'+fileLocation+titles[i]+' heatmap.png')
				plt.savefig(fileLocation+titles[i]+' heatmap.png')
				plt.close()

			if saveFile:
				others.iloc[r_ind,:].style.apply(background_gradient, cmap = 'bwr', m = min_val, M = max_val).to_excel(writer, sheet_name = str(cell_lines[i]), engine = 'openpyxl')

			if not display:
				plt.close()

	if saveFile:
		with pd.ExcelWriter("{}{}.xlsx".format(fileLocation, saveTitle)) as writer:
			looper()
	else:
		looper()


def pcaToReference(cellData, cell_lines, time_points, time_points_seconds, name = "", display = True, saveFile = False, saveFig = False,  fileLocation = "", fullscreen = False, colors = None):
	"""Plots separate PCAs comparing each others to reference individually.

	Args:
		intersection (list of df): full intersection of replicate data (all otherss and reference)
		cell_lines (list of int): cell lines in order
		time_points (list of int): time points in order
		time_points_seconds (list of int): time points in seconds in order
		name (str): plot title
		display (bool): whether to display graph or save it to file
		fileLocation (str): file location to save to
		fullscreen (bool): whether to display graph fullscreen

	Notes:
		Useful tutorial https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

	"""

	cell_data = copy.deepcopy(cellData)

	for i in range(len(cell_data)):
		cell_data[i] = cell_data[i].T

	#separates intersection data into separate cell lines, transposes it, and scales it
	for i in range(len(cell_data)):
		cell_data[i] = StandardScaler().fit_transform(cell_data[i])

	sns.set()

	#list of pca objects
	pcas = [PCA(n_components=2)]*(len(cell_lines)-1)
	#list of transformed data (has x and y of each time point for each cell line in pca space)
	transformed = [0]*(len(cell_lines)-1)
	explained = [0]*(len(cell_lines)-1)
	loadings = [0]*(len(cell_lines)-1)

	#performs pca and creates plottable transformation of time points
	for i in range(len(pcas)):
		# print(cell_data[i])
		pcas[i].fit(cell_data[i])
		transformed[i] = pcas[i].transform(cell_data[i])
		explained[i] = pcas[i].explained_variance_ratio_
		

	#plot each cell line individually
	for i in range(len(pcas)):

		#sets figure so user can call plt.show() after plotting multiple graphs and not have them on the same figure
		f = plt.figure(random.randint(1,1000))

		cell_indices = [i, -1]

		chunked = [each for each in chunks(transformed[i],5)]

		for j in range(len(chunked)):
			plt.plot(chunked[j][:,0],chunked[j][:,1],label=str(cell_lines[cell_indices[j]]),color=colors[cell_indices[j]])
		#if you want a scatter with no lines, uncomment this life, comment plt.plot above, change plt to ax for all but plt.show(), and add a param to label_point for ax
		#ax = sns.scatterplot(chunks[i][:,0],chunks[i][:,1],label=str(cell_lines[i]),marker='$.$')
			label_point(chunked[j][:,0],chunked[j][:,1],time_points_seconds,colors[cell_indices[j]])

		#sets overall plot aesthetics
		plt.title(name+" PCA of {} to {}".format(cell_lines[i], cell_lines[-1]))
		plt.xlabel('1st principal component ('+str(round(explained[i][0]*100,2))+"%)")
		plt.ylabel('2nd principal component ('+str(round(explained[i][1]*100,2))+"%)")
		plt.legend(loc='upper right')

		if fullscreen:
			mng = plt.get_current_fig_manager()
			mng.resize(*mng.window.maxsize())

		if saveFig:
			print('writing to:{} pca of {} to {}.png'.format(fileLocation,name,cell_lines[i],cell_lines[-1]))
			plt.savefig(fileLocation+name+' pca of '+str(cell_lines[i])+' to '+str(cell_lines[-1])+'.png')
			plt.close()

		if not display:
			plt.close()

	if saveFile:
		print(cellData)
		analyzeLoadings(cellData, cell_lines, [], [])

	return pcas

def pca(intersection, cell_lines, time_points, time_points_seconds, name = "", display = True, saveFile = False, saveFig = False,  fileLocation = "", fullscreen = False, colors = None):
	"""Plots one PCA comparing all others and reference at once.

	Args:
		intersection (list of df): full intersection of replicate data (all otherss and reference)
		cell_lines (list of int): cell lines in order
		time_points (list of int): time points in order
		time_points_seconds (list of int): time points in seconds in order
		name (str): plot title
		display (bool): whether to display graph or save it to file
		fileLocation (str): file location to save to
		fullscreen (bool): whether to display graph fullscreen

	"""

	#sets figure so user can call plt.show() after plotting multiple graphs and not have them on the same figure
	f = plt.figure(random.randint(1,1000))

	#transpose DataFrame so components are peptides and overall conditions are time points
	intersection_T = intersection.T

	#scale to fit [0,1] based on advice from https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py
	intersection_T_scaled = StandardScaler().fit_transform(intersection_T)

	#create PCA object with two principal components
	pca = PCA(n_components=2)

	#performs pca
	pca.fit(intersection_T_scaled)

	#creates plottable pca transformation of time points (gives x and y of each time point in pca space)
	transformed = pca.transform(intersection_T_scaled)

	#variance explained by 1st and 2nd principal components
	# print(pca.explained_variance_ratio_)

	#weight components
	# print(pca.components_[0])

	#sets seaborn overlay onto pyplot
	sns.set()

	#separates transformed into chunks for each cell line
	chunked = [each for each in chunks(transformed,5)]

	#plot each cell line individually
	for i in range(len(chunked)):
		plt.plot(chunked[i][:,0],chunked[i][:,1],label=str(cell_lines[i]), color = colors[i])
		#if you want a scatter with no lines, uncomment this life, comment plt.plot above, change plt to ax for all but plt.show(), and add a param to label_point for ax
		#ax = sns.scatterplot(chunks[i][:,0],chunks[i][:,1],label=str(cell_lines[i]),marker='$.$')
		label_point(chunked[i][:,0],chunked[i][:,1],time_points_seconds,colors[i])

	#sets overall plot aesthetics
	plt.title(name+" PCA")
	plt.xlabel('1st principal component ('+str(round(pca.explained_variance_ratio_[0]*100,2))+"%)")
	plt.ylabel('2nd principal component ('+str(round(pca.explained_variance_ratio_[1]*100,2))+"%)")
	plt.legend(loc='upper right')

	if fullscreen:
		mng = plt.get_current_fig_manager()
		mng.resize(*mng.window.maxsize())

	if saveFig:
		print('writing to:'+fileLocation+name+' pca.png')
		plt.savefig(fileLocation+name+' pca.png')
		plt.close()

	if saveFile:
		# print(intersection)
		loadings = pd.DataFrame(pca.components_.T, columns = ['Loading 1', 'Loading 2'], index = intersection.index)
		with pd.ExcelWriter("{}{} pca.xlsx".format(fileLocation, name)) as writer:
			loadings.to_excel(writer)

	if not display:
		plt.close()

	return pca

def save(structure, name, subnames = None, replicates = False):
	"""Saves structures to file.

	Args:
		structure (df, list of df, Experiment, ExperimentalReplicate): structure to save to file
		name (str): location and name of file

	Examples:
		save(exp,"../plots/Experiment 1) saves exp object in a new folder called plots outside of the current one with a filename Experiment 1

	"""
	try:
		#single df
		if type(structure) == pd.DataFrame:
			with pd.ExcelWriter(name+".xlsx") as writer:
				try:
					structure.to_excel(writer)
				except IndexError:
					pass
		#list of df
		elif type(structure) == list:
			if type(structure[0]) == pd.DataFrame:
				with pd.ExcelWriter(name+".xlsx") as writer:
					for i in range(len(structure)):
						try:
							if subnames:
								structure[i].to_excel(writer, sheet_name = str(subnames[i]))
							else:
								structure[i].to_excel(writer, sheet_name = str(i))
						except IndexError:
							pass
			elif type(structure[0]) == list:
				for i in range(len(structure)):
					if subnames:
						save(structure[i], name+" "+str(subnames[i][0]), subnames[i][1])
					else:
						save(structure[i], name+" "+str(i))
			else:
				print("ERROR in save; program aborted. Can only save DataFrames, Experiments, ExperimentalReplicates, or lists of these.")
				print("Cannot save lists of generic types (int, str, etc)")
				sys.exit(1)
		#Experiment or ExperimentalReplicate
		elif type(structure) == classes.Experiment or type(structure) == classes.ExperimentalReplicate:
			structure.save(name, replicates)
	except PermissionError:
		print("ERROR: Close all files before resaving.")

def splitCluster(mut, linkage, cellLine, title, fileLocation, display, saveFig):
	'''Computes the last 4 hierarchical cluster groups based on an existing linkage, then plots average trajectory for each with 95% CI.

	Args:
		mut (df): peptide trajectory information for one cell line
		linkage (linkage object): information from hierarchical clustering
		cellLine (int): name of current cell line
		title (str): title to include at beginning
		fileLocation (str): relative location to save file to
		display (bool): whether to display or save

	'''

	#dictionary for converting mixed time points to seconds
	conversion = {'0':0,'30':30,'1':60,'2':120,'5':300}

	rootNode, nodeList = hierarchy.to_tree(linkage, rd=True)

	check = [rootNode.right.right, rootNode.right.left, rootNode.left.right, rootNode.left.left]

	groups = []
	valid = 0
	for each in check:
		if each:
			groups.append([i for i in each.pre_order(lambda x: x.id)])
			valid += 1

	colors = ['#FEB308', '#9b59b6', '#7BB274', '#A8A495']
	sizes = {}

	sortedIndices = np.argsort([len(each) for each in groups])

	mut.reset_index(inplace=True)

	for i, ind in enumerate(sortedIndices):
		mut.loc[mut.index.isin(groups[ind]), 'Group'] = colors[i]
		mut.loc[mut.index.isin(groups[ind]), 'Sizes'] = len(groups[ind])
		if str(len(groups[ind])) in sizes:
			sizes[str(float(len(groups[ind])))] = "yes"
			mut.loc[mut.index.isin(groups[ind]), 'Size'] = str(float(len(groups[ind])))
		else:
			sizes[str(len(groups[ind]))] = "yes"
			mut.loc[mut.index.isin(groups[ind]), 'Size'] = str(len(groups[ind]))

	letterSort = np.argsort(list(sizes.keys()))

	#unpivots df from wide to long format for easy plotting with sns
	#makes peptide labels accessible for melt
	# print([each for each in mut])
	melted = mut.melt(id_vars = ["peptide-phosphosite","Master Protein Descriptions","Overflow Protein Descriptions","Sizes","Size", "Group"], value_name = "Log2 FC")
	# print(melted)
	melted = melted.set_index(["peptide-phosphosite", "Master Protein Descriptions", "Overflow Protein Descriptions"])

	#splits cellLine-timePoint into cellLine and timePoint for grouping
	splt = melted['variable'].values
	splt = np.array([each.split('-') for each in splt])
	melted['Cell Line'] = splt[:,0].astype(int)
	melted['Time Point'] = [conversion[each] for each in splt[:,1]]
	# melted['Time Point'] = [conversion[each] for each in melted['Time Point'].astype(str)]

	melted.drop('variable', axis = 1, inplace = True)
	melted.sort_values(by=['Sizes'], ascending = True, inplace = True)
	# print(melted['Size'])
	# print(np.unique(melted['Size']))

	g = sns.lineplot(x = "Time Point", y = "Log2 FC", data = melted, hue = "Size", palette = [colors[i] for i in letterSort], ci = 'sd', marker = "o")
	plt.title(title.format(cellLine))

	if saveFig:
		print('writing trajectory plot')
		plt.savefig(fileLocation+title+' traj.png')
		plt.close()
	# else:
	# 	plt.savefig(fileLocation+title.format(cellLine)+" line.png")
	# 	plt.close()

	return melted, mut

#CORRELATION CALCULATION
def custom_corr(one,two):
	'''Computes correlation matrix between two matrices element-wise.

	Args:
		one (df): first df
		two (df): second df

	Notes:
		Modified from https://github.com/pandas-dev/pandas/blob/v0.25.1/pandas/core/frame.py#L7451-L7534
	
	'''

	#gets peptide trajectory values
	trj1 = one.values
	trj2 = two.values

	#instantiates new df
	correl = np.empty((len(one), len(one)), dtype=float)

	for i, ac in enumerate(trj1):
		for j, bc in enumerate(trj2):

			#only calculates half of the triangle
			if i > j:
				continue

			#value will be pearson coefficient between the two trajectory vectors
			c = pearsonr(ac,bc)[0]
			correl[i,j] = c
			correl[j,i] = c

	#builds a new df to store the r values
	return pd.DataFrame(correl,index=one.index, columns = one.index)

def diagonal_corr(one,two,line):
	'''Computes correlation between two matrices element-wise, but only for diagonal.

	Args:
		one (df): first df
		two (df): second df
		line (str): name of line being compared to reference

	Notes:
		Modified from https://github.com/pandas-dev/pandas/blob/v0.25.1/pandas/core/frame.py#L7451-L7534
		Will yield a len(one)-long Series of correlation values from [-1,1], where list[i] is the correlation between one[i] and two[i].
	
	'''

	#gets peptide trajectory values
	trj1 = one.values
	trj2 = two.values

	#instantiates new df
	correl = np.empty(len(one), dtype=float)

	for i in range(len(trj1)):
		correl[i] = pearsonr(trj1[i],trj2[i])[0]

	return pd.DataFrame(correl, index = one.index, columns = [line]).reset_index()

def clustermap(corr, mut, cellLine1, cellLine2, title, fileLocation, display, saveFig):
	'''Plots a custom clustermap with the last 4 hierarchical cluster groups indicated by different colors.

	Args:
		corr (df): cell line to cell line or cell line to reference correlation matrix
		mut (df): cell line time point data for plotting trajectories
		cellLine1 (int): name of first cell line
		cellLine2 (int): name of reference cell line or first cell line again
		title (str): plot title to add
		fileLocation (str): location to save file to
		display (bool): whether to display plot

	Notes:
		#modified from https://seaborn.pydata.org/examples/many_pairwise_correlations.html

	'''

	#generate a mask for the upper triangle
	# mask = np.zeros_like(corr, dtype=np.bool)
	# mask[np.triu_indices_from(mask)] = True

	#compute linkage via scipy (matrix is symmetric, so row and column linkage is the same)
	correlations_array = np.asarray(corr)
	row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method='average')
	col_linkage = hierarchy.linkage(distance.pdist(np.transpose(correlations_array)), method='average')

	#computes last 4 hierarchical cluster groups, then plots average peptide trajectories for each
	melted, mut = splitCluster(mut.copy(), row_linkage, cellLine1, title, fileLocation, display, saveFig)

	mut = mut.set_index(["peptide-phosphosite", "Master Protein Descriptions", "Overflow Protein Descriptions"])

	#draw the clustermap. cg is ClusterGrid object returned by seaborn
	cg = sns.clustermap(corr, row_linkage=row_linkage, col_linkage = col_linkage, cmap = 'bwr', center = 0,
				norm = matplotlib.colors.Normalize(vmin=-1, vmax=1), row_colors = mut["Group"], xticklabels = False, yticklabels = False)

	#reorders peptides by hierarhical clusters
	mut = mut.iloc[cg.dendrogram_row.reordered_ind]

	#adds labels to axes of heatmap object
	cg.ax_heatmap.set_xlabel(str(cellLine1))
	cg.ax_heatmap.set_ylabel(str(cellLine2))
	#adds title to overall figure (can't just do cg.ax_heatmap.set_title() because then cluster linkages obscure title)
	cg.fig.suptitle(title)

	return mut

def color_groups(s):
	'''Colors rows based on hierarchical clustering groups
	'''
	return ['background-color: {}'.format(s['Group']) for v in s]

def color_groups_only(s):
	'''Colors group column only based on hierarchical clustering groups
	'''
	if s.name[0] == 'Group':
		return ['background-color: {}'.format(g) for g in s]
	else:
		return ['']*s.shape[0]

def correlationToReference(ints, time_points, second_time_points, cell_lines, name='', display = True, saveFile = False, saveFig = False, fileLocation = '', normalization = 'refbasal'):
	'''For each cell line, computes the correlation of all peptide trajectories to all reference peptide trajectories.

	Parameters:
		ints (list of dfs): list of individual cell line intersections with reference
		cell_lines (list): list of cell line names
		time_points (list): list of time points in trajectories
		second_time_points (list): list of second time points
		name (str): name to include in plots
		display (bool): whether to display plots
		saveFile (bool): whether to save data to Excel files
		saveFig (bool): whether to save figs
		fileLocation (str): location to save to
		normalization (str): normalization scheme, either ownbasal, reftime, or refbasal

	Notes:
		Computes the Pearson Coefficient (R) between two peptide trajectories over time.
			-1 is a negative linear relationship
			0 is no linear relationship
			+1 is a positive linear relationship

		If two peptides have trajectories over time that are positive linear scalings of each other, they will have a Pearson R close to +1

		MutantPeptide.A will be correlated with ReferencePeptide.A, .B, ... Z
		MutantPeptide.B will be ....			ReferencePeptide.A, .B, ... Z
		...
	'''
	normalizedInts = []
	sns.set(style='white')

	#NORMALIZATION
	# biggest = -1*float('inf')
	# smallest = float('inf')
	for fullInt in ints:
		if normalization == "ownbasal":
			#for normalizing to basal
			def basalNorm(x):
				line = x.name.split("-")[0]
				return np.log2(x.div(fullInt[line+"-"+str(time_points[0])]))

			fullInt = fullInt.apply(basalNorm,axis=0)
		elif normalization == "reftime":
			#for normalizing to all reference timepoints
			def timeNorm(x):
				time = x.name.split("-")[1]
				return np.log2(x.div(fullInt[str(cell_lines[-1])+"-"+time]))

			fullInt = fullInt.apply(timeNorm,axis=0)
		elif normalization == "refbasal":
			#normalizes to unstimulated reference
			fullInt = np.log2(fullInt.div(fullInt[str(cell_lines[-1])+"-"+str(time_points[0])], axis = 0))

		normalizedInts.append(fullInt)

	referenceName = cell_lines[-1]
	if name == '':
		title = ''
		saveTitle = ''
	else:
		title = name+" "
		saveTitle = name+" "
	title += "{} to " + str(referenceName) + " ({} normalized) corr".format(normalization)
	saveTitle += 'correlation to {} ({} normalized)'.format(referenceName, normalization)

	trajs = []

	#iterates through cell lines, isolating the correct columns in the df and then calculating the correlation matrix
	def looper():
		for i in range(len(normalizedInts)):
			print("\nCorrelating {} with {}".format(cell_lines[i], cell_lines[-1]))
			mut = normalizedInts[i].iloc[:,range(len(time_points))]
			reference = normalizedInts[i].iloc[:,range(len(time_points),len(time_points)*2)]
			mutName = cell_lines[i]
			fullCorr = custom_corr(mut,reference)
			mut = clustermap(fullCorr,mut,mutName,referenceName,title.format(mutName), fileLocation, display = display, saveFig = saveFig)

			if display:
				plt.show()

			if saveFig:
				print('writing correlation plot')
				plt.savefig(fileLocation+title.format(mutName)+'.png')
				plt.close()

			if saveFile:
				mut = mut.reset_index()
				mut.drop(['Sizes'], axis = 1, inplace = True)
				descriptions = mut["Master Protein Descriptions"].values
				mut["Gene Name"] = [re.split("( OS=)",each)[0] for each in descriptions]
				trajs.append(mut)

				print('writing to: {}{} C.xlsx, sheet name: {}-C'.format(fileLocation, saveTitle, mutName))

				fullCorr.to_excel(writer, sheet_name = str(mutName)+'-C')
				# mut.style.apply(color_groups, axis = 1).to_excel(writer, sheet_name = str(mutName)+'-T', engine = 'openpyxl')
				# mut = mut.set_index(["peptide-phosphosite",'Master Protein Descriptions'])
				# fullCorr['Group'] = mut['Group']
				# fullCorr.style.apply(color_groups_only, axis = 0).to_excel(writer, sheet_name = str(mutName)+'-C', engine = 'openpyxl')

			if not display:
				plt.close()

	if saveFile:
		with pd.ExcelWriter("{}{} C.xlsx".format(fileLocation, saveTitle)) as writer:
			looper()
		with pd.ExcelWriter("{}{} T.xlsx".format(fileLocation, saveTitle)) as writer:
			print('\nFinishing files.')
			for i in range(len(trajs)):
				print('writing to: {}{} T.xlsx, sheet name: {}-T'.format(fileLocation, saveTitle, cell_lines[i]))
				trajs[i].style.apply(color_groups, axis = 1).to_excel(writer, sheet_name = str(cell_lines[i])+'-T', engine = 'openpyxl')
	else:
		looper()

def correlationToSelf(inds, time_points, second_time_points, cell_lines, name, display, saveFile, saveFig, fileLocation, normalization):
	'''For each cell line, computes the correlation of all peptide trajectories to all peptide trajectories.

	Parameters:
		inds (list of dfs): list of combined experimental data for each cell line individually
		cell_lines (list): list of cell line names
		time_points (list): list of time points in trajectories
		second_time_points (list): list of second time points
		name (str): name to include in plots
		display (bool): whether to display plots
		saveFile (bool): whether to save data to Excel files
		saveFig (bool): whether to save figs
		fileLocation (str): location to save to
		normalization (str): normalization scheme, either ownbasal, reftime, or refbasal

	Notes:
		Computes the Pearson Coefficient (R) between two peptide trajectories over time.
			-1 is a negative linear relationship
			0 is no linear relationship
			+1 is a positive linear relationship
		If two peptides have trajectories over time that are positive linear scalings of each other, they will have a Pearson R close to +1

		MutantPeptide.A will be correlated with MutantPeptide.A, .B, ... Z
		MutantPeptide.B will be ....			MutantPeptide.A, .B, ... Z
		...

		If saveFile is True, will save two files, one ending in T for trajectory and one ending in C for correlation.

		The trajectory plot shows the average peptide trajectory for each of the 4 largest hierarchical clustering groups, including a spaded 95% confidence interval.
	'''
	#dictionary for converting mixed time points to seconds
	keys = [str(each) for each in time_points]
	values = [int(each) for each in second_time_points]
	conversion = dict(zip(keys, values))

	normalizedInds = []

	#NORMALIZATION
	# biggest = -1*float('inf')
	# smallest = float('inf')
	for fullInt in inds:
		fullInt = fullInt.set_index(["peptide-phosphosite",'Master Protein Descriptions', 'Overflow Protein Descriptions'])
		if normalization == "ownbasal":
			#for normalizing to basal
			def basalNorm(x):
				line = x.name.split("-")[0]
				return np.log2(x.div(fullInt[line+"-"+str(time_points[0])]))
			fullInt = fullInt.apply(basalNorm,axis=0)
		else:
			print("WARNING: isolated cell lines cannot be normalized to reference without losing peptide information")
			print("Normalize using 'ownbasal'")
			sys.exit(0)

		normalizedInds.append(fullInt)

	sns.set(style="white")

	if name == '':
		title = ''
		saveTitle = ''
	else:
		title = name+" "
		saveTitle = name+''
	title += "{} to {}" + " ({} normalized) corr".format(normalization)
	saveTitle += 'correlation to self ({} normalized)'.format(normalization)

	trajs = []

	def looper():
		for i in range(len(normalizedInds)):
			print("\nCorrelating {} with itself".format(cell_lines[i]))
			mut = normalizedInds[i]
			corr = mut.transpose().corr()
			cellLine = cell_lines[i]

			mut = clustermap(corr, mut, cellLine, cellLine, title.format(cellLine, cellLine), fileLocation, display = display, saveFig = saveFig)
			
			if display:
					plt.show()

			if saveFig:
				print('writing correlation plot')
				plt.savefig(fileLocation+title.format(cellLine, cellLine)+'.png')
				plt.close()

			if saveFile:
				mut = mut.reset_index()
				mut.drop(['Sizes'], axis = 1, inplace = True)
				descriptions = mut["Master Protein Descriptions"].values
				mut["Gene Name"] = [re.split("( OS=)",each)[0] for each in descriptions]

				print('writing to: {}{} C.xlsx, sheet name: {}-C'.format(fileLocation, saveTitle, cellLine))

				# mut.style.apply(color_groups, axis = 1).to_excel(writer, sheet_name = str(cellLine)+'-T', engine = 'openpyxl')
				# corrs.append(corr)

				mut = mut.set_index(["peptide-phosphosite",'Master Protein Descriptions','Overflow Protein Descriptions', 'Gene Name'])
				trajs.append(mut)

				# corr['Gene Name'] = mut['Gene Name']
				# corr.reset_index(inplace = True)
				# corr.set_index(["peptide-phosphosite",'Master Protein Descriptions', 'Gene Name'], inplace = True)

				corr.to_excel(writer, sheet_name = str(cellLine)+'-C')
				# corr.style.apply(color_groups_only, axis = 0).to_excel(writer, sheet_name = str(cellLine)+'-C', engine = 'openpyxl')
				# break
				# fullCorr.style.apply(color_groups_only, axis = 0).to_excel(writer, sheet_name = str(mutName)+'-C', engine = 'openpyxl')

			if not display:
				plt.close()

	if saveFile:
		with pd.ExcelWriter("{}{} C.xlsx".format(fileLocation, saveTitle)) as writer:
			looper()
		with pd.ExcelWriter("{}{} T.xlsx".format(fileLocation, saveTitle)) as writer:
			print('\nFinishing files.')
			for i in range(len(trajs)):
				print('writing to: {}{} T.xlsx, sheet name: {}-T'.format(fileLocation, saveTitle, cell_lines[i]))
				trajs[i].style.apply(color_groups, axis = 1).to_excel(writer, sheet_name = str(cell_lines[i])+'-T', engine = 'openpyxl')
	else:
		looper()

def correlationToReferenceDiagonal(ints, time_points, second_time_points, cell_lines, name='', display = True, saveFile = False, saveFig = False, fileLocation = '', normalization = 'refbasal', colors = [], bins = None, kde = False):
	'''For each cell line, computes the correlation of each peptide's trajectory to the corresponding reference peptide trajectory.

	Parameters:
		ints (list of dfs): list of individual cell line intersections with reference
		cell_lines (list): list of cell line names
		time_points (list): list of time points in trajectories
		second_time_points (list): list of second time points
		name (str): name to include in plots
		display (bool): whether to display plots
		saveFile (bool): whether to save data to Excel files
		saveFig (bool): whether to save figs
		fileLocation (str): location to save to
		normalization (str): normalization scheme, either ownbasal, reftime, or refbasal
		colors (list): colors to use for cell lines
		bins (int): number of histograms to make
		kde (bool): whether to plot a gaussian kernel density estimate or the raw histogram (https://seaborn.pydata.org/generated/seaborn.distplot.html)

	Notes:
		Computes the Pearson Coefficient (R) between two peptide trajectories over time.
			-1 is a negative linear relationship
			0 is no linear relationship
			+1 is a positive linear relationship
		If two peptides have trajectories over time that are positive linear scalings of each other, they will have a Pearson R close to +1

		MutantPeptide.A will be correlated with ReferencePeptide.A ONLY
		MutantPeptide.B will be correlated with ReferencePeptide.B ONLY
		...

		If saveFile is True, will save two files, one ending in T for trajectory and one ending in C for correlation.

		The trajectory plot shows the average peptide trajectory for each of the 4 largest hierarchical clustering groups, including a spaded 95% confidence interval.
	'''
	normalizedInts = []
	sns.set(style='white')

	#NORMALIZATION
	# biggest = -1*float('inf')
	# smallest = float('inf')
	for fullInt in ints:
		if normalization == "ownbasal":
			#for normalizing to basal
			def basalNorm(x):
				line = x.name.split("-")[0]
				return np.log2(x.div(fullInt[line+"-"+str(time_points[0])]))

			fullInt = fullInt.apply(basalNorm,axis=0)
		elif normalization == "reftime":
			#for normalizing to all reference timepoints
			def timeNorm(x):
				time = x.name.split("-")[1]
				return np.log2(x.div(fullInt[str(cell_lines[-1])+"-"+time]))

			fullInt = fullInt.apply(timeNorm,axis=0)
		elif normalization == "refbasal":
			#normalizes to unstimulated reference
			fullInt = np.log2(fullInt.div(fullInt[str(cell_lines[-1])+"-"+str(time_points[0])], axis = 0))

		# big = fullInt.max().max()
		# biggest = max(big, biggest)
		# small = fullInt.min().min()
		# smallest = min(small, smallest)
		normalizedInts.append(fullInt)

	referenceName = cell_lines[-1]
	if name == '':
		title = ''
		saveTitle = ''
	else:
		title = name+" "
		saveTitle = name+" "
	title += "Diagonal Correlation to " + str(referenceName) + " ({} normalized) corr".format(normalization)
	saveTitle += 'diagonal correlation to {} ({} normalized)'.format(referenceName, normalization)

	#iterates through cell lines, isolating the correct columns in the df and then calculating the correlation matrix
	def looper():
		for i in range(len(normalizedInts)):
			print("\nCorrelating {} with {}".format(cell_lines[i], cell_lines[-1]))
			mut = normalizedInts[i].iloc[:,range(len(time_points))]
			reference = normalizedInts[i].iloc[:,range(len(time_points),len(time_points)*2)]
			mutName = cell_lines[i]

			if i == 0:
				fullCorr = diagonal_corr(mut, reference, mutName)
				if kde:
					ax = sns.distplot(fullCorr.iloc[:,-1], hist = False, kde = True, color = colors[i], label = cell_lines[i], bins = bins)
					ax.set_title(title)
			else:
				corr = diagonal_corr(mut, reference, mutName)
				fullCorr = pd.merge(fullCorr,corr,how="outer",on=["peptide-phosphosite","Master Protein Descriptions",'Overflow Protein Descriptions'])
				if kde:
					sns.distplot(corr.iloc[:,-1], hist = False, kde = True, color = colors[i], label = cell_lines[i], bins = bins)
					ax.set_xlabel('Pearson R Coefficient')
					ax.set_ylabel('KDE for Number of Peptides')

			if not kde:
				hist, bin_edges = np.histogram(fullCorr.iloc[:,-1].dropna().values, bins = bins)
				loc = [(bin_edges[i]+bin_edges[i-1])/2 for i in range(1,len(bin_edges))]
				plt.plot(loc,hist, color = colors[i])
				plt.title(title)
				plt.xlabel('Pearson R Coefficient')
				plt.ylabel('Number of Peptides')
			
		if display:
			plt.show()

		if saveFig:
			print('writing correlation plot')
			plt.savefig(fileLocation+title+'.png')
			plt.close()
		
		if saveFile:
			descriptions = fullCorr['Master Protein Descriptions'].values
			fullCorr['Gene Name'] = [re.split("( OS=)",each)[0] for each in descriptions]
			fullCorr = fullCorr.set_index(["peptide-phosphosite",'Master Protein Descriptions','Overflow Protein Descriptions', 'Gene Name'])
			fullCorr.to_excel(writer)

		if not display:
			plt.close()
		# melted = pd.melt(fullCorr, id_vars=["peptide-phosphosite",'Master Protein Descriptions', 'Gene Name'])

		# ax = sns.barplot(x = 'peptide-phosphosite', y = 'value', hue = 'variable', data = melted, ci = None)
		# ax.set_xticklabels(None)
		# sns.distplot(fullCorr.iloc[:,0], kde = False)
		# plt.show()

		#draw the clustermap. cg is ClusterGrid object returned by seaborn
		# cg = sns.clustermap(fullCorr, cmap = 'bwr', center = 0, norm = matplotlib.colors.Normalize(vmin=-1, vmax=1), xticklabels = False, yticklabels = False)
		# plt.show()
		# #adds labels to axes of heatmap object
		# cg.ax_heatmap.set_xlabel(str(cellLine1))
		# cg.ax_heatmap.set_ylabel(str(cellLine2))
		# #adds title to overall figure (can't just do cg.ax_heatmap.set_title() because then cluster linkages obscure title)
		# cg.fig.suptitle(title)


			# break
			# mut = clustermap(fullCorr,mut,mutName,referenceName,title.format(mutName), fileLocation, display = display, saveFig = saveFig)

			# if display:
			# 	plt.show()

			# if saveFig:
			# 	print('writing correlation plot')
			# 	plt.savefig(fileLocation+title.format(mutName)+'.png')
			# 	plt.close()

			# if saveFile:
			# 	mut = mut.reset_index()
			# 	mut.drop(['Sizes'], axis = 1, inplace = True)
			# 	descriptions = mut["Master Protein Descriptions"].values
			# 	mut["Gene Name"] = [re.split("( OS=)",each)[0] for each in descriptions]

			# 	print('writing to sheet name: {}'.format(fileLocation, saveTitle, mutName))
			# 	mut.style.apply(color_groups, axis = 1).to_excel(writer, sheet_name = str(mutName), engine = 'openpyxl')
				# mut = mut.set_index(["peptide-phosphosite",'Master Protein Descriptions'])
				# fullCorr['Group'] = mut['Group']
				# fullCorr.style.apply(color_groups_only, axis = 0).to_excel(writer, sheet_name = str(mutName)+'-C', engine = 'openpyxl')

	if saveFile:
		with pd.ExcelWriter("{}{}.xlsx".format(fileLocation, saveTitle)) as writer:
			looper()
	else:
		looper()

def tukey(y, lines):
	#resets index and names it Cell Line just in case
	y = y.reset_index()
	y.rename(columns={'index':'Cell Line'}, inplace = True)
	#tukey HSD
	#melt to prepare for post hoc tukey Honest Significance Difference (HSD) Test
	mm = y.melt(id_vars="Cell Line", var_name = "Replicate", value_name = "Migration Rate Relative to Reference")
	#drops any NaNs
	mm.dropna(inplace = True)
	#builds object
	MultiComp = MultiComparison(mm['Migration Rate Relative to Reference'], mm['Cell Line'])
	#performs tukeyHSD
	#assumptions (from https://en.wikipedia.org/wiki/Tukey%27s_range_test)
		# The observations being tested are independent within and among the groups.
		# The groups associated with each mean in the test are normally distributed.
		# There is equal within-group variance across the groups associated with each mean in the test (homogeneity of variance).
	results_summary = MultiComp.tukeyhsd().summary()
	#converts to pandas df
	data = results_summary.data
	thsd = pd.DataFrame(data[1:], columns = data[0])
	return thsd


def letterBased(thsd, lines):
	#LETTER-BASED REPRESENTATION
	#based on pseudocode from
	#Piepho, H. (2004). An Algorithm for a Letter-Based Representation of All-Pairwise Comparisons.
		# Journal of Computational and Graphical Statistics, 13(2), 456-466. doi:10.1198/1061860043515
	#R uses the same algorithm, but couldn't interpret the source code
		# https://www.rdocumentation.org/packages/multcompView/versions/0.1-8/topics/multcompLetters

	#creates letter-based df
	lb = pd.DataFrame(index=lines)

	# #debug1 from Piepho 2004
	# thsd = pd.DataFrame(columns = ['group1', 'group2', 'reject'])
	# thsd.loc[0] = ['T1', 'T4', 'True']
	# lb = pd.DataFrame(index=['T1', 'T2', 'T3', 'T4'])

	# #debug2 from Piepho 2004
	# thsd = pd.DataFrame(columns = ['group1', 'group2', 'reject'])
	# thsd.loc[0] = ['T1', 'T2', 'True']
	# thsd.loc[1] = ['T1', 'T3', 'True']
	# thsd.loc[2] = ['T1', 'T4', 'True']
	# thsd.loc[3] = ['T2', 'T4', 'True']
	# lb = pd.DataFrame(index=['T1', 'T2', 'T3', 'T4'])

	# #debug3 from Piepho 2004
	# thsd = pd.DataFrame(columns = ['group1', 'group2', 'reject'])
	# thsd.loc[0] = ['T1', 'T7', 'True']
	# thsd.loc[1] = ['T1', 'T8', 'True']
	# thsd.loc[2] = ['T2', 'T4', 'True']
	# thsd.loc[3] = ['T2', 'T5', 'True']
	# thsd.loc[4] = ['T3', 'T5', 'True']
	# lb = pd.DataFrame(index=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'])

	#possible group letters
	alph = string.ascii_lowercase
	#current index in alphabet for next new group letter
	a_i = 1
	# cg_dict = dict(zip(alph, [False]*len(alph)))
	#STEP: assumption
	#initializes lb to have all cell lines in one group 'a'
	lb['a'] = True

	#iterate through significant differences (if there are any)
	for i, row in thsd.loc[thsd['reject'] == 'True'].iterrows():
		#row will have, among other columns, group1 group2 reject
			#since we rejected the null hypothesis (reject=True), we know that the cell lines in group1 and group2 must belong to different letter groups
		#INSERT STEP
		#checks all existing letter groups to see if the current statement about membership contradicts the assumed groups
			#if row is T1, T2, True, then T1 and T2 cannot be in the same group
				#however, if 
					#	 a 		 b
					#T1	 T 		 T
					#T2	 T 		 T  --> both col a and b assume T1 and T2 are in the same group, so both must be corrected
					#T3	 F 		 T
					#T4	 F 		 T
		for col in lb:
			#if the cell line in group1 and group2 are erreously assumed to be in the letter group col
			if lb.loc[row['group1'],col] and lb.loc[row['group2'],col]:
				#(1) using a new letter, duplicate the column to the wrong assertion regarding the two cell lines
				lb[alph[a_i]] = lb[col]
				#(2) in the first of the two columns, delete the letter corresponding to the one cell line
				lb.loc[row['group1'],col] = False
				#(3) in the second of the the two columns, delete the letter corresponding to the other cell line
				lb.loc[row['group2'],alph[a_i]] = False
				#updates current index in alphabet
				a_i += 1

		#ABSORPTION STEP
		#mask of changed groups from INSERT STEP
		mask = np.isin(lb.columns.values, [col, alph[a_i-1]])
		#finds preserved columns
		olds = lb.loc[:,~mask]
		#for each of the two new columns added
		for new in [col, alph[a_i-1]]:
			#for each of the old columns
			for old in olds:
				#if the new column is a subset of the old column
					#new is a subset of old if the groupings in new are redundant
						#if new says: T1, T2 are in 'a' and T3, T4 are not and old says: T1, T2, T3 are in 'a' and T4 is not, then new is redundant
					#new is not a subset of old if for at least one cell line, new says that cell line is in the group while old does not
						#if we add each (cell line value in old) - (value in new), we can test this
						#in lb, if lb[cell_line, group] = True = 1, then cell_line is in group
						#in lib, if lb[cell_line, group] = False = 0, then cell_line is not in group
					#			New
					#			0 -1
					#		  _______  --> if any cell line's value after doing old-new is -1, we know new is not a subset of old
					#Old	0 | 0 -1
					#		1 | 1  0
						#New	Old 	Old-New
					#T1	 T 		 T 		   0
					#T2	 T 		 T 		   0   --> new is a subset of old
					#T3	 F 		 T 		   1
					#T4	 F 		 F 		   0

					#New	Old 	Old-New
					#T1	 T 		 T 		   0
					#T2	 T 		 T 		   0   --> new is not a subset of old
					#T3	 F 		 T 		   1
					#T4	 T 		 F 		   -1

				#if new has not additional information compared to old (and is thus a subset)
				if -1 not in (-1*lb[new]+olds[old]).values:
					#drop redundant column
					lb.drop(new, inplace = True, axis = 1)
					#no need to check against other columns
					break
		# print(lb)

	#renames columns to use continuous list of letters
	lb.columns = list(alph[:len(lb.columns.values)])

	#generating concatenated group names for plotting
	lb['groups'] = ''
	for i, row in lb.iterrows():
		#generate concatenated group names
			#since 'a'*True = 'a' and 'a'*False = '', can just concatenate( if cell line is in group * group letter )
		lb.loc[i,'groups'] = ''.join(np.multiply(row.values[:-1],lb.columns.values[:-1]))
	# print(lb)
	return lb

def bio_volcano(d="dataframe", lfc=None, pv=None, lfc_thr=1, pv_thr=0.05, color=("green", "red"), valpha=1, geneid=None, genenames=None, gfont=8, dim=(5, 5), r=300, ar=90, dotsize=8, markerdot="o", sign_line=False, gstyle=1, show=False, figtype='png', axtickfontsize=9, axtickfontname="Arial", axlabelfontsize=9, axlabelfontname="Arial", axxlabel=None, axylabel=None, xlm=None, ylm=None):
	_x = r'$ log_{2}(Fold Change)$'
	_y = r'$ -log_{10}(P-value)$'
	color = color
	d.loc[(d[lfc] >= lfc_thr) & (d[pv] < pv_thr), 'color'] = color[0]  # upregulated
	d.loc[(d[lfc] <= -lfc_thr) & (d[pv] < pv_thr), 'color'] = color[1]  # downregulated
	d['color'].fillna('grey', inplace=True)  # intermediate
	d['logpv'] = -(np.log10(d[pv]))
	# plot
	plt.subplots(figsize=dim)
	plt.scatter(d[lfc], d['logpv'], c=d['color'], alpha=valpha, s=dotsize, marker=markerdot)
	if sign_line:
		plt.axhline(y=-np.log10(pv_thr), linestyle='--', color='#7d7d7d', linewidth=1)
		plt.axvline(x=lfc_thr, linestyle='--', color='#7d7d7d', linewidth=1)
		plt.axvline(x=-lfc_thr, linestyle='--', color='#7d7d7d', linewidth=1)
	geneplot(d, geneid, lfc, lfc_thr, pv_thr, genenames, gfont, pv, gstyle)
	if axxlabel:
		_x = axxlabel
	if axylabel:
		_y = axylabel
	general.axis_labels(_x, _y, axlabelfontsize, axlabelfontname)
	general.axis_ticks(xlm, ylm, axtickfontsize, axtickfontname, ar)
	# general.get_figure(show, r, figtype, 'volcano')

def volcano(intersections, n_replicates, timePoints, cellLines, cutoff, enrichment, label, name, display, saveFile, saveFig, fileLocation, colors):
	def highlight(s):
		#highlights entire Excel row s if that peptide meets p-value and Log2FC cutoffs
		if s['p-value'] < cutoff and s['log2FC'] >= enrichment:
			return ['background-color: #E10600; color: white']*len(s)
		elif s['p-value'] < cutoff and s['log2FC'] <= -1*enrichment:
			return ['background-color: #00239C; color: white']*len(s)
		else:
			return ['background-color: white']*len(s)

	def hl(x):
		df = x.copy()
		#initialize to blank
		df.iloc[:,:] = ''
		#goes by lfc, pval column pair
		for i in range(3,x.shape[1]-1):
			lfc = df.columns.values[i]
			pval = df.columns.values[i+1]
			#if >= enrichment, set it red
			df.loc[x[lfc] >= enrichment, [pval, lfc]] = 'background-color: #E10600; color: white'
			#if <= -enrichment, set it blue
			df.loc[x[lfc] <= -enrichment, [pval, lfc]] = 'background-color: #00239C; color: white'
		return df

	if not intersections:
		print("You must combine replicates before generating volcano plots.")
		return None

	#number of expected columns in dfs
	size = n_replicates*2*len(timePoints)
	referenceName = cellLines[-1]

	#function to loop through each cell line and time point
		#either called with an Excel writer if saveFile is True or not, which allows user to suspend program without corrupting file if saveFile is False
	#i_cell is for accessing name of cell line

	def looper(full_df = pd.DataFrame(), i_cell = 0):
		#for each others cell line
		for each in intersections:

			#for accessing name of time point
			i_point = 0

			#splits into others and reference data
			others_half = each.iloc[:,:size//2]
			reference_half = each.iloc[:,size//2:]

			#for each time point
			for i_time in range(0,size//2,n_replicates):

				cur_cell = cellLines[i_cell]
				cur_time = timePoints[i_point]

				print("volcano for {}-{}".format(cur_cell, cur_time))

				#df with data
				volcano_df = pd.DataFrame()

				#current time point data
				mut = others_half.iloc[:,i_time:i_time+n_replicates]
				reference = reference_half.iloc[:,i_time:i_time+n_replicates]
				
				#log2FC = log2(others mean) - log2(reference mean)
				volcano_df['log2FC'] = np.log2(mut.mean(axis=1)) - np.log2(reference.mean(axis=1))
				volcano_df['p-value'] = 0
				
				#for each peptide...
				for p in range(mut.shape[0]):
					#scipy.stats calculates T-test for two independent samples without assuming equal variance
					#drops all NaNs from consideration (NaNs will appear if no n_cutoff is set for combineReplicates)
					volcano_df.iloc[p,-1] = ttest_ind(mut.iloc[p,:].dropna().values, reference.iloc[p,:].dropna().values, equal_var = False)[1]

				volcano_df.reset_index(inplace = True)
				#finds gene names
				formating = lambda x: x[x.find("GN=")+3:x.find(" PE=")]
				volcano_df['Name'] = volcano_df['Master Protein Descriptions'].map(formating)

				#saves data
				if saveFile:
					selected_df = volcano_df.loc[volcano_df['p-value'] < cutoff]
					selected_df = selected_df.loc[(selected_df['log2FC'] >= enrichment) | (selected_df['log2FC'] <= -enrichment)].copy()
					selected_df = selected_df.rename(columns={'p-value':'{}-{} p'.format(cur_cell, cur_time),
															'log2FC':'{}-{} L2FC'.format(cur_cell, cur_time)})
					if full_df.empty:
						full_df = selected_df[['peptide-phosphosite', 'Master Protein Descriptions', 'Overflow Protein Descriptions','Name', '{}-{} L2FC'.format(cur_cell, cur_time), '{}-{} p'.format(cur_cell, cur_time)]]
					else:
						full_df = pd.merge(full_df,selected_df,how="outer",on=["peptide-phosphosite","Master Protein Descriptions", 'Overflow Protein Descriptions' ,"Name"])
						
					#sorts by p value
					volcano_df.sort_values(by='p-value', ascending=True, inplace = True)
					#applies highlighting style and saves log2fc and p to file
					volcano_df.style.apply(highlight, axis = 1).to_excel(writer, sheet_name = "{}-{}".format(cellLines[i_cell],timePoints[i_point]), engine = 'openpyxl')

				# raise ValueError
				#plots
				if display or saveFig:
					#labels
					if label:
						bio_volcano(volcano_df, pv_thr = cutoff, lfc_thr = enrichment, lfc = 'log2FC', pv = 'p-value', geneid = 'Name', genenames = 'deg', gstyle=2, show = display, color=colors, xlm=[-2.5,2.5,0.5], ylm=[0,4.5,0.5])
					else:
						bio_volcano(volcano_df, pv_thr = cutoff, lfc_thr = enrichment, lfc = 'log2FC', pv = 'p-value', show = display, color=colors, xlm=[-2.5,2.5,0.5], ylm=[0,4.5,0.5])
					plt.title("{} volcano for {}-{}\nreference: {}\np: {}, fc: {}".format(name, cellLines[i_cell],timePoints[i_point],referenceName,cutoff, enrichment))
					if display:
						plt.show()
					
					if saveFig:
						plt.savefig(fileLocation+"{} [{}][{}][{}] volcano for {}-{}.png".format(name, referenceName, cutoff, enrichment,cellLines[i_cell],timePoints[i_point]))
						plt.close()

					if not display:
						plt.close()

				i_point+=1
			i_cell += 1
		return full_df

	if saveFile:
		with pd.ExcelWriter(fileLocation+"{} [{}][{}][{}] volcano values.xlsx".format(name, referenceName, cutoff, enrichment)) as writer:
			full_df = looper()
			full_df.sort_values(by='Name', inplace = True)
			full_df.style.apply(hl, axis = None).to_excel(writer, sheet_name = 'Master', engine = 'openpyxl')
	else:
		looper()

#Stuff from bioinfokit ;)
class general():
	def __init__(self):
		pass

	rand_colors = ('#a7414a', '#282726', '#6a8a82', '#a37c27', '#563838', '#0584f2', '#f28a30', '#f05837',
				   '#6465a5', '#00743f', '#be9063', '#de8cf0', '#888c46', '#c0334d', '#270101', '#8d2f23',
				   '#ee6c81', '#65734b', '#14325c', '#704307', '#b5b3be', '#f67280', '#ffd082', '#ffd800',
				   '#ad62aa', '#21bf73', '#a0855b', '#5edfff', '#08ffc8', '#ca3e47', '#c9753d', '#6c5ce7')

	def get_figure(show, r, figtype, fig_name):
		if show:
			plt.show()
		else:
			plt.savefig(fig_name+'.'+figtype, format=figtype, bbox_inches='tight', dpi=r)
		plt.close()

	def axis_labels(x, y, axlabelfontsize=None, axlabelfontname=None):
		plt.xlabel(x, fontsize=axlabelfontsize, fontname=axlabelfontname)
		plt.ylabel(y, fontsize=axlabelfontsize, fontname=axlabelfontname)
		# plt.xticks(fontsize=9, fontname="sans-serif")
		# plt.yticks(fontsize=9, fontname="sans-serif")

	def axis_ticks(xlm=None, ylm=None, axtickfontsize=None, axtickfontname=None, ar=None):
		if xlm:
			plt.xlim(left=xlm[0], right=xlm[1])
			plt.xticks(np.arange(xlm[0], xlm[1], xlm[2]),  fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
		else:
			plt.xticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)

		if ylm:
			plt.ylim(bottom=ylm[0], top=ylm[1])
			plt.yticks(np.arange(ylm[0], ylm[1], ylm[2]),  fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
		else:
			plt.yticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)

	def depr_mes(func_name):
		print("This function is deprecated. Please use", func_name )
		print("Read docs at https://reneshbedre.github.io/blog/howtoinstall.html")

def geneplot(d, geneid, lfc, lfc_thr, pv_thr, genenames, gfont, pv, gstyle):
	if genenames is not None and genenames == "deg":
		for i, row in d.loc[d['color'] != 'grey'].iterrows():
			if gstyle==1:
				plt.text(row[lfc], row['logpv'], row[geneid],
							  fontsize=gfont)
			elif gstyle==2:
				plt.annotate(row[geneid], xy=(row[lfc], row['logpv']),
							 xycoords='data', xytext=(5, -15), textcoords='offset points', size=6,
							 bbox=dict(boxstyle="round", alpha=0.1),
							 arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1, relpos=(0, 0)))
			else:
				print("Error: invalid gstyle choice")
				sys.exit(1)
