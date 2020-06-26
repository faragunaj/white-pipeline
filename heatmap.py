import classes
import modules

import matplotlib.pyplot as plt


#	#	DATA SETUP	#	#
#list of file locations for replicates
	#first list: BR1
	#second list: BR2
	#third list: BR3
#all locations are in order of cell lines (2934,2935,2936,2937,2938,3126,3127,3138)
#can be in different order as long as
	#order is consistent across all replicates,
	#order is consistent with cell_lines variable defined below,
	#and WT IS LAST (my code assumes this when performing calculations)
locations = [
	["../Excel files preprocessed/csv/2nM/2934_2nM_BR1.csv",
	"../Excel files preprocessed/csv/2nM/2935_2nM_BR1.csv",
	"../Excel files preprocessed/csv/2nM/2936_2nM_BR1.csv",
	"../Excel files preprocessed/csv/2nM/2937_2nM_BR1.csv",
	"../Excel files preprocessed/csv/2nM/2938_2nM_BR1.csv",
	"../Excel files preprocessed/csv/2nM/3126_2nM_BR1.csv",
	"../Excel files preprocessed/csv/2nM/3127_2nM_BR1.csv",
	"../Excel files preprocessed/csv/2nM/3138_2nM_BR1.csv"],
	["../Excel files preprocessed/csv/2nM/2934_2nM_BR2.csv",
	"../Excel files preprocessed/csv/2nM/2935_2nM_BR2.csv",
	"../Excel files preprocessed/csv/2nM/2936_2nM_BR2.csv",
	"../Excel files preprocessed/csv/2nM/2937_2nM_BR2.csv",
	"../Excel files preprocessed/csv/2nM/2938_2nM_BR2.csv",
	"../Excel files preprocessed/csv/2nM/3126_2nM_BR2.csv",
	"../Excel files preprocessed/csv/2nM/3127_2nM_BR2.csv",
	"../Excel files preprocessed/csv/2nM/3138_2nM_BR2.csv"],
	["../Excel files preprocessed/csv/2nM/2934_2nM_BR3.csv",
	"../Excel files preprocessed/csv/2nM/2935_2nM_BR3.csv",
	"../Excel files preprocessed/csv/2nM/2936_2nM_BR3.csv",
	"../Excel files preprocessed/csv/2nM/2937_2nM_BR3.csv",
	"../Excel files preprocessed/csv/2nM/2938_2nM_BR3.csv",
	"../Excel files preprocessed/csv/2nM/3126_2nM_BR3.csv",
	"../Excel files preprocessed/csv/2nM/3127_2nM_BR3.csv",
	"../Excel files preprocessed/csv/2nM/3138_2nM_BR3.csv"]]

#technical replicate, same as above
technicalReplicate = ["../Excel files preprocessed/csv/2nM/2934_2nM_BR2TR.csv",
	"../Excel files preprocessed/csv/2nM/2935_2nM_BR2TR.csv",
	"../Excel files preprocessed/csv/2nM/2936_2nM_BR2TR.csv",
	"../Excel files preprocessed/csv/2nM/2937_2nM_BR2TR.csv",
	"../Excel files preprocessed/csv/2nM/2938_2nM_BR2TR.csv",
	"../Excel files preprocessed/csv/2nM/3126_2nM_BR2TR.csv",
	"../Excel files preprocessed/csv/2nM/3127_2nM_BR2TR.csv",
	"../Excel files preprocessed/csv/2nM/3138_2nM_BR2TR.csv"]

#phenotypic replicate_s_
phenotypicMeasurement = ['../phenotypic data/PhenoClass 1.csv',
	'../phenotypic data/PhenoClass 3.csv',
	'../phenotypic data/PhenoClass 4.csv']

cell_lines = ['2934','2935','2936','2937','2938','3126','3127','3138']

time_points = [0, 30, 1, 2, 5]

second_time_points = [0, 30, 60, 120, 300]


#	#	EXPERIMENT CREATION	#	#
exp = classes.Experiment(locations, cell_lines, time_points, second_time_points, case_insensitive = True, names = ["BR1","BR2","BR3"], fileLocation = 'output/')
exp.addTechnicalReplicate(technicalReplicate,1)
exp.combineReplicates()
exp.addPhenotypicMeasurement(phenotypicMeasurement, phenotypicType = 'Migration Rate')
# print(exp)
# exp.setReference('2934')
# print(exp)

#	#	ANALYSIS	#	#
# exp.heatmapToReference(normalization = 'ownbasal')
# exp.heatmapToReference(normalization = 'reftime')
# exp.pcaToReference()
# exp.setReference('2934')
# exp.pcaToReference()
# plt.show()
# exp.heatmap(normalization = 'reftime')
# exp.heatmap(normalization = 'refbasal')
# exp.heatmap(normalization = 'ownbasal')

# exp.simpleCorr()
# exp.correlationToReference()
# exp.correlationToSelf()
# exp.loadings()
# exp[0].pca()
# plt.show()


exp[0].pca(display=False)
# exp.replicatePlot(display=False)
# exp.groupPlot()
# exp.phenotypicMeasurements['Migration Rate'].replicatePlot(relativeToReference = True, display = False)
plt.show()
