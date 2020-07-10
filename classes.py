import modules

import re
import math
import string
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)


class Experiment:
    """Store entire experiment data.

    Attributes:
        experimentalReplicates (list of ExperimentalReplicate): Experimental replicates
        combinedReplicates (list of DataFrame): Averaged peptide values across experimental replicates based on user-provided thresholds, independently for each cell line
        combinedReplicatesData (list of DataFrame): Averaged peptide values, as well as n and std dev metrics
        separatedCombinedReplicatesData (list of DataFrame): separate peptide values for each replicate provided the peptide meets n and std cutoffs
        experimentFullIntersection (DataFrame): Inner pd.merge() of combined replicate data
        experimentReferenceIntersections (list of DataFrame): Separate inner pd.merge() for each cell line of combined replicate data to reference
        experimentSeparatedFullIntersection (DataFrame): Inner pd.merge() of separated combined replicate data
        experimentSeparatedReferenceIntersections (list of DataFrame): Separate inner pd.merge() for each cell line of separated combined replicate data to reference

        cellLines (list of int or str): Names of cell lines
        timePoints (list of int): Time points
        secondTimePoints (list of int): Time points in seconds
        caseInsensitive (bool): Case insensitivity of overall experiment, default True and only False if all replicates are False
        names (list of str): Names of experimental replicates
        colors (list of str): default colors to use for plotting cell lines
        fileLocation (str): default file location to save plots to

        n_cutoff (int): Minimum number of replicates peptide must be found in to be included in combinedReplicates
        std_cutoff (int): Maximum std dev for abundance of peptide across all time points to be included in combinedReplicates

    """
    experimentalReplicates = []
    combinedReplicates = None
    combinedReplicatesData = None
    separatedCombinedReplicatesData = None
    experimentFullIntersection = None
    experimentReferenceIntersections = None
    experimentSeparatedFullIntersection = None
    experimentSeparatedReferenceIntersections = None

    phenotypicMeasurements = {}

    cellLines = []
    timePoints = []
    secondTimePoints = []
    caseInsensitive = True
    names = []
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    fileLocation = ''

    n_cutoff = 0
    std_cutoff = 0

    def __init__(self, replicates, cell_lines=None, time_points=None, second_time_points=None, case_insensitive=None, names=None, colors = None, fileLocation = ''):
        """Initializes Experiment from file locations or existing ExperimentalReplicate.

        Notes:
            Accepts several replicate formats
                List of str: file locations for single experimental replicate
                list of list of str: file locations for multiple experimental replicates
                ExperimentalReplicate: single ExperimentalReplicate
                list of ExperimentalReplicate: multiple ExperimentalReplicate
            Creates new ExperimentalReplicate objects for each replicate

        Args:
            replicates (list of str, list of list of str, ExperimentalReplicate, list of ExperimentalReplicate): replicates to be stored in Experiment
            cell_lines (list of int): cell line identifiers
            time_points (list of int): time points
            second_time_points (list of int): time points in seconds
            case_insensitive (bool): case-insensitivity of peptide-phosphosite labels for comparisons 
            names (list of str): names of replicates
            colors (list of str): colors to use for cell lines
            fileLocation (str): file location to save to

        """

        #replicates is a single ExperimentalReplicate object
        if type(replicates) == ExperimentalReplicate:
            self.experimentalReplicates = [replicates]
            self.cellLines = replicates.cellLines
            self.timePoints = replicates.timePoints
            self.secondTimePoints = replicates.secondTimePoints
            self.colors = replicates.colors
            self.fileLocation = fileLocation
            if case_insensitive != None:
                self.caseInsensitive = case_insensitive
            else:
                self.caseInsensitive = replicates.caseInsensitive

        #replicates is a list of ExperimentalReplicate objects
        elif type(replicates[0]) == ExperimentalReplicate:
            self.experimentalReplicates = replicates
            self.cellLines = replicates[0].cellLines
            self.timePoints = replicates[0].timePoints
            self.secondTimePoints = replicates[0].secondTimePoints
            self.colors = replicates.colors
            self.fileLocation = fileLocation
            #if overall experimental case insensitivity is proviuded, makes experimental replicates with it
            if case_insensitive != None:
                self.caseInsensitive = case_insensitive
            #otherwise, default overall experimental case insensitivity is False, unless at least one replicate is case sensitive
            else:
                self.caseInsensitive = False
                for each in replicates:
                    if each.caseInsensitive:
                        self.caseInsensitive = True
                        break

        #replicates is a list of locations that need to be used to create a list of one ExperimentalReplicate object
        elif type(replicates[0]) == str:
            #if colors have been passed
            if colors:
                self.colors = colors
            #if there are too many cell lines for the default colors
            elif len(self.cellLines) > len(self.colors)+1:
                print("ERROR: Too many cell lines for the default colors: {}\nConsider passing a list of colors into the experiment".format(self.colors+['tab:gray']))
                exit(0)
            #if there are fewer or just enough cell lines for the colors
            else:
                #adjusts the number of colors so gray is always wt
                self.colors = self.colors[:(len(cell_lines)-1)]
                self.colors.append('tab:gray')

            #if overall experimental case insensitivity is provided, makes experimental replicates with it
            if case_insensitive != None:
                self.experimentalReplicates = [ExperimentalReplicate(replicates, cell_lines, time_points, second_time_points, case_insensitive, names, self.colors, fileLocation)]
                self.caseInsensitive = case_insensitive
            #defaults to case insensitivte
            else:
                self.experimentalReplicates = [ExperimentalReplicate(replicates, cell_lines, time_points, second_time_points, replicate_name=names, colors = self.colors, fileLocation = fileLocation)]
                self.caseInsensitive = True
            self.cellLines = cell_lines
            self.timePoints = time_points
            self.secondTimePoints = second_time_points
            self.fileLocation = fileLocation

        ##replicates is a list of lists of locations that need to be used to create a list of multiple ExperimentalReplicate objects
        elif type(replicates[0]) == list:
            if names == None:
                names = ["Replicate {}".format(i) for i in range(len(replicates))]
            if colors:
                self.colors = colors
            #if there are too many cell lines for the default colors
            elif len(self.cellLines) > len(self.colors)+1:
                print("ERROR: Too many cell lines for the default colors: {}\nConsider passing a list of colors into the experiment".format(self.colors+['tab:gray']))
                exit(0)
            #if there are fewer or just enough cell lines for the colors
            else:
                #adjusts the number of colors so gray is always wt
                self.colors = self.colors[:(len(cell_lines)-1)]
                self.colors.append('tab:gray')

            if case_insensitive != None:
                self.experimentalReplicates = [ExperimentalReplicate(replicates[i], cell_lines, time_points, second_time_points, case_insensitive, names[i], self.colors, fileLocation) for i in range(len(replicates))]
                self.caseInsensitive = case_insensitive
            else:
                self.experimentalReplicates = [ExperimentalReplicate(replicates[i], cell_lines, time_points, second_time_points, replicate_name=names[i], colors = self.colors, fileLocation = fileLocation) for i in range(len(replicates))]
                self.caseInsensitive = True
            self.cellLines = cell_lines
            self.timePoints = time_points
            self.secondTimePoints = second_time_points
            self.fileLocation = fileLocation

        else:
            raise TypeError

        #performs fullIntersection and referenceIntersections for each ExperimentalReplicate stored in Experiment
        for each in self.experimentalReplicates:
            each.fullIntersection()
            each.referenceIntersections()

    def __str__(self):
        """Human readable string describing experiment.

        """
        if self.combinedReplicates:
            printout = "\nCOMBINED MS REPLICATES WITH n = " + str(self.n_cutoff) + " and std dev = " + str(self.std_cutoff) + "\nCell Lines: " + str(self.cellLines).strip("[]") + "\nSize: " + str([self.combinedReplicates[i].shape[0] for i in range(len(self.combinedReplicates))]).strip("[]") + "\nIntersection Size: " + str(self.experimentFullIntersection.shape[0]) + "\n"
            printout += "\n"
        else:
            printout = ""
        printout += "\n".join([str(each) for each in self.experimentalReplicates]).strip("[]")
        if self.phenotypicMeasurements:
            printout += "\n"
            printout += "".join([str(each) for each in self.phenotypicMeasurements.values()]).strip("[]")

        return printout

    def __iter__(self):
        """Provides support for iterating over experimental replicates.

        Examples:
            [each for each in experiment] --> each replicate in experiment

        """
        self.n = 0
        return self

    def __next__(self):
        """Provides support for iterating over experimental replicates.

        """
        if self.n < len(self.experimentalReplicates):
            self.n += 1
            return self.experimentalReplicates[self.n - 1]
        else:
            raise StopIteration

    def __getitem__(self, index):
        """Provides support for indexing experiment for replicates.

        Examples:
            experiment[0]

        """
        return self.experimentalReplicates[index]

    def save(self, name, replicates=False):
        """Saves contents of Experiment to multiple excel files.

        Args:
            name (str): location and common name of all files
            replicates (bool): whether to also save replicates

        Notes:
            Experiment saves as "name.xlsx."
            Replicates save as "name replicateName.xlsx"

        """

        #saves replicates
        if replicates:
            for each in self:
                replicateName = name + " " + each.replicateName + ".xlsx"
                writer = pd.ExcelWriter(replicateName)
                each.save(writer)

        #saves Experiment
        experimentName = name + ".xlsx"
        experimentWriter = pd.ExcelWriter(experimentName)

        with experimentWriter as writer:
            #saves combinedReplicates
            try:
                for i in range(len(self.combinedReplicates)):
                    try:
                        self.combinedReplicates[i].to_excel(writer, sheet_name=str(self.cellLines[i]))
                    except IndexError:
                        pass
                    try:
                        self.combinedReplicatesData[i].to_excel(writer,sheet_name=str(self.cellLines[i]) + " data")
                    except IndexError:
                        pass
            except AttributeError:
                print("ERROR: Experiment not combined")
                pass
            #saves experimentReferenceIntersections
            for i in range(len(self.experimentReferenceIntersections)):
                try:
                    self.experimentReferenceIntersections[i].to_excel(writer, sheet_name=str(self.cellLines[i]) + " to reference")
                except AttributeError:
                    print("ERROR: Experiment reference intersections not calculated")
                    break
                except IndexError:
                    pass
            #saves experimentFullIntersection
            try:
                self.experimentFullIntersection.to_excel(writer, sheet_name="Full Intersection")
            except AttributeError:
                print("ERROR: Experiment full intersection not calculated")
                pass
            except IndexError:
                pass

    def fullIntersection(self, data):
        """Executes an inner pd.merge() on a list of dfs.

        Args:
            data (list of dfs): data to combine, usually experiment's combined replicate data

        """
        # try:
        return modules.fullIntersection(data, self.caseInsensitive)
        # except TypeError:
        #     print("ERROR: Combine replicates before calling fullIntersection")
        #     return None

    def referenceIntersections(self, data, combine=False,):
        """Executes a separate inner pd.merge() to reference for each cell line on a list of dfs.

        Args:
            data (list of dfs): data to combine, usually combined replicate data
            combine (bool): Whether to include reference data at the end of returned list of DataFrames comparing others to reference

        """
        try:
            return modules.separateIntersections(data[:-1], data[-1],self.caseInsensitive, combine)
        except TypeError:
            print("Combine replicates before calling referenceIntersections")
            return None

    def addTechnicalReplicate(self, replicate, i):
        """Adds a technical replicate to the ith experimental replicate (Python is 0 indexed).

        Notes:
            Technical replicate is added via outer pd.merge, so all peptides in at least the experimenal or technical replicate are included.
            Overlapping peptide abundances are averaged. Ranges are not tracked because they are not considered separate replicates.

        Args:
            replicate (list of str, ExperimentalReplicate): List of locations to be imported, or ExperimentalReplicate object to be used as technical replicate
            i (int): Experimental replicate to add technical replicate to

        """
        iReplicate = self.experimentalReplicates[i]
        technicalReplicate = ExperimentalReplicate(replicate, iReplicate.cellLines, iReplicate.timePoints, iReplicate.caseInsensitive)
        iReplicate.addTechnicalReplicate(technicalReplicate)

    def addPhenotypicMeasurement(self, locations, phenotypicType='Generic'):
        """Adds a set of phenotypic measurements of the same type (e.g. migration rates).

        Notes:
            Used to add all replicates of one measurement type at once.

        Args:
            locations (list of str, str): list or single location of data to be imported
            phenotypicType (str): identifier for phenotypic measurement type

        """
        self.phenotypicMeasurements[phenotypicType] = PhenotypicMeasurement(locations, phenotypicType, self.cellLines, self.fileLocation)

    def combineReplicates(self, n_cutoff=0, std_cutoff=float('inf')):
        """Combines experimental replicates into a list of dfs and performs fullIntersection and referenceIntersections.

        Args:
            n_cutoff (int): minimum number of replicates a peptide shows up in to be included
            std_cutoff (int): maximum std dev across replicates for a peptide to be included

        Examples:
            combineReplicates(n_cutoff=2, std_cutoff=0.5)

        Notes:
            Uses modules.combineReplicates().

        """
        self.n_cutoff = n_cutoff
        self.std_cutoff = std_cutoff
        #trivial case with 1 replicate
        if len(self.experimentalReplicates) == 1:
            self.combinedReplicates = self.experimentalReplicates[0].cellData
            
            #computes intersections
            try:
                self.experimentFullIntersection = self.fullIntersection(self.combinedReplicates)
                self.experimentReferenceIntersections = self.referenceIntersections(self.combinedReplicates)
            except AttributeError:
                print("ERROR: combinereplicates cutoffs are too stringent: there is no data left.")

        else:
            self.combinedReplicatesData, self.combinedReplicates, self.separatedCombinedReplicatesData = modules.combineReplicates(self, n_cutoff, std_cutoff) 

            #computes intersections
            try:
                self.experimentFullIntersection = self.fullIntersection(self.combinedReplicates)
                self.experimentReferenceIntersections = self.referenceIntersections(self.combinedReplicates)
                self.experimentSeparatedFullIntersection = self.fullIntersection(self.separatedCombinedReplicatesData)
                self.experimentSeparatedReferenceIntersections = self.referenceIntersections(self.separatedCombinedReplicatesData)
            except AttributeError:
                print("ERROR: combineReplicates cutoffs are too stringent: there is no data left.")

    def setReference(self, line):
        """Updates REFERENCE cell line to a new one.

        Args:
            line (str): new cell line to be set to REFERENCE

        Notes:
            Also redoes each replicate's reference intersections and the overall experimental intersections

        """
        #updates experimental lines
        original = self.cellLines.copy()
        updatedIndices = list(range(len(self.cellLines)))
        ind = self.cellLines.index(line)
        self.cellLines.append(self.cellLines.pop(ind))
        self.colors.append(self.colors.pop(ind))
        updatedIndices.append(updatedIndices.pop(ind))

        #updates replicates lines and redoes reference intersections
        for each in self.experimentalReplicates:
            each.setReference(updatedIndices)
            each.referenceIntersections()
        #redoes combining replicates to recalculate intersections on an experimental level
        if self.combinedReplicates:
            self.combineReplicates(n_cutoff = self.n_cutoff, std_cutoff = self.std_cutoff)
        #resets phenotypicMeasurements too
        for each in self.phenotypicMeasurements.values():
            each.setReference(line, updatedIndices)

        print("New REFERENCE set to {}".format(line))

    def setOutputLocation(self, location):
        '''Updates output file location.

        Args:
            location (str): new output file location
        '''
        self.fileLocation = location

        for each in self.experimentalReplicates:
            each.fileLocation = location
        for each in self.phenotypicMeasurements.values():
            each.fileLocation = location

        print("New output set to {}".format(location))

    def log2Comparison(self, keyword=None, threshold=0, normalizeToBasal=False, tail="both", title=None):
        """Compares and graphs percentage of peptides in individual others that show a specified log2 fold difference compared to reference.

        Args:
            keyword (str): optional keyword to include to limit peptides considered in comparison
            threshold (str): log2 cutoff threshold
            normalizeToBasal (bool): whether to subtract all others time points by each others basal (0), and subtract all reference time points by reference basal (0)
            tail (str): tail of threshold to consider
                "both": peptides with log2 >= threshold and log2 <= -1*threshold
                "lower": peptides with log2 <= -1*threshold
                "upper": peptides with log2 >= threshold

        Notes:
            Each others is separately overlapped with reference when performing comparison to preserve maximum data, so not all peptides are expressed across all points.

        Returns:
            comparison (list of list of df): peptides meeting threshold as a list of replicates, each of which is a list of cell line DataFrames
            originalSizes (list of list of int): overlapping peptides for each others to reference comparison, useful to see how many peptides each percentage represents


        """
        comparison = []
        originalSizes = []

        f = plt.figure(random.randint(1, 1000))

        for i in range(len(self.experimentalReplicates)):
            data, sizes = self[i].log2Comparison(keyword, threshold, normalizeToBasal, tail, replicateNumber=i, fig=f, givenTitle=title)
            comparison.append(data)
            originalSizes.append(sizes)

        return comparison, originalSizes

    def peptidePicker(self, threshold=0, normalizeToBasal=False, tail="both", trajectory=None):
        """Selects peptides that meet a given trajectory compared to a threshold across individual cell lines.

        Args:
            threshold (str): log2 cutoff threshold
            normalizeToBasal (bool): whether to subtract all others time points by each others basal (0), and subtract all reference time points by reference basal (0)
            tail (str): tail of threshold to consider
                "both": peptides with log2 >= threshold and log2 <= -1*threshold
                "lower": peptides with log2 <= -1*threshold
                "upper": peptides with log2 >= threshold
            trajectory (list of bool/str): trajectory of peptides compared to threshold, default value is [True, True, True, True, True]
                True: peptide must meet threshold at this time point to be included
                False: peptide cannot meet threshold at this time point to be included
                "All": time point does not enforce a threshold (all peptides pass this time point)

        Notes:
            Each others is separately overlapped with reference when performing comparison to preserve maximum data, so not all peptides are expressed across all points.

        Examples:
            exp.peptidePicker(threshold = 1.5, tail = 'upper', trajectory = ["All", "All", False, False, True])
                Finds peptides such that peptide_1_minute < 1.5, peptide_2_minute < 1.5, and peptide_5_minute.
                0 and 30 second time points do not enforce threshold.

        """
        comparison = []
        originalSizes = []

        if trajectory == None:
            trajectory = [True] * len(self.timePoints)

        for i in range(len(self.experimentalReplicates)):
            data, sizes = self[i].peptidePicker(threshold, normalizeToBasal, tail, trajectory)
            comparison.append(data)
            originalSizes.append(sizes)

        return comparison, originalSizes

    def heatmap(self, name="", display=True, saveFile = False, saveFig = False, fileLocation="", fullscreen=False, normalization='refbasal'):
        """Plots a heatmap comparing all cell lines at once.

        Args:
            name (str): plot title
            display (bool): whether to display a graph
            saveFile (bool): whether to save data to Excel files
            saveFig (bool): whether to save figs
            fileLocation (str): file location to save to
            fullscreen (bool): whether to display graph fullscreen
            normalization (str): normalization scheme to use, default refbasal but can use refbasal, reftime, ownbasal

        """
        if fileLocation == '':
            fileLocation = self.fileLocation
        # try:
        modules.heatmap(self.experimentFullIntersection.copy(), self.cellLines, self.timePoints, name, display, saveFile, saveFig, fileLocation, fullscreen, normalization)
        # except AttributeError:
        #     print("ERROR: Combine replicates first.")

    def heatmapToReference(self, name="", display=True, saveFile = False, saveFig = False, fileLocation="", fullscreen = False, normalization='refbasal'):
        """Plots multiple heatmaps comparing cell lines to reference.

        Args:
            name (str): plot title
            display (bool): whether to display a graph
            saveFile (bool): whether to save data to Excel files
            saveFig (bool): whether to save figs
            fileLocation (str): file location to save to
            fullscreen (bool): whether to display graph fullscreen
            normalization (str): normalization scheme to use, default refbasal but can use refbasal, reftime, ownbasal

        """
        if fileLocation == '':
            fileLocation = self.fileLocation
        try:
            modules.heatmapToReference(self.experimentReferenceIntersections.copy(), self.cellLines, self.timePoints, name, display, saveFile, saveFig, fileLocation, fullscreen, normalization)
        except AttributeError:
            print("ERROR: Combine replicates first.")

    def pcaToReference(self, name="", display=True, saveFile = False, saveFig = False, fileLocation="", fullscreen=False):
        """Plots separate PCAs comparing each others to reference intersection individually.

        Args:
            name (str): plot title
            display (bool): whether to display graph
            saveFile (bool): whether to save PCA loadings to Excel files
            saveFig (bool): whether to save figs
            fileLocation (str): file location to save to
            fullscreen (bool): whether to display graph fullscreen

        Notes:
            Useful tutorial https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60


        """
        if fileLocation == '':
            fileLocation = self.fileLocation
        # try:
        return modules.pcaToReference(self.experimentReferenceIntersections.copy(), self.cellLines, self.timePoints, self.secondTimePoints, name, display, saveFile, saveFig, fileLocation, fullscreen, self.colors)
        # except AttributeError:
        #     print("ERROR: Combine replicates first.")

    def pca(self, name="", display=True, saveFile = False, saveFig = False, fileLocation="", fullscreen=False):
        """Plots one PCA comparing all others and reference at once.

        Args:
            name (str): plot title
            display (bool): whether to display graph
            saveFile (bool): whether to save PCA loadings to Excel files
            saveFig (bool): whether to save figs
            fileLocation (str): file location to save to
            fullscreen (bool): whether to display graph fullscreen

        """
        if fileLocation == '':
            fileLocation = self.fileLocation
        try:
            return modules.pca(self.experimentFullIntersection.copy(), self.cellLines, self.timePoints, self.secondTimePoints, name, display, saveFile, saveFig, fileLocation, fullscreen, self.colors)
        except AttributeError:
            print("ERROR: Combine replicates first.")

    def volcano(self, cutoff = 0.05, enrichment = 1, label = False, name="", display=False, saveFile = False, saveFig = False, fileLocation = "", colors = ("#E10600FF","#00239CFF")):
        """Plots a volcano plot for each time point for each cell line. Also saves an Excel file with p-values and L2FC.

        Args:
            cutoff (int): p-value cutoff for significance and color highlight, default is 0.05
            label (bool): whether to display significant gene names, default is False
            name (str): experiment identifier to include in all plots
            display (bool): whether to display all plots, default is False
            saveFile (bool): whether to save Excel file, default is False
            saveFig (bool): whether to save plot as a PNG, default is False
            fileLocation (str): directory and file prefix for Excel file
            colors (list): colors for up and downregulated peptides

        Examples:
            exp.volcano(display=True)
                Displays all graphs with p-value cutoff of 0.05
            exp.volcano(saveFile=True, fileLocation = "../JOE'S ")
                Does not display any graphs and saves Excel file to the directory above the current one as "JOE's volcanoe values.xlsx"

        """
        if fileLocation == '':
            fileLocation = self.fileLocation
        test = modules.volcano(self.experimentSeparatedReferenceIntersections.copy(), len(self.experimentalReplicates), self.timePoints, self.cellLines, cutoff, enrichment, label, name, display, saveFile, saveFig, fileLocation, colors)
        return test

    def correlationToReference(self, name="", display=True, saveFile = False, saveFig = False, fileLocation = "", normalization = 'refbasal'):
        '''For each cell line, computes the correlation of all peptide trajectories to all reference peptide trajectories.

        Args:
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
            MutantPeptide.B will be ....            ReferencePeptide.A, .B, ... Z
            ...

            For trajectory plots, the shaded regions are 95% confidence intervals and the solid line is the mean.
        '''
        if fileLocation == '':
            fileLocation = self.fileLocation
        modules.correlationToReference(self.experimentReferenceIntersections.copy(), self.timePoints, self.secondTimePoints, self.cellLines, name, display, saveFile, saveFig, fileLocation, normalization)
        
    def correlationToSelf(self, name="", display=True, saveFile = False, saveFig = False, fileLocation = "", normalization = 'ownbasal'):
        '''For each cell line, computes the correlation of all peptide trajectories to all peptide trajectories.

        Args:
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
            MutantPeptide.B will be ....            MutantPeptide.A, .B, ... Z
            ...

            For trajectory plots, the shaded regions are 95% confidence intervals and the solid line is the mean.
        '''
        if fileLocation == '':
            fileLocation = self.fileLocation
        modules.correlationToSelf(self.combinedReplicates.copy(), self.timePoints, self.secondTimePoints, self.cellLines, name, display, saveFile, saveFig, fileLocation, normalization)

    def correlationToReferenceDiagonal(self, name = '', display = True, saveFile = False, saveFig = False, fileLocation = '', normalization = 'refbasal', bins = None, kde = False):
        '''For each cell line, computes the correlation of each peptide's trajectory to the corresponding reference peptide trajectory.

        Args:
            name (str): name to include in plots
            display (bool): whether to display plots
            saveFile (bool): whether to save data to Excel files
            saveFig (bool): whether to save figs
            fileLocation (str): location to save to
            normalization (str): normalization scheme, either ownbasal, reftime, or refbasal
            bins (int): number of histogram bins to use
            kde (bool): whether to plot gaussian kernal density estimate or not (https://seaborn.pydata.org/generated/seaborn.distplot.html)

        Notes:
            WARNING: kde plots a smoothed, more interpretable histogram where the y axis is no longer the true number of peptides in the histogram. It's unclear whether this plot is an accurate reflection of trends between cell lines.

            Computes the Pearson Coefficient (R) between two peptide trajectories over time.
                -1 is a negative linear relationship
                0 is no linear relationship
                +1 is a positive linear relationship

            If two peptides have trajectories over time that are positive linear scalings of each other, they will have a Pearson R close to +1

            MutantPeptide.A will be correlated with ReferencePeptide.A ONLY
            MutantPeptide.B will be correlated with ReferencePeptide.B ONLY
            ...
            ...
        '''
        if fileLocation == '':
            fileLocation = self.fileLocation
        modules.correlationToReferenceDiagonal(self.experimentReferenceIntersections.copy(), self.timePoints, self.secondTimePoints, self.cellLines, name, display, saveFile, saveFig, fileLocation, normalization, self.colors, bins, kde)

    def groupPlot(self, key = None, relativeToReference = False, display = True, name = '', fileLocation = ''):
        '''Plots one type of phenotypic data for the experiment, including std. dev. error bars and letter-based groups based on Tukey's HSD.

        Args:
            key (str): identifier of phenotypic replicate to plot
            relativeToFerence (bool): whether to normalize to reference phenotype or not

        '''
        if fileLocation == '':
            fileLocation = self.fileLocation
        if not key:
            if len(self.phenotypicMeasurements) == 1:
                key = list(self.phenotypicMeasurements.keys())[0]
                print("Assuming use of only phenotypic measurement, {}".format(key))
            else:
                print("Specify the name of a phenotypic measurement from {}".format(self.phenotypicMeasurements.keys()))
                exit(0)

        try:
            return self.phenotypicMeasurements[key].groupPlot(relativeToReference, name, display, fileLocation)
        except KeyError:
            print("No such phenotypic measurement found.")
            exit(0)

    def replicatePlot(self, key = None, relativeToReference = False, display = True, name = '', fileLocation = ''):
        '''Plots one phenotypic measurement, indicating the individual phenotypic replicates' data.

        Args:
            relativeToReference (bool): whether to normalize to reference or plot as a separate cell line.
            name (str): name to include in plots
            display (bool): whether to display plots
            fileLocation (str): location to save to

        Notes:
            Useful for debugging whether one phenotypic measurement should be excluded.

        '''
        if fileLocation == '':
            fileLocation = self.fileLocation
        if not key:
            if len(self.phenotypicMeasurements) == 1:
                key = list(self.phenotypicMeasurements.keys())[0]
                print("Assuming use of only phenotypic measurement, {}".format(key))
            else:
                print("Specify the name of a phenotypic measurement from {}".format(self.phenotypicMeasurements.keys()))
                exit(0)

        try:
            return self.phenotypicMeasurements[key].replicatePlot(relativeToReference, name, display, fileLocation)
        except KeyError:
            print("No such phenotypic measurement found.")
            exit(0)


class ExperimentalReplicate:
    """Store entire experiment data.

    Attributes:
        cellData (list of df): cell data separated into different cell lines
        cellLines (list of int or str): Names of cell lines
        timePoints (list of int): Time points
        secondTimePoints (list of int): Time points in seconds
        caseInsensitive (bool): Case insensitivity of overall experiment, default True and only False if all replicates are False
        replicateName (str): Name of replicate
        fileLocation (str): Default file location for saving plots

        cellFullIntersection (DataFrame): Inner pd.merge() of all cell lines
        cellReferenceIntersections (list of DataFrame): Separate inner pd.merge() for each cell line to reference

    Notes:
        Last cell line in cellLines is assumed to be reference!

    """
    replicateName = ""

    cellData = []
    cellFullIntersection = None
    cellReferenceIntersections = None

    cellLines = []
    timePoints = []
    secondTimePoints = []
    caseInsensitive = True
    fileLocation = ''

    def __init__(self, locations, cell_lines=None, time_points=None, second_time_points=None, case_insensitive=True, replicate_name=None, colors = None, fileLocation = ''):
        #existing ExperimentalReplicate
        if type(locations) == ExperimentalReplicate:
            self.cellData = locations.cellData
            self.cellLines = locations.cellLines
            self.timePoints = locations.timePoints
            self.secondTimePoints = locations.secondTimePoints
            self.caseInsensitive = locations.caseInsensitive
            self.replicateName = locations.replicateName
            self.colors = colors
            self.fileLocation = fileLocation

            self.cellFullIntersection = locations.cellFullIntersection
            self.cellReferenceIntersections = locations.cellReferenceIntersections
        #new list of locations to read in
        else:
            self.cellData = [0] * len(locations)
            self.cellLines = cell_lines
            self.timePoints = time_points
            self.secondTimePoints = second_time_points
            self.caseInsensitive = case_insensitive
            self.colors = colors
            self.fileLocation = fileLocation
            if replicate_name:
                self.replicateName = replicate_name

            for i in range(len(locations)):
                #skips header
                self.cellData[i] = pd.read_csv(locations[i], skiprows=0)
                #names columns to pep.-phos., MPD, and then cellLine-timePoint
                self.cellData[i].columns = ['peptide-phosphosite', 'Master Protein Descriptions']+['{}-{}'.format(self.cellLines[i],t) for t in self.timePoints]

                #splits descriptions if they contain more than one peptide
                descriptions = self.cellData[i]['Master Protein Descriptions'].str.split(';',1).values
                master = [0]*len(descriptions)
                secondary = [0]*len(descriptions)
                for j in range(len(descriptions)):
                    master[j] = descriptions[j][0]
                    try:
                        secondary[j] = descriptions[j][1]
                    except:
                        secondary[j] = ''
                self.cellData[i]['Master Protein Descriptions'] = master
                self.cellData[i].insert(2, 'Overflow Protein Descriptions', secondary, True)

                if self.caseInsensitive:
                    self.cellData[i]['peptide-phosphosite'] = self.cellData[i][
                        "peptide-phosphosite"].str.upper()

    def __str__(self):
        """Human readable string describing replicate.

        """
        return "{} MS REPLICATE\nCell Lines: {}\nSize: {}\nIntersection Size: {}\n".format(self.replicateName, str(self.cellLines).strip("[]"), str([self.cellData[i].shape[0] for i in range(len(self.cellData))]).strip("[]"), self.cellFullIntersection.shape[0])

    def __iter__(self):
        """Provides support for iterating over replicate.

        Examples:
            [each for each in exp[0]] --> each cell line in replicate

        """
        self.n = 0
        return self

    def __next__(self):
        """Provides support for iterating over replicate.

        """
        if self.n < len(self.cellData):
            self.n += 1
            return self.cellData[self.n - 1]
        else:
            raise StopIteration

    def __getitem__(self, index):
        """Provides support for indexing experiment for replicates.

        Examples:
            experiment[0]

        """
        return self.cellData[index]

    def save(self, passedWriter, replicates=False):
        """Saves contents of replicate to one excel file.

        Args:
            passedWriter (pd.ExcelWriter): specifies file destination

        Notes:
            Each cell line, as well as cellReferenceIntersections and cellFullIntersection, get saved to separate tabs.

        """

        #saves replicates
        with passedWriter as writer:
            for i in range(len(self.cellData)):
                try:
                    self[i].to_excel(writer, sheet_name=str(self.cellLines[i]))
                except IndexError:
                    pass
            for i in range(len(self.cellReferenceIntersections)):
                try:
                    self.cellReferenceIntersections[i].to_excel(writer, sheet_name=str(self.cellLines[i]) + " to reference")
                except AttributeError:
                    print("ERROR: Replicate reference intersections not calculated")
                    break
                except IndexError:
                    pass
            try:
                self.cellFullIntersection.to_excel(writer, sheet_name="Full Intersection")
            except AttributeError:
                print("ERROR: Replicate full intersection not calculated")
                pass
            except IndexError:
                pass

    def setReference(self, updatedIndices):
        """Updates REFERENCE cell line to a new one.

        Args:
            updatedIndices (list): list of new indices to rearrange.

        """
        # self.colors[:] = [self.colors[i] for i in updatedIndices]
        self.cellData[:] = [self.cellData[i] for i in updatedIndices]

    def fullIntersection(self):
        """Executes an inner pd.merge() on all of experiment's combined replicate data.

        """
        self.cellFullIntersection = modules.fullIntersection(self.cellData, self.caseInsensitive)

    def referenceIntersections(self, combine=False):
        """Executes a separate inner pd.merge() to reference for each cell line of combined replicate data.

        Args:
            combine (bool): Whether to include reference data at the end of returned list of DataFrames comparing others to reference.

        """
        self.cellReferenceIntersections = modules.separateIntersections(self.cellData[:-1], self.cellData[-1], self.caseInsensitive, combine)

    def addTechnicalReplicate(self, technical_replicate):
        """Adds a technical replicate to the ith experimental replicate.

        Notes:
            Technical replicate is added via outer pd.merge, so all peptides in at least the experimenal or technical replicate are included.
            Overlapping peptide abundances are averaged. Ranges are not tracked because they are not considered separate replicates.

        Args:
            technical_replicate (ExperimentalReplicate): ExperimentalReplicate object to be used as technical replicate

        """
        technicalReplicate = ExperimentalReplicate(technical_replicate)

        for i in range(len(self.cellData)):
            self.cellData[i] = pd.merge(
                self.cellData[i],
                technicalReplicate.cellData[i],
                how="outer",
                on=["peptide-phosphosite", "Master Protein Descriptions", "Overflow Protein Descriptions"])

            for each in self.timePoints:
                cellLine = self.cellLines[i]
                replicate_x = str(cellLine) + "-" + str(each) + "_x"
                replicate_y = str(cellLine) + "-" + str(each) + "_y"

                self.cellData[i][str(cellLine) + "-" + str(each)] = self.cellData[i][[replicate_x, replicate_y]].mean(axis=1)

                self.cellData[i] = self.cellData[i].drop(columns=[replicate_x, replicate_y])

        self.fullIntersection()
        self.referenceIntersections()

    def log2Comparison(self, keyword=None, threshold=0, normalizeToBasal=False, tail="both", replicateNumber=0, fig=None, givenTitle=None):
        """Compares and graphs percentage of peptides in individual others that show a specified log2 fold difference compared to reference.

        Args:
            keyword (str): optional keyword to include to limit peptides considered in comparison
            threshold (str): log2 cutoff threshold
            normalizeToBasal (bool): whether to subtract all others time points by each others basal (0), and subtract all reference time points by reference basal (0)
            tail (str): tail of threshold to consider
                "both": peptides with log2 >= threshold and log2 <= -1*threshold
                "lower": peptides with log2 <= -1*threshold
                "upper": peptides with log2 >= threshold
            replicateNumber (int): used to select line style (specified if multiple replicates are plotted on the same plot)
            fig (plt.figure): figure to graph on (specified if multiple replicates are plotted on the same plot)

        Notes:
            Each others is separately overlapped with reference when performing comparison to preserve maximum data, so not all peptides are expressed across all points.

        Returns:
            comparison (list of list of df): peptides meeting threshold as a list of replicates, each of which is a list of cell line DataFrames
            originalSizes (list of list of int): overlapping peptides for each others to reference comparison, useful to see how many peptides each percentage represents

        """

        if fig == None:
            f = plt.figure(random.randint(1, 1000))

        others = []
        originalSizes = []

        reference = self.cellData[-1]
        reference = reference.set_index(["peptide-phosphosite", "Master Protein Descriptions", "Overflow Protein Descriptions"])
        if normalizeToBasal:
            reference = reference.div([reference.iloc[:, 0]] * len(self.timePoints), axis='columns')

        for i in range(len(self.cellData) - 1):
            if keyword != None:
                if type(keyword) == str:
                    lowerData = self.cellData[i]["Master Protein Descriptions"].str.lower()
                    tempCellData = self.cellData[i][lowerData.str.contains(keyword.lower())]

                elif type(keyword) == list:
                    conditions = []

                    lowerData = self.cellData[i]["Master Protein Descriptions"].str.lower()
                    tempCellData = self.cellData[i][lowerData.str.contains(keyword[0].lower())]

                    for each in keyword[1:]:
                        lowerData = self.cellData[i]["Master Protein Descriptions"].str.lower()
                        loopCellData = self.cellData[i][lowerData.str.contains(each.lower())]

                        tempCellData = pd.concat([tempCellData, loopCellData],join="outer")
                        tempCellData = tempCellData.drop_duplicates(subset=["peptide-phosphosite","Master Protein Descriptions", "Overflow Protein Descriptions"])

                else:
                    print("ERROR: Keyword must be either str, or list of str")
                    raise Exception

                tempCellData = tempCellData.set_index(["peptide-phosphosite", "Master Protein Descriptions", "Overflow Protein Descriptions"])
            else:
                #set descriptions to index
                tempCellData = self.cellData[i].set_index(["peptide-phosphosite", 'Master Protein Descriptions', 'Overflow Protein Descriptions'])

            #normalize others to basal level
            if normalizeToBasal:
                tempCellData = tempCellData.div([tempCellData.iloc[:, 0]] *len(self.timePoints),axis='columns')
            #renames reference column headers so pandas div can compare correct rows (i.e. 30 sec time point to 30 sec time point)
            reference.columns = tempCellData.columns
            #divides others/reference, then performs log2
            tempCellData = np.log2(tempCellData.div(reference))
            #drops rows where peptide does not overlap in both others and reference
            tempCellData = tempCellData.dropna()

            originalSizes.append(tempCellData.shape[0])

            timePoints = []

            #identifies peptides above threshold for each time point separately
            for k in range(tempCellData.shape[1]):
                if tail == "both":
                    criteria = (tempCellData.iloc[:, k] >= threshold) | (tempCellData.iloc[:, k] <= -1 * threshold)
                    title = "Peptides with abs(log2(Mut/Reference)) >= " + str(threshold)
                elif tail == "upper":
                    criteria = (tempCellData.iloc[:, k] >= threshold)
                    title = "Peptides with log2(Mut/Reference) >= " + str(threshold)
                elif tail == "lower":
                    criteria = (tempCellData.iloc[:, k] <= -1 * threshold)
                    title = "Peptides with log2(Mut/Reference) <= " + str(-1 * threshold)

                timePoints.append(tempCellData[criteria])

            others.append(timePoints)

        #plotting
        sns.set()

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple','tab:brown', 'tab:pink', 'tab:gray']
        linestyles = ['-', '--', '-.', ':']

        for k in range(len(others)):
            #number of peptides above threshold for cell line in replicate
            thresholdPeptides = [each.shape[0] for each in others[k]]
            #scales number to total peptides in cell line comparison to reference for replicate
            scaledPeptides = [each / originalSizes[k] * 100 for each in thresholdPeptides]

            plt.plot(self.secondTimePoints,scaledPeptides, label=self.replicateName + " " + str(self.cellLines[k]), color=colors[k], ls=linestyles[replicateNumber], marker=".")

        if normalizeToBasal == True:
            title += ", normalized to basal"

        if keyword:
            title = title + ", from " + str(keyword)

        if givenTitle == None:
            title = str(self.replicateName) + " " + title
            plt.title(title)
        else:
            plt.title(givenTitle)

        plt.xlabel("Time Points (seconds)")
        plt.ylabel("Percent of Peptides")
        plt.legend()

        return others, originalSizes

    def peptidePicker(self, threshold=0, normalizeToBasal=False, tail="both", trajectory=None):
        """Selects peptides that meet a given trajectory compared to a threshold across individual cell lines.

        Args:
            threshold (str): log2 cutoff threshold
            normalizeToBasal (bool): whether to subtract all others time points by each others basal (0), or subtract all reference time points by reference basal (0)
            tail (str): tail of threshold to consider
                "both": peptides with log2 >= threshold and log2 <= -1*threshold
                "lower": peptides with log2 <= -1*threshold
                "upper": peptides with log2 >= threshold
            trajectory (list of bool/str): trajectory of peptides compared to threshold, default value is [True, True, True, True, True]
                True: peptide must meet threshold at this time point to be included
                False: peptide cannot meet threshold at this time point to be included
                "All": time point does not enforce a threshold (all peptides pass this time point)

        Notes:
            Each others is separately overlapped with reference when performing comparison to preserve maximum data, so not all peptides are expressed across all points.

        Examples:
            exp[0].peptidePicker(threshold = 1.5, tail = 'upper', trajectory = ["All", "All", False, False, True])
                Finds peptides such that peptide_1_minute < 1.5, peptide_2_minute < 1.5, and peptide_5_minute.
                0 and 30 second time points do not enforce threshold.

        """
        others = []
        originalSizes = []

        f = plt.figure(random.randint(1, 1000))

        if trajectory == None:
            trajectory = [True] * len(self.timePoints)

        reference = self.cellData[-1]
        reference = reference.set_index(["peptide-phosphosite", "Master Protein Descriptions", "Overflow Protein Descriptions"])
        if normalizeToBasal:
            reference = reference.div([reference.iloc[:, 0]] * len(self.timePoints), axis='columns')

        for i in range(len(self.cellData) - 1):
            #set descriptions to index
            tempCellData = self.cellData[i].set_index(["peptide-phosphosite", 'Master Protein Descriptions', 'Overflow Protein Descriptions'])

            #normalize others to basal level
            if normalizeToBasal:
                tempCellData = tempCellData.div([tempCellData.iloc[:, 0]] * len(self.timePoints),axis='columns')
            #renames reference column headers so pandas div can compare correct rows (i.e. 30 sec time point to 30 sec time point)
            reference.columns = tempCellData.columns
            #divides others/reference, then performs log2
            tempCellData = np.log2(tempCellData.div(reference))
            #drops rows where peptide does not overlap in both others and reference
            tempCellData = tempCellData.dropna()

            originalSizes.append(tempCellData.shape[0])

            #identifies peptides above threshold for each time point separately
            for k in range(tempCellData.shape[1]):
                if tail == "both":
                    criteria = (tempCellData.iloc[:, k] >= threshold) | (tempCellData.iloc[:, k] <= -1 * threshold)

                    title = self.replicateName + " peptides with abs(log2(Mut/Reference)) >= " + str(threshold)
                elif tail == "upper":
                    criteria = (tempCellData.iloc[:, k] >= threshold)
                    title = self.replicateName + " peptides with log2(Mut/Reference) >= " + str(threshold)
                elif tail == "lower":
                    criteria = (tempCellData.iloc[:, k] <= -1 * threshold)
                    title = self.replicateName + " peptides with log2(Mut/Reference) <= " + str(-1 * threshold)

                if not trajectory[k]:
                    tempCellData = tempCellData[~criteria]
                elif trajectory[k] == "All":
                    pass
                elif trajectory[k]:
                    tempCellData = tempCellData[criteria]
                else:
                    print("Must use True, False, or 'Both' for trajectory listings")
                    raise ValueError

            others.append(tempCellData)

        #plotting
        sns.set()

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple','tab:brown', 'tab:pink', 'tab:gray']

        for k in range(len(others)):
            for pep in range(others[k].shape[0]):
                pepName = others[k].iloc[pep, :].name[1].split("=")[0][0:-3]
                plt.plot(self.secondTimePoints,others[k].iloc[pep, :],label=str(self.cellLines[k]) + " " + pepName,color=colors[k],marker=".")

        if normalizeToBasal == True:
            title += ", normalized to basal"

        plt.title(title)
        plt.xlabel("Time Points (seconds)")
        plt.ylabel("Peptide Trajectories")
        plt.legend()

        return others, originalSizes

    def heatmap(self, name="", display=True, saveFile = False, saveFig = False, fileLocation="", fullscreen=False, normalization='refbasal'):
        """Plots a heatmap comparing all cell lines at once.

        Args:
            name (str): plot title
            display (bool): whether to display a graph
            saveFile (bool): whether to save data to Excel files
            saveFig (bool): whether to save figs
            fileLocation (str): file location to save to
            fullscreen (bool): whether to display graph fullscreen
            normalization (str): normalization scheme to use, default refbasal but can use refbasal, reftime, ownbasal

        """
        if fileLocation == '':
            fileLocation = self.fileLocation
        modules.heatmap(self.cellFullIntersection.copy(), self.cellLines, self.timePoints, name, display, saveFile, saveFig, fileLocation, fullscreen, normalization)

    def heatmapToReference(self, name="", display=True, saveFile = False, saveFig = False, fileLocation="", fullscreen = False, normalization='refbasal'):
        """Plots multiple heatmaps comparing cell lines to reference.

        Args:
            name (str): plot title
            display (bool): whether to display a graph
            saveFile (bool): whether to save data to Excel files
            saveFig (bool): whether to save figs
            fileLocation (str): file location to save to
            fullscreen (bool): whether to display graph fullscreen
            normalization (str): normalization scheme to use, default refbasal but can use refbasal, reftime, ownbasal

        """
        if fileLocation == '':
            fileLocation = self.fileLocation
        modules.heatmapToReference(self.cellReferenceIntersections.copy(), self.cellLines, self.timePoints, name, display, saveFile, saveFig, fileLocation, fullscreen, normalization)

    def pcaToReference(self, name="", display=True, saveFile = False, saveFig = False, fileLocation="", fullscreen=False):
        """Plots separate PCAs comparing each others to reference intersection individually.

        Args:
            name (str): plot title
            display (bool): whether to display graph
            saveFile (bool): whether to save PCA loadings to Excel files
            saveFig (bool): whether to save figs
            fileLocation (str): file location to save to
            fullscreen (bool): whether to display graph fullscreen

        Notes:
            Useful tutorial https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60


        """
        if fileLocation == '':
            fileLocation = self.fileLocation
        modules.pcaToReference(self.cellReferenceIntersections.copy(), self.cellLines, self.timePoints, self.secondTimePoints, name, display, fileLocation, fullscreen, self.colors)

    def pca(self, name="", display=True, saveFile = False, saveFig = False, fileLocation="", fullscreen=False):
        """Plots one PCA comparing all others and reference at once.

        Args:
            name (str): plot title
            display (bool): whether to display graph
            saveFile (bool): whether to save PCA loadings to Excel files
            saveFig (bool): whether to save figs
            fileLocation (str): file location to save to
            fullscreen (bool): whether to display graph fullscreen

        """
        if fileLocation == '':
            fileLocation = self.fileLocation
        modules.pca(self.cellFullIntersection.copy(), self.cellLines, self.timePoints, self.secondTimePoints, name, display, saveFile, saveFig, fileLocation, fullscreen, self.colors)

class PhenotypicMeasurement:
    """Stores all replicates of one type of phenotypic measurement associated with the Experiment.

    """

    phenotypicType = ""
    phenotypicData = None
    cellLines = []
    fileLocation = ''

    def __init__(self, locations, phenotypicType='Generic', cellLines = None, fileLocation = ''):
        self.phenotypicType = phenotypicType
        self.cellLines = cellLines
        df = pd.DataFrame()
        self.fileLocation = fileLocation

        #list of replicate locations to read in
        if type(locations) == list:
            #reads in cell line names and first replicate
            df = pd.read_csv(locations[0])
            df.set_index(df.columns[0], inplace = True)
            #makes sure order of cell lines matches those in experiment
            df.reindex(cellLines)
            #reads other replicates
            for i in range(1,len(locations)):
                temp = pd.read_csv(locations[i])
                temp.set_index(temp.columns[0], inplace = True)
                temp.reindex(cellLines)
                #adds replicate data
                df[temp.columns[0]] = temp.iloc[:,0]

        #pandas df
        elif type(locations) == pd.DataFrame:
            df = locations

        #one replicate location to read in
        elif type(locations) == str:
            df = pd.read_csv(locations, index_col = 0)
            df.set_index(df.columns[0], inplace = True)
            df = df.reindex(cellLines)
        else:
            print("Input type is not supported.")
            exit(0)

        #sets data
        self.phenotypicData = df

    def __str__(self):
        """Human readable string describing replicate.

        """
        return "{} PHENOTYPIC MEASUREMENT\nCell Lines: {}\n{} Replicate(s)".format(self.phenotypicType,str(self.cellLines).strip("[]"),self.phenotypicData.shape[1])

    def __iter__(self):
        """Provides support for iterating over replicates in PhenotypicMeasurement.

        """
        self.n = 0
        return self

    def __next__(self):
        """Provides support for iterating over replicates in PhenotypicMeasurement.

        """
        pass

    def __getitem__(self, index):
        """Provides support for indexing PhenotypicMeasurement for replicates.

        """
        pass

    def save(self, passedWriter, replicates=False):
        """Saves contents of PhenotypicMeasurement to one excel file.

        Args:
            passedWriter (pd.ExcelWriter): specifies file destination

        Notes:
            Each cell line, as well as cellReferenceIntersections and cellFullIntersection, get saved to separate tabs.

        """
        pass

    def setReference(self, line, updatedIndices = None):
        """Updates REFERENCE cell line to a new one.

        Args:
        line (str): line to be set to reference
            updatedIndices (list): list of new indices to rearrange.

        """
        #called from a higher class
        if updatedIndices:
            updated = [self.phenotypicData.index.values[i] for i in updatedIndices]
            self.phenotypicData = self.phenotypicData.reindex(updated)
            # self.colors[:] = [self.colors[i] for i in updatedIndices]
        #called by user directly
        else:
            original = self.cellLines.copy()
            updated = list(self.phenotypicData.index.values)
            ind = updated.index(line)
            updated.append(updated.pop(ind))
            self.phenotypicData = self.phenotypicData.reindex(updated)
            # self.colors.append(self.colors.pop(ind))

    def groupPlot(self, relativeToReference = False, name="", display=True, fileLocation = ""):
        '''Plots one phenotypic measurement, including std. dev. error bars and letter-based groups based on Tukey's HSD.
        
        Args:
            relativeToReference (bool): whether to normalize to reference or plot as a separate cell line.
            name (str): name to include in plots
            display (bool): whether to display plots
            fileLocation (str): location to save to
        '''
        #computes tukey HSD
        thsd = modules.tukey(self.phenotypicData, self.cellLines)
        #computes letter-based groups based on Tukrey HSD pairwise p-values
        lb = modules.letterBased(thsd, self.cellLines)
        
        #plotting
        y = self.phenotypicData.copy()
        #computes mean and std dev across replicates for each cell line
        y['mean'] = 0
        y['std'] = 0

        for i in range(y.shape[0]):
            y.iloc[i,-2] = y.iloc[i,0:-2].mean()
            y.iloc[i,-1] = y.iloc[i,0:-2].std()

        if relativeToReference:
            #new dataframe for data relative to REFERENCE
            y_a = pd.DataFrame()
            #divide mean
            y_a['mean'] = y['mean']/y.iloc[-1,-2]
            y_a['std'] = 0
            #std dev error propagation from ipl.physics.harvard.edu/wp-uploads/2013/03/PS3_Error_Propagation_sp13.pdf
            # f = A/B, std dev_f = |f|*sqrt((std dev_A/A)^2+(std dev_B/B)^2)
            for i in range(y.shape[0]):
                y_a.iloc[i,-1] = abs(y_a.iloc[i,-2])*math.sqrt((y.iloc[i,-1]/y.iloc[i,-2])**2 + (y.iloc[-1,-1]/y.iloc[-1,-2])**2)
            #linearly spaces cell line data for plotting, the actual labels are added below
            y_a['Cell Line'] = range(len(self.cellLines))
            #drops REFERENCE
            y_a = y_a.iloc[:-1]
            ylab = '{} {} Relative to {}'.format(name,self.phenotypicType,self.cellLines[-1])
        else:
            y_a = y[['mean', 'std']].copy()
            #linearly spaces cell line data for plotting, the actual labels are added below
            y_a['Cell Line'] = range(len(self.cellLines))
            ylab = '{} Absolute {}'.format(name,self.phenotypicType)

        #plot markers at means
        g = sns.swarmplot(data=y_a.iloc[:,[0,2]], x = 'Cell Line', y = 'mean', edgecolor = 'k', linewidth = 1, size = 10, color = ".50")
        ax = plt.gca()
        #plot errorbars as std deviations
        container = plt.errorbar(y_a['Cell Line'], y_a['mean'], y_a['std'], fmt = 'none', ecolor = "0.50")
        plt.ylabel(ylab)
        #set x ticks to cell lines
        plt.setp(ax, xticklabels=self.cellLines)

        #grabs vertical errorbars from container
        errorbars = container.lines[2][0].get_segments()
        #offset is independent of y scale
        ymin, ymax = ax.get_ylim()
        offset = (ymax-ymin)/50

        #for adjusting plot ylim for group annotations
        annotation_max = errorbars[0][1][1]

        for i, location in enumerate(errorbars):
            #annotates with group labels
            annotation_max = max(location[1][1]+offset,annotation_max)
            ax.annotate(lb.iloc[i,-1], xy=(location[1][0], location[1][1]+.02), horizontalalignment='center', annotation_clip = False)
        #sets new ylim to not cut off labels
        ax.set_ylim((ymin, annotation_max+offset*2))
        plt.title("Grouped {} Data".format(self.phenotypicType))

        if not display:
            plt.savefig(fileLocation+ylab+'.png')
            plt.close()

    def replicatePlot(self, relativeToReference = False, name="", display=True, fileLocation = ""):
        '''Plots one phenotypic measurement, indicating the individual phenotypic replicates' data.

        Args:
            relativeToReference (bool): whether to normalize to reference or plot as a separate cell line.
            name (str): name to include in plots
            display (bool): whether to display plots
            fileLocation (str): location to save to

        Notes:
            Useful for debugging whether one phenotypic measurement should be excluded.

        '''
        #resets index
        ref = self.phenotypicData.index.values[-1]
        if relativeToReference:
            y = self.phenotypicData.copy()
            y = y.iloc[0:-1].divide(y.loc[ref])
            y = y.reset_index()
        else:
            y = self.phenotypicData.reset_index()

        y.rename(columns={'index':'Cell Line'}, inplace = True)

        if name == "":
            title = self.phenotypicType
        else:
            title = name+" "+self.phenotypicType

        #melt data for seaborn plotting
        mm = y.melt(id_vars="Cell Line", var_name = "Replicate", value_name = title)
        #shows individual replicates
        sns.boxplot(data=mm, x="Cell Line", y=title, color=".50")
        g = sns.swarmplot(data=mm, x="Cell Line", y=title, hue="Replicate", palette="Set2", edgecolor="k", linewidth=1, size=10)
        # y_lim = g.axes.get_ylim()
        plt.title("{} Phenotypic Data".format(title))
        if relativeToReference:
            plt.ylabel("{} Relative to {}".format(title,ref))
        else:
            plt.ylabel("Absolute {}".format(title))

        if not display:
            plt.savefig(fileLocation+title+' Phenotypic Replicate Data.png')
            plt.close()
