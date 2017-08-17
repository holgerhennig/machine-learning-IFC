# machine-learning-IFC
We provide a machine learning workflow for imaging flow cytometry (IFC). This is a step by step guide for the high-content analysis of IFC images with machine learning. It is based on the following articles (please cite these when using the workflow):
- [Hennig et al. An open-source solution for advanced imaging flow cytometry data analysis using machine learning. Methods 112, 201 (2017)](http://www.sciencedirect.com/science/article/pii/S1046202316302912)
- [Blasi et al. Label-free cell cycle analysis for high-throughput imaging flow cytometry. Nature Communications 7, 10256 (2016)](https://www.nature.com/articles/ncomms10256)

For our deep learning workflow, please see https://github.com/broadinstitute/deepometry

## Step 1: extracting cell subpopulations in IDEAS
Gating in IDEAS enables to define different subpuplations of cells (for example cancer cells, normal cells, treated cells). In the menu in IDEAS, choose extract subpopulation and generate a .cif file for each subpopulation. The .cif files will be processed in the next step.

## Step 2: Generate image montages
A montage is a collection of (e.g.) 900 cells, arranged in a 30x30 grid. The montages are automatically generated from a .cif file by a [stitching script](https://github.com/CellProfiler/stitching).  
Input: example.cif file  
Output: For each of the max. 12 channels, montages are created.  
ch1.tif  
ch2.tif  
...   
ch12.tif

## Step 3: Extract features with CellProfiler
How to identify the cells in the image montages and extract hundreds of features with CellProfiler?  
This repository contains a CellProfiler pipeline IFC_template.cppipe. Start CellProfiler, import the pipeline and drag & drop your image montages into CellProfiler. You probably need to adapt the pipeline to your image assay.   
Download Cellprofiler: http://cellprofiler.org/releases/  
For help on how to improve or tweak your pipeline, see the [CellProfiler forum](http://forum.cellprofiler.org/).  

## Step 4: Machine learning in python
A machine learning script in python is provided in this repository, which may server as a starting point for your high-content image data analysis and exploration. The script works with the example data below without modifications.

## Examples & Tutorial
The examples folder contains a quick example with a small data set to test our workflow before analyzing your own data. The example data is also suitable for workshops & tutorials. The CellProfiler pipeline finishes in ~4 Minutes, and machine learning predictions in python finish almost instantly.  
The full data set from [Hennig et al. Methods 112, 201 (2017)](http://www.sciencedirect.com/science/article/pii/S1046202316302912) with a detailed description is available on http://cellprofiler.org/imagingflowcytometry/index.html

## Q&A
Q: I have an IFC experiment with several markers. However, the Blasi paper focuses on a label-free analysis method. Can I use the machine learning workflow including the marker channels?  
A: Yes, there is already one marker image defined in the CellProfiler pipeline, simply add further marker images in the measurement modules in CellProfiler.

Q: The object segmentation ("finding the cell objects") does not work well with my pipeline.  
A: If you have a nuclear stain, you can adapt the pipeline accordingly (identify primary objects on the nuclei, and identify secondary objects on the cells with nuclei as guiding objects). For help on how to improve or tweak your pipeline, see the [CellProfiler forum](http://forum.cellprofiler.org/).

Q: My image montages are empty (all black)  
A: Try the following
- Many image viewers don't scale the image range. Open the montages in CellProfiler or [ImageJ](https://fiji.sc/)  
- Look up in IDEAS in which channels images have been recorded, the other channels are empty and the respective montages can be discarded.  
