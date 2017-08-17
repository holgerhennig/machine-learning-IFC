# machine-learning-IFC
A machine learning workflow for imaging flow cytometry (IFC).
This is a step by step guide to analyze images from imaging flow cytometry with classical machine learning.
It is based on the following articles (please cite these when using the workflow):
- [Hennig et al. An open-source solution for advanced imaging flow cytometry data analysis using machine learning. Methods 112, 201 (2017)](http://www.sciencedirect.com/science/article/pii/S1046202316302912)
- [Blasi et al. Label-free cell cycle analysis for high-throughput imaging flow cytometry. Nature Communications 7, 10256 (2016)](https://www.nature.com/articles/ncomms10256)

For our deep learning workflow, please see https://github.com/broadinstitute/deepometry

## Step 1: extracting subpopulations in IDEAS

## Step 2: Generate montages
A montage is a collection of (e.g.) 900 cells, arranged in a 30x30 grid.
The montages are automatically generated from a cif file by a [stitching script](https://github.com/CellProfiler/stitching)
Input: example.cif file
Output: For each of the max. 12 channels, montages are created.
ch1.tif
ch2.tif
...
ch12.tif

## Step 3: Extract features with CellProfiler
This CellProfiler pipeline is ...


Q&A
Q: I have an IFC experiment with several markers. You have developed a label-free analysis method. Can I use your workflow including the marker channels?
A: Yes, there is already one marker image defined in the CellProfiler pipeline, simply add further marker images in the measurement modules.

Q: The object segmentation ("finding the cell objects") does not work well with my pipeline.
A: If you have a nuclear stain, you can adapt the pipeline accordingly (identify primary objects on the nuclei, and identify secondary objects on the cells with nuclei as guiding objects). For help on how to improve or tweak your pipeline, see the [CellProfiler forum](http://forum.cellprofiler.org/).
