# -*- coding: utf-8 -*-

import difflib
import numpy as np
import os
import SimpleITK as sitk
import scipy.spatial

# Set the path to the source data (e.g. the training data for self-testing)
# and the output directory of that subject
testDir = ''  # For example: '/data/Utrecht/0'
participantDir = ''  # For example: '/output/teamname/0'


def do(testImage, resultImage):
    """Main function"""

    dsc = getDSC(testImage, resultImage)
    # h95 = getHausdorff(testImage, resultImage)
    avd = getAVD(testImage, resultImage)
    recall, f1 = getLesionDetection(testImage, resultImage)

    # print(
    # 'Dice', dsc, '(higher is better, max=1)')
    # print(
    # 'HD', h95, 'mm', '(lower is better, min=0)')
    # print(
    # 'AVD', avd, '%', '(lower is better, min=0)')
    # print(
    # 'Lesion detection', recall, '(higher is better, max=1)')
    # print(
    # 'Lesion F1', f1, '(higher is better, max=1)')

    return dsc, avd, recall, f1


def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and non-WMH masked."""
    testImage = sitk.ReadImage(testFilename)
    resultImage = sitk.ReadImage(resultFilename)

    # Check for equality
    assert testImage.GetSize() == resultImage.GetSize()

    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)

    # Remove non-WMH from the test and result images, since we don't evaluate on that
    maskedTestImage = sitk.BinaryThreshold(testImage, 0.5, 1.5, 1, 0)  # WMH == 1
    nonWMHImage = sitk.BinaryThreshold(testImage, 1.5, 2.5, 0, 1)  # non-WMH == 2
    maskedResultImage = sitk.Mask(resultImage, nonWMHImage)

    # Convert to binary mask
    if 'integer' in maskedResultImage.GetPixelIDTypeAsString():
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 1, 1000, 1, 0)
    else:
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)

    return maskedTestImage, bResultImage


def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()

    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)


def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""

    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage = sitk.BinaryErode(testImage, (1, 1, 0))
    eResultImage = sitk.BinaryErode(resultImage, (1, 1, 0))

    hTestImage = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)

    hTestArray = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)

    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates = np.apply_along_axis(testImage.TransformIndexToPhysicalPoint, 1, np.transpose(np.flipud(np.nonzero(hTestArray))).astype(int))
    resultCoordinates = np.apply_along_axis(testImage.TransformIndexToPhysicalPoint, 1, np.transpose(np.flipud(np.nonzero(hResultArray))).astype(int))

    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)

    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))


def getLesionDetection(testImage, resultImage):
    """Lesion detection metrics, both recall and F1."""

    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))

    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)

    # recall = (number of detected WMH) / (number of true WMH)
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH

    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))

    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)

    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return recall, f1


def getAVD(testImage, resultImage):
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()

    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)

    div = float(testStatistics.GetSum())
    if div == 0:
        return 0
    else:
        return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100
