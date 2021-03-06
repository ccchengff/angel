package com.tencent.angel.ml.FPGBDT.algo.storage

import java.util

import com.tencent.angel.ml.math.vector.{SparseDoubleSortedVector, SparseDoubleVector}
import com.tencent.angel.ml.param.FPGBDTParam
import com.tencent.angel.ml.utils.Maths
import com.yahoo.sketches.quantiles.DoublesSketch
import org.apache.commons.logging.{Log, LogFactory}

/**
  * Created by ccchengff on 2017/11/16.
  */
class TrainDataStore(param: FPGBDTParam, numInstance: Int) extends FPRegTDataStore(param, numInstance) {
  val LOG: Log = LogFactory.getLog(classOf[TrainDataStore])

  private val numFeature: Int = param.featHi - param.featLo

  private val featRows = new Array[SparseDoubleSortedVector](numFeature)
  private val featIndices = new Array[Array[Int]](numFeature)
  private val featBins = new Array[Array[Int]](numFeature)
  //private val splits = new Array[Array[Float]](numFeature)
  //private val zeroBins = new Array[Int](numFeature)
  private val splits = new Array[Array[Float]](param.numFeature)
  private val zeroBins = new Array[Int](param.numFeature)

  //def getFeatRow(fid: Int) = featRows(fid - param.featLo)

  def setFeatureRow(fid: Int, featRow: SparseDoubleSortedVector): Unit = {
    this.featRows(fid - param.featLo) = featRow
  }

  def setFeatureRow(fid: Int, featRow: SparseDoubleVector): Unit = {
    val indices = featRow.getIndices
    val totalNnz = indices.length
    util.Arrays.sort(indices)
    val values = new Array[Double](totalNnz)
    for (i <- 0 until totalNnz) {
      values(i) = featRow.get(indices(i))
    }
    this.featRows(fid - param.featLo) =
      new SparseDoubleSortedVector(numInstance, indices, values)
  }

  def createSketch(numSplit: Int, numThread: Int): Unit = {
    val fracs = new Array[Double](numSplit)
    for (i <- 0 until numSplit) {
      fracs(i) = i.toDouble / numSplit.toDouble
    }

    var minNnz = numInstance
    var maxNnz = 0
    for (fid <- param.featLo until param.featHi) {
      val i = fid - param.featLo
      // 1. create quantile sketch
      val sketch = DoublesSketch.builder().build()
      val values = featRows(i).getValues
      val nnz = values.length
      //minNnz = Math.min(minNnz, Math.max(0, nnz))
      if (nnz < minNnz && nnz > 0) minNnz = nnz
      if (nnz > maxNnz) maxNnz = nnz
      for (ins <- 0 until nnz) {
        sketch.update(values(ins))
      }
      // 2. get quantiles as split values, find zero bin
      splits(i) = Maths.double2Float(sketch.getQuantiles(fracs))
      LOG.info(s"Candidates of feature[$fid]: [" + splits(i).mkString(", ") + "]")
      zeroBins(i) = findZeroBin(splits(i))
      LOG.info(s"Zero bin of feature[$fid]: ${zeroBins(i)}, nonzero: $nnz")
      // 3. find bin of each instance
      featBins(i) = new Array[Int](nnz)
      for (ins <- 0 until nnz) {
        featBins(i)(ins) = indexOf(values(ins).toFloat, splits(i), zeroBins(i))
      }
      // 4. for memory efficient
      featIndices(i) = featRows(i).getIndices.clone()
      featRows(i).clear()
    }
    LOG.info(s"Feature range: [${param.featLo}-${param.featHi}), min nnz=$minNnz, max nnz=$maxNnz")
  }

  def findZeroBin(arr: Array[Float]): Int = {
    val size: Int = arr.length
    var zeroIdx: Int = 0
    if (arr(0) > 0.0f) {
      zeroIdx = 0
    }
    else if (arr(size - 1) < 0.0f) {
      zeroIdx = size - 1
    }
    else {
      var t: Int = 0
      while (t < size - 1 && arr(t + 1) < 0.0f)
        t += 1
      zeroIdx = t
    }
    zeroIdx
  }

  def indexOf(x: Float, arr: Array[Float], zeroIdx: Int): Int = {
    val size = arr.length
    var left = zeroIdx
    var right = zeroIdx
    if (x < 0.0f) {
      left = 0
    }
    else {
      right = size - 1
    }
    while (left <= right) {
      val mid = left + ((right - left) >> 1)
      if (arr(mid) <= x) {
        if (mid + 1 == size || arr(mid + 1) > x)
          return mid
        else
          left = mid + 1
      }
      else
        right = mid - 1
    }
    zeroIdx
  }

  def indexOf(x: Float, fid: Int): Int = {
    indexOf(x, splits(fid), zeroBins(fid))
  }

  def getFeatIndices(fid: Int) = featIndices(fid - param.featLo)

  /*def setFeatIndices(fid: Int, indices: Array[Int]): Unit = {
    featIndices(fid - param.featLo) = indices
  }*/

  def getFeatBins(fid: Int) = featBins(fid - param.featLo)

  /*def setFeatBins(fid: Int, bins: Array[Int]): Unit = {
    featBins(fid - param.featLo) = bins
  }*/

  def getFeatureRow(fid: Int): (Array[Int], Array[Int]) = {
    (featIndices(fid - param.featLo), featBins(fid - param.featLo))
  }

  def setFeatureRow(fid: Int, indices: Array[Int], bins: Array[Int]): Unit = {
    featIndices(fid - param.featLo) = indices
    featBins(fid - param.featLo) = bins
  }

  /*def getSplit(fid: Int, splitIdx: Int) = splits(fid - param.featLo)(splitIdx)

  def getSplits(fid: Int) = splits(fid - param.featLo)

  def setSplits(fid: Int, splits: Array[Float]): Unit = {
    this.splits(fid - param.featLo) = splits
  }

  def getZeroBin(fid: Int) = zeroBins(fid - param.featLo)*/

  def getSplit(fid: Int, splitIdx: Int) = splits(fid)(splitIdx)

  def getSplits(fid: Int) = splits(fid)

  def setSplits(fid: Int, splits: Array[Float]): Unit = {
    this.splits(fid) = splits
    this.zeroBins(fid) = findZeroBin(splits)
  }

  def getZeroBin(fid: Int) = zeroBins(fid)

  def getZeroBins = zeroBins

}
