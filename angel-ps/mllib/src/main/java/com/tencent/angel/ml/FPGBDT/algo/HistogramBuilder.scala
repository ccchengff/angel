package com.tencent.angel.ml.FPGBDT.algo

import java.util

import com.tencent.angel.ml.FPGBDT.algo.storage.TrainDataStore
import com.tencent.angel.ml.GBDT.algo.RegTree.{GradPair, RegTNodeStat}
import com.tencent.angel.ml.math.vector.DenseFloatVector
import com.tencent.angel.ml.param.FPGBDTParam
import org.apache.commons.logging.LogFactory

/**
  * Created by ccchengff on 2017/12/2.
  */
/*class HistogramBuilder(controller: FPGBDTController,
                       param: FPGBDTParam, trainDataStore: TrainDataStore,
                       activeNodeSet: util.Set[Int], builderId: Int) extends Runnable {
  private var finished: Boolean = false

  override def run(): Unit = {
    val sampleFeats: Array[Int] = controller.fset.get(controller.currentTree)
    val insToNode: Array[Int] = controller.insToNode
    val gradPairs: util.List[GradPair] = controller.gradPairs
    val histograms: util.Map[Int, DenseFloatVector] = controller.histograms.get(controller.currentTree)
    val nodeStats: util.List[RegTNodeStat] = controller.forest(controller.currentTree).stats
    // 1. get responsible feature range
    val numThread: Int = param.numThread
    val fStart: Int = builderId * (sampleFeats.length / numThread)
    val fEnd: Int = if (builderId + 1 == numThread) sampleFeats.length else
      fStart +  (sampleFeats.length / numThread)
    val LOG = LogFactory.getLog(classOf[HistogramBuilder])
    LOG.info(s"Builder[$builderId] responsible range[${sampleFeats(fStart)}-${sampleFeats(fEnd - 1)}]")
    // 2. build histograms
    for (i <- fStart until fEnd) {
      // 2.1. get info of current feature
      val fid: Int = sampleFeats(i)
      //val indices: Array[Int] = trainDataStore.getFeatIndices(fid)
      //val bins: Array[Int] = trainDataStore.getFeatBins(fid)
      val (indices, bins) = trainDataStore.getFeatureRow(fid)
      val nnz: Int = indices.length
      val gradOffset: Int = i * param.numSplit * 2
      val hessOffset: Int = gradOffset + param.numSplit
      var gradTaken: Float = 0.0f
      var hessTaken: Float = 0.0f
      LOG.info(s"Builder[$builderId] feature[$fid] nnz[$nnz] gradOffset[$gradOffset] hessOffset[$hessOffset]")
      // 2.2. loop non-zero instances, add to histogram, and record the gradients taken
      for (j <- 0 until nnz) {
        val insIdx: Int = indices(j)
        val nid: Int = insToNode(insIdx)
        if (activeNodeSet.contains(nid)) {
          val hist: DenseFloatVector = histograms.get(nid)
          val binIdx: Int = bins(j)
          val gradIdx: Int = gradOffset + binIdx
          val hessIdx: Int = hessOffset + binIdx
          val gradPair: GradPair = gradPairs.get(insIdx)
          hist.set(gradIdx, hist.get(gradIdx) + gradPair.getGrad)
          hist.set(hessIdx, hist.get(hessIdx) + gradPair.getHess)
          gradTaken += gradPair.getGrad
          hessTaken += gradPair.getHess
        }
      }
      // 2.3. add remaining grad and hess to zero bin
      val zeroIdx: Int = trainDataStore.getZeroBin(fid)
      val gradIdx: Int = gradOffset + zeroIdx
      val hessIdx: Int = hessOffset + zeroIdx
      val iter: util.Iterator[Int] = activeNodeSet.iterator()
      while (iter.hasNext) {
        val nid: Int = iter.next()
        val nodeStat: RegTNodeStat = nodeStats.get(nid)
        val hist: DenseFloatVector = histograms.get(nid)
        hist.set(gradIdx, nodeStat.sumGrad - gradTaken)
        hist.set(hessIdx, nodeStat.sumHess - hessTaken)
        LOG.info(s"Builder[$builderId] node[$nid] zero[$zeroIdx] grad[$gradIdx, ${nodeStat.sumGrad}, $gradTaken], hess[$hessIdx, ${nodeStat.sumHess}, $hessTaken]")
      }
    }
    finished = true
  }

  def isFinished = finished
}
*/

class HistogramBuilder(controller: FPGBDTController, param: FPGBDTParam,
                       trainDataStore: TrainDataStore, nid: Int) {
  private var histogram: DenseFloatVector = _

  def getHistogram = histogram

  def build(): Unit = {
    if (param.numClass == 2) {
      binaryClassBuild()
    } else {
      multipleClassBuild(param.numClass)
    }
  }

  def binaryClassBuild(): Unit = {
    // 1. allocate histogram
    val sampleFeats: Array[Int] = controller.fset.get(controller.currentTree)
    val numSampleFeats: Int = sampleFeats.length
    val numSplit: Int = this.param.numSplit
    histogram = new DenseFloatVector(numSampleFeats * numSplit * 2)
    // 2. build histograms
    var shortcut = false
    if (nid != 0) {
      val siblingNid = if (nid % 2 == 0) nid - 1 else nid + 1
      if (controller.histograms.get(controller.currentTree).containsKey(siblingNid)) {
        val siblingHist = controller.histograms.get(controller.currentTree).get(siblingNid)
        val parentNid = (nid - 1) / 2
        val parentHist = controller.histograms.get(controller.currentTree).get(parentNid)
        for (i <- 0 until histogram.getDimension) {
          histogram.set(i, parentHist.get(i) - siblingHist.get(i))
        }
        shortcut = true
      }
    }
    if (!shortcut) {
      // 2.1. get sum of grad & hess
      val gradSum = controller.forest(controller.currentTree).stats.get(nid).sumGrad
      val hessSum = controller.forest(controller.currentTree).stats.get(nid).sumHess
      for (i <- 0 until numSampleFeats) {
        // 2.2. get info of current feature
        val fid: Int = sampleFeats(i)
        //val indices: Array[Int] = trainDataStore.getFeatRow(fid).getIndices
        //val indices: Array[Int] = trainDataStore.getFeatIndices(fid)
        //val bins: Array[Int] = trainDataStore.getFeatBins(fid)
        val (indices, bins) = trainDataStore.getFeatureRow(fid)
        val nnz: Int = indices.length
        val gradOffset: Int = i * numSplit * 2
        val hessOffset: Int = gradOffset + numSplit
        var gradTaken: Float = 0
        var hessTaken: Float = 0
        // 2.3. loop non-zero instances, add to histogram, and record the gradients taken
        for (j <- 0 until nnz) {
          val insIdx: Int = indices(j)
          if (controller.insToNode(insIdx) == nid) {
            val binIdx: Int = bins(j)
            val gradIdx: Int = gradOffset + binIdx
            val hessIdx: Int = hessOffset + binIdx
            val gradPair: GradPair = controller.gradPairs.get(insIdx)
            histogram.set(gradIdx, histogram.get(gradIdx) + gradPair.getGrad)
            histogram.set(hessIdx, histogram.get(hessIdx) + gradPair.getHess)
            gradTaken += gradPair.getGrad
            hessTaken += gradPair.getHess
          }
        }
        /* // merge sorted array schema, but slow
      var iter1: Int = nodeStart
      var iter2: Int = 0
      while (iter1 <= nodeEnd && iter2 < nnz) {
        if (this.nodeToIns(iter1) == indices(iter2)) {
          val insIdx: Int = indices(iter2)
          val binIdx: Int = bins(iter2)
          val gradIdx: Int = gradOffset + binIdx
          val hessIdx: Int = hessOffset + binIdx
          val gradPair: GradPair = gradPairs(insIdx)
          histogram.set(gradIdx, histogram.get(gradIdx) + gradPair.getGrad)
          histogram.set(hessIdx, histogram.get(hessIdx) + gradPair.getHess)
          gradTaken += gradPair.getGrad
          hessTaken += gradPair.getHess
          iter1 += 1
          iter2 += 1
        }
        else if (this.nodeToIns(iter1) < indices(iter2))
          iter1 += 1
        else
          iter2 += 1
      }*/
        /*// binary search schema, but slow
      if (nodeEnd - nodeStart + 1 < nnz) {
        // loop over all instances on current node
        for (j <- nodeStart to nodeEnd) {
          val insIdx: Int = this.instancePos(j)
          val index: Int = util.Arrays.binarySearch(indices, insIdx)
          // whether this instance has nonzero value on current feature
          if (index >= 0) {
            val binIdx: Int = bins(index)
            val gradIdx: Int = gradOffset + binIdx
            val hessIdx: Int = hessOffset + binIdx
            val gradPair: GradPair = gradPairs(insIdx)
            histogram.set(gradIdx, histogram.get(gradIdx) + gradPair.getGrad)
            histogram.set(hessIdx, histogram.get(hessIdx) + gradPair.getHess)
            gradTaken += gradPair.getGrad
            hessTaken += gradPair.getHess
          }
        }
      }
      else {
        // loop over all instances that have nonzero values on current feature
        for (j <- 0 until nnz) {
          val insIdx: Int = indices(j)
          val index: Int = util.Arrays.binarySearch(
            this.instancePos, nodeStart, nodeEnd + 1, insIdx)
          // whether this instance locates on current node
          if (index >= 0) {
            val binIdx: Int = bins(j)
            val gradIdx: Int = gradOffset + binIdx
            val hessIdx: Int = hessOffset + binIdx
            val gradPair: GradPair = gradPairs(insIdx)
            histogram.set(gradIdx, histogram.get(gradIdx) + gradPair.getGrad)
            histogram.set(hessIdx, histogram.get(hessIdx) + gradPair.getHess)
            gradTaken += gradPair.getGrad
            hessTaken += gradPair.getHess
          }
        }
      }*/
        // 2.4. add remaining grad and hess to zero bin
        val zeroIdx: Int = trainDataStore.getZeroBin(fid)
        val gradIdx: Int = gradOffset + zeroIdx
        val hessIdx: Int = hessOffset + zeroIdx
        histogram.set(gradIdx, gradSum - gradTaken)
        histogram.set(hessIdx, hessSum - hessTaken)
      }
    }
  }

  def multipleClassBuild(numClass: Int): Unit = {
    // 1. allocate histogram
    val sampleFeats: Array[Int] = controller.fset.get(controller.currentTree)
    val numSampleFeats: Int = sampleFeats.length
    val numSplit: Int = this.param.numSplit
    histogram = new DenseFloatVector(numSampleFeats * numClass * numSplit * 2)
    // 2. build histograms
    var shortcut = false
    if (nid != 0) {
      val siblingNid = if (nid % 2 == 0) nid - 1 else nid + 1
      if (controller.histograms.get(controller.currentTree).containsKey(siblingNid)) {
        val siblingHist = controller.histograms.get(controller.currentTree).get(siblingNid)
        val parentNid = (nid - 1) / 2
        val parentHist = controller.histograms.get(controller.currentTree).get(parentNid)
        for (i <- 0 until histogram.getDimension) {
          histogram.set(i, parentHist.get(i) - siblingHist.get(i))
        }
        shortcut = true
      }
    }
    if (!shortcut) {
      // 2.1. get sum of grad & hess
      val gradSum = controller.forest(controller.currentTree).stats.get(nid).sumGrad
      val hessSum = controller.forest(controller.currentTree).stats.get(nid).sumHess
      for (i <- 0 until numSampleFeats) {
        // 2.2. get info of current feature
        val fid: Int = sampleFeats(i)
        val (indices, bins) = trainDataStore.getFeatureRow(fid)
        val nnz: Int = indices.length
        val gradOffset: Int = i * numClass * numSplit * 2
        val hessOffset: Int = gradOffset + numSplit
        val gradTaken: Array[Float] = new Array[Float](numClass)
        val hessTaken: Array[Float] = new Array[Float](numClass)
        // 2.3. loop non-zero instances, add to histogram, and record the gradients taken
        for (j <- 0 until nnz) {
          val insIdx: Int = indices(j)
          if (controller.insToNode(insIdx) == nid) {
            val binIdx: Int = bins(j)
            for (k <- 0 until numClass) {
              val gradIdx: Int = gradOffset + k * numSplit * 2 + binIdx
              val hessIdx: Int = hessOffset + k * numSplit * 2 + binIdx
              val gradPair: GradPair = controller.gradPairs.get(insIdx * numClass + k)
              histogram.set(gradIdx, histogram.get(gradIdx) + gradPair.getGrad)
              histogram.set(hessIdx, histogram.get(hessIdx) + gradPair.getHess)
              gradTaken(k) += gradPair.getGrad
              hessTaken(k) += gradPair.getHess
            }
          }
        }
        // 2.4. add remaining grad and hess to zero bin
        val zeroIdx: Int = trainDataStore.getZeroBin(fid)
        for (k <- 0 until numClass) {
          val gradIdx: Int = gradOffset + k * numSplit * 2 + zeroIdx
          val hessIdx: Int = hessOffset + k * numSplit * 2 + zeroIdx
          histogram.set(gradIdx, gradSum - gradTaken(k))
          histogram.set(hessIdx, hessSum - hessTaken(k))
        }
      }
    }
  }
}