package com.tencent.angel.ml.FPGBDT.algo

import java.util

import com.tencent.angel.ml.FPGBDT.algo.FPRegTreeDataStore.TrainDataStore
import com.tencent.angel.ml.GBDT.algo.RegTree.{GradPair, RegTNodeStat}
import com.tencent.angel.ml.math.vector.DenseFloatVector
import com.tencent.angel.ml.param.FPGBDTParam
import org.apache.commons.logging.LogFactory

/**
  * Created by ccchengff on 2017/12/2.
  */
class HistogramBuilder(controller: FPGBDTController,
                       param: FPGBDTParam, trainDataStore: TrainDataStore,
                       activeNodeSet: util.Set[Int], builderId: Int) extends Runnable {
  private var finished: Boolean = false

  override def run(): Unit = {
    val sampleFeats: Array[Int] = controller.fset.get(controller.currentTree)
    val insToNode: Array[Int] = controller.insToNode
    val gradPairs: Array[GradPair] = controller.gradPairs
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
      val indices: Array[Int] = trainDataStore.getFeatIndices(fid)
      val bins: Array[Int] = trainDataStore.getFeatBins(fid)
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
          val gradPair: GradPair = gradPairs(insIdx)
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
