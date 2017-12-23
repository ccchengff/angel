package com.tencent.angel.ml.FPGBDT.algo

import java.util
import java.util.Collections
import java.util.concurrent.{ExecutorService, Executors}

import com.tencent.angel.ml.FPGBDT.{FPGBDTLearner, FPGBDTModel}
import com.tencent.angel.ml.FPGBDT.algo.FPRegTreeDataStore.{TestDataStore, TrainDataStore}
import com.tencent.angel.ml.FPGBDT.psf.{RangeBitSetGetRowFunc, RangeBitSetGetRowResult, RangeBitSetUpdateFunc}
import com.tencent.angel.ml.FPGBDT.psf.RangeBitSetUpdateFunc.BitsUpdateParam
import com.tencent.angel.ml.GBDT.algo.RegTree.{GradPair, GradStats, RegTNodeStat, RegTree}
import com.tencent.angel.ml.GBDT.algo.tree.{SplitEntry, TNode}
import com.tencent.angel.ml.conf.MLConf
import com.tencent.angel.ml.math.vector._
import com.tencent.angel.ml.metric.{GlobalMetrics, LogErrorMetric}
import com.tencent.angel.ml.objective.Loss
import com.tencent.angel.ml.param.FPGBDTParam
import com.tencent.angel.ml.utils.Maths
import com.tencent.angel.worker.task.TaskContext
import org.apache.commons.logging.LogFactory

import scala.util.Random

/**
  * Created by ccchengff on 2017/11/16.
  */
class FPGBDTController(ctx: TaskContext, model: FPGBDTModel, param: FPGBDTParam,
                       trainDataStore: TrainDataStore, validDataStore: TestDataStore) {
  val LOG = LogFactory.getLog(classOf[FPGBDTController])

  val forest: Array[RegTree] = new Array[RegTree](param.numTree)
  var currentTree: Int = 0
  val maxNodeNum: Int = Maths.pow(2, param.maxDepth) - 1
  var phase: Int = FPGBDTPhase.NEW_TREE

  //val objFunc: ObjFunc = new RegLossObj(new Loss.BinaryLogisticLoss)
  //var gradPairs: util.List[GradPair] = new util.ArrayList[GradPair]()
  val objective = new Loss.BinaryLogisticLoss
  val gradPairs: Array[GradPair] = new Array[GradPair](trainDataStore.numInstance)
  for (label <- trainDataStore.labels) {
    if (!objective.checkLabel(label)) {
      LOG.error(objective.labelErrorMsg())
    }
  }

  var fset: util.List[Array[Int]] = _   // save all sampled feature sets for incremental training
  val activeNode: Array[Int] = new Array[Int](maxNodeNum) // active tree node, 0:inactive, 1:active, 2:ready
  val activeNodeStat: Array[Int] = new Array[Int](maxNodeNum) // >=1:running, 0:finished, -1:failed

  // map tree node to instance, each item is instance id
  // be cautious we need each span of `instancePos` in ascending order
  val nodeToIns: Array[Int] = new Array[Int](trainDataStore.numInstance)
  val nodePosStart: Array[Int] = new Array[Int](maxNodeNum)
  val nodePosEnd: Array[Int] = new Array[Int](maxNodeNum)
  // map instance to tree node, each item is tree node that it locates on
  val insToNode: Array[Int] = new Array[Int](trainDataStore.numInstance)

  val histograms: util.List[util.Map[Int, DenseFloatVector]] =
    new util.ArrayList[util.Map[Int, DenseFloatVector]](param.numTree)

  val threadPool: ExecutorService = Executors.newFixedThreadPool(param.numThread)

  /*// find splits for each feature, try to do this during dataset transpose process
  def createSketch(): Unit = {
    trainDataStore.createSketch(param.numSplit, 1)
    this.phase = FPGBDTPhase.NEW_TREE
  }*/

  // sample features for current tree
  // TODO: sample features via feature importance
  def sampleFeature(): Unit = {
    LOG.info("------Sample feature------")
    val start = System.currentTimeMillis()
    if (param.colSample < 1) {
      val featList = new util.ArrayList[java.lang.Integer]()
      do {
        for (fid <- param.featLo until param.featHi) {
          val prob = param.colSample  // TODO: change probability via feature importance
          if (Random.nextFloat() < prob) {
            featList.add(fid)
          }
        }
      } while (featList.size() == 0)
      if (fset == null) {
        fset = new util.ArrayList[Array[Int]]()
      }
      fset.add(Maths.intList2Arr(featList))
    }
    else if (fset == null) { // create once
      fset = new util.ArrayList[Array[Int]]()
      val fullSet = (param.featLo until param.featHi).toArray
      for (tree <- 0 until param.numTree) {
        fset.add(fullSet)
      }
    }
    LOG.info(s"Sample feature cost: ${System.currentTimeMillis() - start} ms, "
      + s"sample ratio ${param.colSample}, sample ${fset.get(currentTree).length} features")
  }

  // calculate gradients of all instances, push root node stats to PS
  def calGradPairs(): Unit = {
    LOG.info(s"------Calculate grad pairs------")
    var gradSum: Float = 0.0f
    var hessSum: Float = 0.0f
    for (insIdx <- 0 until trainDataStore.numInstance) {
      val pred: Float = trainDataStore.getPred(insIdx)
      val label: Float = trainDataStore.getLabel(insIdx)
      val weight: Float = trainDataStore.getWeight(insIdx)
      val prob: Float = objective.transPred(pred)
      val grad: Float = objective.firOrderGrad(prob, label) * weight
      val hess: Float = objective.secOrderGrad(prob, label) * weight
      gradPairs(insIdx) = new GradPair(grad, hess)
      gradSum += grad
      hessSum += hess
    }
    val rootStats: GradStats = new GradStats(gradSum, hessSum)
    this.forest(this.currentTree).stats.get(0).setStats(rootStats)
    LOG.info(s"Root[0] sumGrad[$gradSum], sumHess[$hessSum], "
      + s"gain[${rootStats.calcGain(param)}]")
    if (ctx.getTaskIndex == 0) {
      updateNodeGradStats(0, rootStats)
    }
    val needFlushMatrices: util.Set[String] = new util.HashSet[String]()
    needFlushMatrices.add(FPGBDTModel.NODE_GRAD_MAT)
    clockAllMatrices(needFlushMatrices, true)
  }

  // create new tree
  def createNewTree(): Unit = {
    LOG.info("------Create new tree------")
    val createStart = System.currentTimeMillis()
    // 1. create new tree, initialize tree nodes and node status
    val tree = new RegTree(param)
    tree.initTreeNodes()
    this.forest(this.currentTree) = tree
    // 2. sample features
    sampleFeature()
    // 3. reserve histogram of each node
    val histMap = new util.HashMap[Int, DenseFloatVector]()
    this.histograms.add(histMap)
    // FOR THE SAKE OF MEMORY, ONLY WHEN DEBUGGING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (this.currentTree > 0) this.histograms.set(this.currentTree - 1, null)
    // 4. reset active tree nodes, set all nodes to be inactive
    for (nid <- 0 until this.maxNodeNum) {
      resetActiveTNode(nid)
    }
    // 5. set root node to be active
    addReadyNode(0)
    // 6. reset instance position, set the root node's span
    for (i <- 0 until trainDataStore.numInstance) {
      this.nodeToIns(i) = i
    }
    this.nodePosStart(0) = 0
    this.nodePosEnd(0) = trainDataStore.numInstance - 1
    for (nid <- 1 until this.maxNodeNum) {
      this.nodePosStart(nid) = -1
      this.nodePosEnd(nid) = -1
    }
    util.Arrays.fill(this.insToNode, 0)
    // 7. cal grad pairs, sum up as root node grad stats, push to PS
    calGradPairs()
    // 8. set phase
    this.phase = FPGBDTPhase.CHOOSE_ACTIVE
    LOG.info(s"Create new tree cost ${System.currentTimeMillis() - createStart} ms")
  }

  // choose a node to split
  def chooseActive(): Unit = {
    LOG.info("------Choose active nodes------")
    var hasActive: Boolean = false
    val readyList: util.List[Int] = new util.ArrayList[Int]()
    val activeList: util.List[Int] = new util.ArrayList[Int]()
    for (nid <- 0 until this.maxNodeNum) {
      if (this.activeNode(nid) == 2) {
        readyList.add(nid)
        if (2 * nid + 1 >= this.maxNodeNum) {
          resetActiveTNode(nid)
          val baseWeight: Float = this.forest(this.currentTree).stats.get(nid).baseWeight
          setNodeToLeaf(nid, baseWeight)
        }
        else {
          if (this.nodePosEnd(nid) - this.nodePosStart(nid) >= 1000) {
            // currently we use level-wise training, so we active it
            addActiveNode(nid)
            hasActive = true
            activeList.add(nid)
          }
          else {
            val leafValue = this.forest(this.currentTree).stats.get(nid).baseWeight
            setNodeToLeaf(nid, leafValue)
            resetActiveTNode(nid)
          }
        }
      }
    }
    if (hasActive) {
      LOG.info(s"${readyList.size()} ready nodes: [" + readyList.toArray().mkString(", ") + "]")
      LOG.info(s"${activeList.size()} active nodes: [" + activeList.toArray().mkString(", ") + "]")
      this.phase = FPGBDTPhase.RUN_ACTIVE
    }
    else {
      LOG.info("No active nodes")
      this.phase = FPGBDTPhase.FINISH_TREE
    }
  }

  // run nodes to be split, cal grad and build hist
  def runActiveNodes(): Unit = {
    LOG.info("------Run active node------")
    val start = System.currentTimeMillis()
    // 1. for each active node, cal grad and build hist
    //val sampleFeats: Array[Int] = this.fset.get(currentTree)
    //val numSampleFeats: Int = sampleFeats.length
    //val numSplit: Int = this.param.numSplit
    //val activeNodeSet: util.Set[Int] = new util.HashSet[Int]()
    //val histogramMap: util.Map[Int, DenseFloatVector] = this.histograms.get(this.currentTree)
    for (nid <- 0 until this.maxNodeNum) {
      if (this.activeNode(nid) == 1) {
        // set status to batch num
        // TODO: multi-thread
        this.activeNodeStat(nid) = 1
        this.histograms.get(this.currentTree)
          .put(nid, buildHistogram(nid))
        //activeNodeSet.add(nid)
        //val hist = new DenseFloatVector(numSampleFeats * numSplit * 2)
        //histogramMap.put(nid, hist)
      }
    }
    //val builders = new Array[HistogramBuilder](param.numThread)
    //LOG.info("Active nodes: " + activeNodeSet.toArray.mkString(" ,"))
    //for (i <- 0 until param.numThread) {
    //  builders(i) = new HistogramBuilder(this, param, trainDataStore, activeNodeSet, i)
    //  threadPool.submit(builders(i))
    //}
    // 2. check if all threads finished
    //var allFinished: Boolean = false
    //do {
    //  allFinished = true
    //  var builderId: Int = 0
    //  while (builderId < param.numThread && allFinished) {
    //    allFinished &= builders(builderId).isFinished
    //    builderId += 1
    //  }
    //} while (!allFinished)
    var hasRunning: Boolean = true
    do {
      hasRunning = false
      var nid: Int = 0
      while (nid < this.maxNodeNum && !hasRunning) {
        val stat: Int = this.activeNodeStat(nid)
        if (stat >= 1) {
          hasRunning = true
        }
        else if (stat == -1) {
          LOG.error(s"Failed to build histogram of node[$nid]")
        }
        nid += 1
      }
      if (hasRunning) {
        LOG.debug("Current has running thread(s)")
      }
    } while (hasRunning)
    // 3. all finished, turn into next phase
    this.phase = FPGBDTPhase.FIND_SPLIT
    LOG.info(s"Run active node cost ${System.currentTimeMillis() - start} ms")
  }

  // build histogram for a node
  def buildHistogram(nid: Int): DenseFloatVector = {
    LOG.info(s"------Build histogram of node[$nid]------")
    val buildStart = System.currentTimeMillis()
    // 1. allocate histogram
    val sampleFeats: Array[Int] = this.fset.get(currentTree)
    val numSampleFeats: Int = sampleFeats.length
    val numSplit: Int = this.param.numSplit
    val histogram: DenseFloatVector = new DenseFloatVector(numSampleFeats * numSplit * 2)
    // 2. get sum of grad & hess
    val gradSum = this.forest(this.currentTree).stats.get(nid).sumGrad
    val hessSum = this.forest(this.currentTree).stats.get(nid).sumHess
    // 3. build histograms
    for (i <- 0 until numSampleFeats) {
      // 3.1. get info of current feature
      val fid: Int = sampleFeats(i)
      //val indices: Array[Int] = trainDataStore.getFeatRow(fid).getIndices
      val indices: Array[Int] = trainDataStore.getFeatIndices(fid)
      val bins: Array[Int] = trainDataStore.getFeatBins(fid)
      val nnz: Int = indices.length
      val gradOffset: Int = i * numSplit * 2
      val hessOffset: Int = gradOffset + numSplit
      var gradTaken: Float = 0
      var hessTaken: Float = 0
      // 3.2. loop non-zero instances, add to histogram, and record the gradients taken
      for (j <- 0 until nnz) {
        val insIdx: Int = indices(j)
        if (this.insToNode(insIdx) == nid) {
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
      // 3.3. add remaining grad and hess to zero bin
      val zeroIdx: Int = trainDataStore.getZeroBin(fid)
      val gradIdx: Int = gradOffset + zeroIdx
      val hessIdx: Int = hessOffset + zeroIdx
      histogram.set(gradIdx, gradSum - gradTaken)
      histogram.set(hessIdx, hessSum - hessTaken)
    }
    this.activeNodeStat(nid) -= 1
    LOG.info(s"Build histogram cost ${System.currentTimeMillis() - buildStart} ms")
    histogram
  }

  // find best splits of active nodes, first find local best, then pull global best
  def findSplit(): Unit = {
    LOG.info("------Find split------")
    val start = System.currentTimeMillis()
    val needFlushMatrixSet: util.Set[String] = new util.HashSet[String]()
    // 1. find local best splits
    val splitFidVec: SparseIntVector = new SparseIntVector(this.maxNodeNum)
    val splitFvalueVec: SparseFloatVector = new SparseFloatVector(this.maxNodeNum)
    val splitGainVec: SparseFloatVector = new SparseFloatVector(this.maxNodeNum)
    for (nid <- 0 until this.maxNodeNum) {
      if (this.activeNode(nid) == 1) {
        val splitEntry: SplitEntry = findLocalBestSplit(nid)
        splitFidVec.set(nid, splitEntry.fid)
        splitFvalueVec.set(nid, splitEntry.fvalue)
        splitGainVec.set(nid, splitEntry.lossChg)
      }
    }
    // 2. push local best splits to PS
    // 2.1. split feature id
    val splitFid = model.getPSModel(FPGBDTModel.LOCAL_FEAT_MAT)
    splitFid.increment(ctx.getTaskIndex, splitFidVec)
    // 2.2. split feature value
    val splitFvalue = model.getPSModel(FPGBDTModel.LOCAL_VALUE_MAT)
    splitFvalue.increment(ctx.getTaskIndex, splitFvalueVec)
    // 2.3. split loss gain
    val splitGain = model.getPSModel(FPGBDTModel.LOCAL_GAIN_MAT)
    splitGain.increment(ctx.getTaskIndex, splitGainVec)
    // 2.4. clock
    needFlushMatrixSet.add(FPGBDTModel.LOCAL_FEAT_MAT)
    needFlushMatrixSet.add(FPGBDTModel.LOCAL_VALUE_MAT)
    needFlushMatrixSet.add(FPGBDTModel.LOCAL_GAIN_MAT)
    needFlushMatrixSet.add(FPGBDTModel.NODE_GRAD_MAT)
    clockAllMatrices(needFlushMatrixSet, true)
    // 3. leader worker pull all local best splits, find the global best split
    // TODO: find global best on server
    // do not clear node grad stats
    needFlushMatrixSet.remove(FPGBDTModel.NODE_GRAD_MAT)
    if (ctx.getTaskIndex == 0) {
      // 3.1. pull local best splits
      val splitFidMat: util.List[SparseIntVector] = new util.ArrayList[SparseIntVector](param.numWorker)
      val splitFvalueMat: util.List[SparseFloatVector] = new util.ArrayList[SparseFloatVector](param.numWorker)
      val splitGainMat: util.List[SparseFloatVector] = new util.ArrayList[SparseFloatVector](param.numWorker)
      for (wid <- 0 until param.numWorker) {
        splitFidMat.add(splitFid.getRow(wid).asInstanceOf[SparseIntVector])
        splitFvalueMat.add(splitFvalue.getRow(wid).asInstanceOf[SparseFloatVector])
        splitGainMat.add(splitGain.getRow(wid).asInstanceOf[SparseFloatVector])
      }
      // 3.2. clear local best split matrices for next update
      clearAllMatrices(needFlushMatrixSet)
      // 3.2. find global best splits
      val splitFidVec: SparseIntVector = new SparseIntVector(this.maxNodeNum)
      val splitFvalueVec: SparseFloatVector = new SparseFloatVector(this.maxNodeNum)
      val splitGainVec: SparseFloatVector = new SparseFloatVector(this.maxNodeNum)
      for (nid <- 0 until this.maxNodeNum) {
        if (this.activeNode(nid) == 1) {
          val bestSplit: SplitEntry = new SplitEntry()
          for (wid <- 0 until param.numWorker) {
            val fid: Int = splitFidMat.get(wid).get(nid)
            val fvalue: Float = splitFvalueMat.get(wid).get(nid)
            val lossChg: Float = splitGainMat.get(wid).get(nid)
            val curSplit: SplitEntry = new SplitEntry(fid, fvalue, lossChg)
            bestSplit.update(curSplit)
          }
          splitFidVec.set(nid, bestSplit.fid)
          splitFvalueVec.set(nid, bestSplit.fvalue)
          splitGainVec.set(nid, bestSplit.lossChg)
        }
      }
      // 3.3. push global best splits to PS
      // 3.3.1. split feature id
      val globalSplitFid = model.getPSModel(FPGBDTModel.GLOBAL_FEAT_MAT)
      globalSplitFid.increment(this.currentTree, splitFidVec)
      // 3.3.2. split feature value
      val globalSplitFvalue = model.getPSModel(FPGBDTModel.GLOBAL_VALUE_MAT)
      globalSplitFvalue.increment(this.currentTree, splitFvalueVec)
      // 3.3.3. split loss gain
      val globalSplitGain = model.getPSModel(FPGBDTModel.GLOBAL_GAIN_MAT)
      globalSplitGain.increment(this.currentTree, splitGainVec)
    }
    // 3.4. clock
    needFlushMatrixSet.add(FPGBDTModel.GLOBAL_FEAT_MAT)
    needFlushMatrixSet.add(FPGBDTModel.GLOBAL_VALUE_MAT)
    needFlushMatrixSet.add(FPGBDTModel.GLOBAL_GAIN_MAT)
    clockAllMatrices(needFlushMatrixSet, true)
    // 4. finish current phase
    this.phase = FPGBDTPhase.AFTER_SPLIT
    LOG.info(s"Find split cost ${System.currentTimeMillis() - start} ms")
  }

  // find local best split of one node
  def findLocalBestSplit(nid: Int): SplitEntry = {
    LOG.info(s"------To find the best split of node[$nid]------")
    val splitEntry: SplitEntry = new SplitEntry()
    val hist: DenseFloatVector = histograms.get(this.currentTree).get(nid)
    if (hist == null) {
      LOG.error(s"null histogram on node[$nid]")
      return splitEntry
    }
    // 1. calculate the gradStats of the node
    val gradSum: Float = this.forest(this.currentTree).stats.get(nid).sumGrad
    val hessSum: Float = this.forest(this.currentTree).stats.get(nid).sumHess
    val nodeStats: GradStats = new GradStats(gradSum, hessSum)
    /*var gradSum: Float = 0.0f
    var hessSum: Float = 0.0f
    for (i <- 0 until param.numSplit) {
      gradSum += hist.get(i)
      hessSum += hist.get(i + param.numSplit)
    }
    val nodeStats: GradStats = new GradStats(gradSum, hessSum)
    LOG.info(s"Node[$nid] sumGrad[$gradSum], sumHess[$hessSum], "
      + s"gain[${nodeStats.calcGain(param)}]")
    if (nid == 0 && ctx.getTaskIndex == 0) {
      updateNodeGradStats(nid, nodeStats)
    }*/
    // 2. loop over features
    val sampleFeats: Array[Int] = this.fset.get(this.currentTree)
    val numSampleFeats: Int = sampleFeats.length
    for (i <- 0 until numSampleFeats) {
      val fid: Int = sampleFeats(i)
      val offset: Int = i * param.numSplit * 2
      val curSplit: SplitEntry = findBestSplitOfOneFeature(fid, hist, offset, nodeStats)
      splitEntry.update(curSplit)
    }

    LOG.info(s"Local best split of node[$nid]: fid[${splitEntry.fid}], "
      + s"fvalue[${splitEntry.fvalue}], loss gain[${splitEntry.lossChg}]")
    splitEntry
  }

  // find the best split result of one feature
  def findBestSplitOfOneFeature(fid: Int, hist: DenseFloatVector,
                                offset: Int, nodeStats: GradStats): SplitEntry = {
    val splitEntry: SplitEntry = new SplitEntry()
    if (offset + 2 * param.numSplit > hist.getDimension) {
      LOG.error("index out of grad histogram size")
      return splitEntry
    }
    // 1. set the feature id
    splitEntry.setFid(fid)
    // 2. create the best left stats and right stats
    val bestLeftStats: GradStats = new GradStats()
    val bestRightStats: GradStats = new GradStats()
    // 3. calculate gain of node, create empty grad stats
    val nodeGain: Float = nodeStats.calcGain(param)
    val leftStats: GradStats = new GradStats()
    val rightStats: GradStats = new GradStats()
    // 4. loop over histogram and find the best
    for (histIdx <- offset until (offset + param.numSplit - 1)) {
      // 4.1. get grad and hess
      val grad: Float = hist.get(histIdx)
      val hess: Float = hist.get(histIdx + param.numSplit)
      leftStats.add(grad, hess)
      // 4.2. check whether we can split
      if (leftStats.sumHess >= param.minChildWeight) {
        // right = root - left
        rightStats.setSubstract(nodeStats, leftStats)
        if (rightStats.sumHess >= param.minChildWeight) {
          // 4.3. calculate gain after current split
          val lossChg: Float = leftStats.calcGain(param) +
            rightStats.calcGain(param) - nodeGain
          // 4.4. check whether we should update the split result
          val splitIdx: Int = histIdx - offset + 1
          if (splitEntry.update(lossChg, fid, trainDataStore.getSplit(fid, splitIdx))) {
            bestLeftStats.update(leftStats.sumGrad, leftStats.sumHess)
            bestRightStats.update(rightStats.sumGrad, rightStats.sumHess)
          }
        }
      }
    }
    // 5. set best left and right grad stats
    splitEntry.leftGradStat = bestLeftStats
    splitEntry.rightGradStat = bestRightStats
    splitEntry
  }

  // after find best split, time to do actual split
  def afterSplit(): Unit = {
    LOG.info("------After split------")
    val start = System.currentTimeMillis()
    // 1. pull global best splits
    // 1.1. split feature id
    val globalSplitFid = model.getPSModel(FPGBDTModel.GLOBAL_FEAT_MAT)
    val bestSplitFidVec = globalSplitFid.getRow(this.currentTree).asInstanceOf[DenseIntVector]
    // 1.2. split feature value
    val globalSplitFvalue = model.getPSModel(FPGBDTModel.GLOBAL_VALUE_MAT)
    val bestSplitFvalueVec = globalSplitFvalue.getRow(this.currentTree).asInstanceOf[DenseFloatVector]
    // 1.3. split loss gain
    val globalSplitGain = model.getPSModel(FPGBDTModel.GLOBAL_GAIN_MAT)
    val bestSplitGainVec = globalSplitGain.getRow(this.currentTree).asInstanceOf[DenseFloatVector]
    // 1.4. node grad stats
    val nodeGradStats = model.getPSModel(FPGBDTModel.NODE_GRAD_MAT)
    var nodeGradStatsVec = nodeGradStats.getRow(this.currentTree).asInstanceOf[DenseFloatVector]
    LOG.info(s"Get best split entries from PS cost ${System.currentTimeMillis() - start} ms")
    // 2. reset instance position
    val resetPosStart = System.currentTimeMillis()
    // 2.0. clear previous split result
    val resultModel = model.getPSModel(FPGBDTModel.SPLIT_RESULT_MAT)
    //resultModel.zero()
    //resultModel.clock().get
    //sync()
    // 2.1. get split result of responsible nodes
    for (nid <- 0 until this.maxNodeNum) {
      if (this.activeNode(nid) == 1) {
        val fid: Int = bestSplitFidVec.get(nid)
        val fvalue: Float = bestSplitFvalueVec.get(nid)
        val lossChg: Float = bestSplitGainVec.get(nid)
        val splitEntry: SplitEntry = new SplitEntry(fid, fvalue, lossChg)
        val sumGrad: Float = nodeGradStatsVec.get(nid)
        val sumHess: Float = nodeGradStatsVec.get(nid + this.maxNodeNum)
        val nodeStat: GradStats = new GradStats(sumGrad, sumHess)
        if (param.featLo <= fid && fid < param.featHi) {
          // 2.1.1. get split result, represented by a bitset
          val nodeStart: Int = this.nodePosStart(nid)
          val nodeEnd: Int = this.nodePosEnd(nid)
          val bitset: RangeBitSet = new RangeBitSet(nodeStart, nodeEnd)
          val (leftChildGradStats, rightChildGradStats) =
            getSplitResult(nid, splitEntry, nodeStat, bitset)
          // 2.1.2. push split result to PS
          val matrixId = resultModel.getMatrixId()
          val bitsUpdate = new RangeBitSetUpdateFunc(
            new BitsUpdateParam(matrixId, false, bitset))
          resultModel.update(bitsUpdate).get
          // 2.1.3. push children grad stats
          updateNodeGradStats(2 * nid + 1, leftChildGradStats)
          updateNodeGradStats(2 * nid + 2, rightChildGradStats)
        }
      }
    }
    // 2.2. push local split result & children grad stats
    //val resultModel = model.getPSModel(FPGBDTModel.SPLIT_RESULT_MAT)
    //val splitResultVec = new DenseIntVector(trainDataStore.numInstance, splitResult)
    //resultModel.increment(0, splitResultVec)
    // 2.2. flush & sync
    val needFlushMatrices: util.Set[String] = new util.HashSet[String]()
    needFlushMatrices.add(FPGBDTModel.SPLIT_RESULT_MAT)
    needFlushMatrices.add(FPGBDTModel.NODE_GRAD_MAT)
    clockAllMatrices(needFlushMatrices, true)
    sync()
    // 2.3. pull global split result & children grad stats
    //splitResult = resultModel.getRow(0).asInstanceOf[DenseIntVector].getValues
    nodeGradStatsVec = nodeGradStats.getRow(this.currentTree).asInstanceOf[DenseFloatVector]
    LOG.info(s"Get split result cost ${System.currentTimeMillis() - resetPosStart} ms")
    // 3. split node
    for (nid <- 0 until this.maxNodeNum) {
      if (this.activeNode(nid) == 1) {
        // 3.1. split entry
        val fid: Int = bestSplitFidVec.get(nid)
        val fvalue: Float = bestSplitFvalueVec.get(nid)
        val lossChg: Float = bestSplitGainVec.get(nid)
        val splitEntry: SplitEntry = new SplitEntry(fid, fvalue, lossChg)
        // 3.2. node grad stats
        val sumGrad: Float = nodeGradStatsVec.get(nid)
        val sumHess: Float = nodeGradStatsVec.get(nid + this.maxNodeNum)
        val nodeStat: GradStats = new GradStats(sumGrad, sumHess)
        // 3.3. left child grad stats
        val leftChildSumGrad: Float = nodeGradStatsVec.get(2 * nid + 1)
        val leftChildSumHess: Float = nodeGradStatsVec.get(2 * nid + 1 + this.maxNodeNum)
        val leftChildStats: GradStats = new GradStats(leftChildSumGrad, leftChildSumHess)
        // 3.4. right child grad stats
        val rightChildSumGrad: Float = nodeGradStatsVec.get(2 * nid + 2)
        val rightChildSumHess: Float = nodeGradStatsVec.get(2 * nid + 2 + this.maxNodeNum)
        val rightChildStats: GradStats = new GradStats(rightChildSumGrad, rightChildSumHess)
        // 3.5. split
        splitNode(nid, splitEntry, nodeStat, leftChildStats, rightChildStats)
        if (fid != -1) {
          // 3.6. reset instance pos
          //resetInstancePos(nid, splitResult)
          val nodeStart: Int = this.nodePosStart(nid)
          val nodeEnd: Int = this.nodePosEnd(nid)
          val getFunc = new RangeBitSetGetRowFunc(resultModel.getMatrixId(), nodeStart, nodeEnd)
          val splitResult = resultModel.get(getFunc).asInstanceOf[RangeBitSetGetRowResult].getRangeBitSet
          resetInstancePos(nid, splitResult)
          // 3.7. set children as ready
          addReadyNode(2 * nid + 1)
          addReadyNode(2 * nid + 2)
        }
        // 3.8. deactivate active node
        resetActiveTNode(nid)
      }
    }
    // 4. finish current phase
    this.phase = FPGBDTPhase.CHOOSE_ACTIVE
    LOG.info(s"After split cost: ${System.currentTimeMillis() - start} ms")
  }

  // get split result of responsible nodes
  def getSplitResult(nid: Int, splitEntry: SplitEntry,
                     nodeStat: GradStats, bitset: RangeBitSet): (GradStats, GradStats) = {
    val splitFid: Int = splitEntry.getFid
    val splitFvalue: Float = splitEntry.getFvalue
    LOG.info(s"------Get split result of node[$nid]: fid[$splitFid] fvalue[$splitFvalue]------")
    val nodeStart: Int = this.nodePosStart(nid)
    val nodeEnd: Int = this.nodePosEnd(nid)
    LOG.info(s"Node[$nid] span: [$nodeStart-$nodeEnd]")
    // if no instance on this node, we do nothing
    var leftChildSumGrad: Float = 0.0f
    var rightChildSumGrad: Float = 0.0f
    var leftChildSumHess: Float = 0.0f
    var rightChildSumHess: Float = 0.0f
    if (nodeStart <= nodeEnd) {
      var rightCount: Int = 0
      //val indices: Array[Int] = trainDataStore.getFeatRow(splitFid).getIndices
      //val values: Array[Double] = trainDataStore.getFeatRow(splitFid).getValues
      val indices: Array[Int] = trainDataStore.getFeatIndices(splitFid)
      val bins: Array[Int] = trainDataStore.getFeatBins(splitFid)
      val nnz: Int = indices.length
      // find out instances that should be in right child
      /*if (splitFvalue < 0.0) {
        // default to right child
        for (j <- nodeStart to nodeEnd) {
          bitset.set(j)
        }
        rightCount = nodeEnd - nodeStart + 1
        // pick out the instances that should be in left child
        // loop over nonzero instance and check whether it locates on current node
        var searchFrom: Int = nodeStart
        for (j <- 0 until nnz) {
          val insIdx: Int = indices(j)
          if (this.insToNode(insIdx) == nid && values(j) <= splitFvalue) {
            val index: Int = util.Arrays.binarySearch(
              this.nodeToIns, searchFrom, nodeEnd + 1, insIdx)
            bitset.clear(index)
            rightCount -= 1
            leftChildSumGrad += this.gradPairs(insIdx).getGrad
            leftChildSumHess += this.gradPairs(insIdx).getHess
            searchFrom = index + 1
          }
        }
        rightChildSumGrad = nodeStat.getSumGrad - leftChildSumGrad
        rightChildSumHess = nodeStat.getSumHess - leftChildSumHess
      }
      else {
        // default to left, pick out the instances that should be in right child
        // loop over nonzero instance and check whether it locates on current node
        var searchFrom: Int = nodeStart
        for (j <- 0 until nnz) {
          val insIdx: Int = indices(j)
          if (this.insToNode(insIdx) == nid && values(j) > splitFvalue) {
            val index: Int = util.Arrays.binarySearch(
              this.nodeToIns, searchFrom, nodeEnd + 1, insIdx)
            bitset.set(index)
            rightCount += 1
            rightChildSumGrad += this.gradPairs(insIdx).getGrad
            rightChildSumHess += this.gradPairs(insIdx).getHess
            searchFrom = index + 1
          }
        }
        rightChildSumGrad = nodeStat.getSumGrad - leftChildSumGrad
        rightChildSumHess = nodeStat.getSumHess - leftChildSumHess
      }*/

      /* // merge sorted array schema, but slow
      var iter1: Int = nodeStart
      var iter2: Int = nnz
      if (splitFvalue < 0.0) {
        // default to right child
        for (j <- nodeStart to nodeEnd) {
          bitset.set(j)
        }
        rightCount = nodeEnd - nodeStart + 1
        // pick out the instances that should be in left child
        while (iter1 <= nodeEnd && iter2 < nnz) {
          if (this.nodeToIns(iter1) == indices(iter2)) {
            if (values(iter2) <= splitFvalue) {
              bitset.clear(iter1)
              rightCount -= 1
              val insIdx: Int = indices(iter2)
              leftChildSumGrad += this.gradPairs(insIdx).getGrad
              leftChildSumHess += this.gradPairs(insIdx).getHess
              iter1 += 1
              iter2 += 1
            }
          }
          else if (this.nodeToIns(iter1) < indices(iter2))
            iter1 += 1
          else
            iter2 += 1
        }
        rightChildSumGrad = nodeStat.getSumGrad - leftChildSumGrad
        rightChildSumHess = nodeStat.getSumHess - leftChildSumHess
      }
      else {
        // default to left child, find out the instances that should be in right child
        while (iter1 <= nodeEnd && iter2 < nnz) {
          if (this.nodeToIns(iter1) == indices(iter2)) {
            if (values(iter2) > splitFvalue) {
              bitset.set(iter1)
              rightCount += 1
              val insIdx: Int = indices(iter2)
              rightChildSumGrad += this.gradPairs(insIdx).getGrad
              rightChildSumHess += this.gradPairs(insIdx).getHess
            }
          }
        }
        leftChildSumGrad = nodeStat.getSumGrad - rightChildSumGrad
        leftChildSumHess = nodeStat.getSumHess - rightChildSumHess
      }*/

      // binary search schema, but slow
      if (nodeEnd - nodeStart + 1 < nnz) {
        // loop over all instances on current node
        var searchFrom: Int = 0
        for (j <- nodeStart to nodeEnd) {
          val insIdx: Int = this.nodeToIns(j)
          //val index: Int = util.Arrays.binarySearch(indices, insIdx)
          val index: Int = util.Arrays.binarySearch(indices, searchFrom, nnz, insIdx)
          // whether this instance has nonzero value on current feature
          if (index >= 0) {
            val insValue: Float = trainDataStore.getSplit(splitFid, bins(index))
            if (insValue >= splitFvalue) {
              bitset.set(j)
              rightCount += 1
              rightChildSumGrad += this.gradPairs(insIdx).getGrad
              rightChildSumHess += this.gradPairs(insIdx).getHess
            }
            searchFrom = index + 1
          }
          else if (splitFvalue < 0.0f) { // default to right child
            bitset.set(j)
            rightCount += 1
            rightChildSumGrad += this.gradPairs(insIdx).getGrad
            rightChildSumHess += this.gradPairs(insIdx).getHess
          }
        }
        leftChildSumGrad = nodeStat.getSumGrad - rightChildSumGrad
        leftChildSumHess = nodeStat.getSumHess - rightChildSumHess
      }
      else {
        // loop over all instances that have nonzero values on current feature
        if (splitFvalue < 0.0f) {
          // default to right child
          for (j <- nodeStart to nodeEnd) {
            bitset.set(j)
            rightCount += 1
          }
          // pick out the instances that should be in the left child
          var searchFrom: Int = nodeStart
          for (j <- 0 until nnz) {
            val insValue: Float = trainDataStore.getSplit(splitFid, bins(j))
            if (insValue < splitFvalue) {
              val insIdx: Int = indices(j)
              //val index: Int = util.Arrays.binarySearch(
              //  this.nodeToIns, nodeStart, nodeEnd + 1, insIdx)
              val index: Int = util.Arrays.binarySearch(
                this.nodeToIns, searchFrom, nodeEnd + 1, insIdx)
              // whether this instance locates on current node
              if (index >= 0) {
                bitset.clear(index)
                rightCount -= 1
                leftChildSumGrad += this.gradPairs(insIdx).getGrad
                leftChildSumHess += this.gradPairs(insIdx).getHess
                searchFrom = index + 1
              }
            }
          }
          rightChildSumGrad = nodeStat.getSumGrad - leftChildSumGrad
          rightChildSumHess = nodeStat.getSumHess - leftChildSumHess
        }
        else {
          // default to left child
          var searchFrom: Int = nodeStart
          for (j <- 0 until nnz) {
            val insValue: Float = trainDataStore.getSplit(splitFid, bins(j))
            if (insValue >= splitFvalue) {
              val insIdx: Int = indices(j)
              //val index: Int = util.Arrays.binarySearch(
              //  this.nodeToIns, nodeStart, nodeEnd + 1, insIdx)
              val index: Int = util.Arrays.binarySearch(
                this.nodeToIns, searchFrom, nodeEnd + 1, insIdx)
              // whether this instance locates on current node
              if (index >= 0) {
                bitset.set(index)
                rightCount += 1
                rightChildSumGrad += this.gradPairs(insIdx).getGrad
                rightChildSumHess += this.gradPairs(insIdx).getHess
                searchFrom = index + 1
              }
            }
          }
          leftChildSumGrad = nodeStat.getSumGrad - rightChildSumGrad
          leftChildSumHess = nodeStat.getSumHess - rightChildSumHess
        }
      }
      LOG.info(s"Node[$nid] split: left with grad stats[$leftChildSumGrad, $leftChildSumHess], " +
        s"right child with grad stats[$rightChildSumGrad, $rightChildSumHess]")
    }
    val leftChildGradStats = new GradStats(leftChildSumGrad, leftChildSumHess)
    val rightChildGradStats = new GradStats(rightChildSumGrad, rightChildSumHess)
    (leftChildGradStats, rightChildGradStats)
  }

  // split a node and set its information
  def splitNode(nid: Int, splitEntry: SplitEntry, nodeStats: GradStats,
                leftGradStats: GradStats, rightGradStats: GradStats): Unit ={
    LOG.info(s"Split node[$nid]: feature[${splitEntry.getFid}], value[${splitEntry.getFvalue}], "
      + s"lossChg[${splitEntry.getLossChg}], sumGrad[${nodeStats.getSumGrad}], sumHess[${nodeStats.getSumHess}]")
    // 1. set split info and grad stats to this node
    val stats: RegTNodeStat = this.forest(this.currentTree).stats.get(nid)
    stats.setSplitEntry(splitEntry)
    stats.setLossChg(splitEntry.lossChg)
    stats.setStats(nodeStats)
    if (splitEntry.getFid != -1) {
      // 2. set children nodes of this node
      val node: TNode = this.forest(this.currentTree).nodes.get(nid)
      node.setLeftChild(2 * nid + 1)
      node.setRightChild(2 * nid + 2)
      // 3. create children nodes
      val leftChild: TNode = new TNode(2 * nid + 1, nid, -1, -1)
      val rightChild: TNode = new TNode(2 * nid + 2, nid, -1, -1)
      this.forest(this.currentTree).nodes.set(2 * nid + 1, leftChild)
      this.forest(this.currentTree).nodes.set(2 * nid + 2, rightChild)
      // 4. create node stats for children nodes, and add them to the tree
      val leftChildStat: RegTNodeStat = new RegTNodeStat(param)
      val rightChildStat: RegTNodeStat = new RegTNodeStat(param)
      leftChildStat.setStats(leftGradStats)
      rightChildStat.setStats(rightGradStats)
      this.forest(this.currentTree).stats.set(2 * nid + 1, leftChildStat)
      this.forest(this.currentTree).stats.set(2 * nid + 2, rightChildStat)
    }
    else {
      // 5. set node as leaf
      setNodeToLeaf(nid, nodeStats.calcGain(param))
    }
  }

  // reset instance position thru global split result
  def resetInstancePos(nid: Int, splitResult: RangeBitSet): Unit = {
    val nodeStart: Int = this.nodePosStart(nid)
    val nodeEnd: Int = this.nodePosEnd(nid)
    LOG.info(s"------Reset instance position of node[$nid] with span [$nodeStart-$nodeEnd]------")
    // if no instance on this node
    if (nodeStart > nodeEnd) {
      // set the span of left child
      this.nodePosStart(2 * nid + 1) = nodeStart
      this.nodePosEnd(2 * nid + 1) = nodeEnd
      // set the span of right child
      this.nodePosStart(2 * nid + 2) = nodeStart
      this.nodePosEnd(2 * nid + 2) = nodeEnd
    }
    else {
      val leftSpan: Array[Int] = new Array[Int](nodeEnd - nodeStart + 1)
      val rightSpan: Array[Int] = new Array[Int](nodeEnd - nodeStart + 1)
      val leftChildNid: Int = 2 * nid + 1
      val rightChildNid: Int = 2 * nid + 2
      var leftCount: Int = 0
      var rightCount: Int = 0
      for (j <- nodeStart until nodeEnd) {
        val insIdx: Int = this.nodeToIns(j)
        if (!splitResult.get(j)) {
          leftSpan(leftCount) = insIdx
          this.insToNode(insIdx) = leftChildNid
          leftCount += 1
        }
        else {
          rightSpan(rightCount) = insIdx
          this.insToNode(insIdx) = rightChildNid
          rightCount += 1
        }
      }
      System.arraycopy(leftSpan, 0, this.nodeToIns, nodeStart, leftCount)
      System.arraycopy(rightSpan, 0, this.nodeToIns, nodeStart + leftCount, rightCount)
      this.nodePosStart(2 * nid + 1) = nodeStart
      this.nodePosStart(2 * nid + 2) = nodeStart + leftCount
      this.nodePosEnd(2 * nid + 1) = nodeStart + leftCount - 1
      this.nodePosEnd(2 * nid + 2) = nodeEnd

      /*var left: Int = nodeStart
      var right: Int = nodeEnd
      while (left < right) {
        // 1. left to right, find the first instance that should be in the right child
        var leftInsIdx: Int = this.nodeToIns(left)
        while (left < right && !splitResult.get(left)) {
          this.insToNode(leftInsIdx) = leftChildNid
          left += 1
          leftInsIdx = this.nodeToIns(left)
        }
        // 2. right to left, find the first instance that should be in the left child
        var rightInsIdx: Int = this.nodeToIns(right)
        while (left < right && splitResult.get(right)) {
          this.insToNode(rightInsIdx) = rightChildNid
          right -= 1
          rightInsIdx = this.nodeToIns(right)
        }
        // 3. swap two instances
        if (left < right) {
          this.insToNode(leftInsIdx) = rightChildNid
          this.insToNode(rightInsIdx) = leftChildNid
          this.nodeToIns(left) = rightInsIdx
          this.nodeToIns(right) = leftInsIdx
          left += 1
          right -= 1
        }
      }
      // 4. find the cut pos
      //val curInsIdx: Int = this.instancePos(left)
      val cutPos = if (!splitResult.get(left)) left else left + 1
      // 5. set the span of children
      this.nodePosStart(2 * nid + 1) = nodeStart
      this.nodePosStart(2 * nid + 2) = cutPos
      this.nodePosEnd(2 * nid + 1) = cutPos - 1
      this.nodePosEnd(2 * nid + 2) = nodeEnd*/
    }
    LOG.info(s"Left child[${2*nid+1}] span: [${this.nodePosStart(2*nid+1)}-${this.nodePosEnd(2*nid+1)}]")
    LOG.info(s"Right child[${2*nid+2}] span: [${this.nodePosStart(2*nid+2)}-${this.nodePosEnd(2*nid+2)}]")
  }

  // set node to active
  def addActiveNode(nid: Int): Unit = {
    this.activeNode(nid) = 1
    this.activeNodeStat(nid) = 0
  }

  // set node to ready
  def addReadyNode(nid: Int): Unit = {
    this.activeNode(nid) = 2
    this.activeNodeStat(nid) = 0
  }

  // set node to be leaf
  def setNodeToLeaf(nid: Int, nodeWeight: Float): Unit = {
    LOG.info(s"Set node[$nid] as leaf node, leaf weight=$nodeWeight")
    this.forest(this.currentTree).nodes.get(nid).chgToLeaf()
    this.forest(this.currentTree).nodes.get(nid).setLeafValue(nodeWeight)
  }

  // set node to inactive
  def resetActiveTNode(nid: Int): Unit = {
    this.activeNode(nid) = 0
    this.activeNodeStat(nid) = 0
  }

  // update node's grad stats on PS, called during splitting
  def updateNodeGradStats(nid: Int, gradStats: GradStats): Unit = {
    LOG.debug(s"Update gradStats of node[$nid]: sumGrad[${gradStats.sumGrad}, sumHess[${gradStats.sumHess}]")
    // 1. create the update
    val vec: DenseFloatVector = new DenseFloatVector(2 * this.maxNodeNum)
    vec.set(nid, gradStats.sumGrad)
    vec.set(nid + this.maxNodeNum, gradStats.sumHess)
    // 2. push the update to PS
    val nodeGradStats = model.getPSModel(FPGBDTModel.NODE_GRAD_MAT)
    nodeGradStats.increment(this.currentTree, vec)
  }

  // finish current tree
  def finishCurrentTree(globalMetrics: GlobalMetrics) {
    updateInsPreds()
    updateLeafPreds()
    val train_error: Tuple1[Float] = eval(trainDataStore)
    val valid_error: Tuple1[Float] = predict(validDataStore)
    globalMetrics.metric(MLConf.TRAIN_ERROR, train_error._1)
    globalMetrics.metric(MLConf.VALID_ERROR, valid_error._1)
    this.currentTree += 1

    if (isFinished) {
      this.phase = FPGBDTPhase.FINISHED
    }
    else {
      this.phase = FPGBDTPhase.NEW_TREE
    }
  }

  // update instance predictions
  def updateInsPreds(): Unit = {
    LOG.info("------Update instance predictions------")
    val start = System.currentTimeMillis()
    for (nid <- 0 until this.maxNodeNum) {
      val node: TNode = this.forest(this.currentTree).nodes.get(nid)
      if (node != null && node.isLeaf) {
        val weight: Float = node.getLeafValue
        val nodeStart: Int = this.nodePosStart(nid)
        val nodeEnd: Int = this.nodePosEnd(nid)
        LOG.info(s"Leaf[$nid] weight: $weight, span: [$nodeStart-$nodeEnd]")
        for (i <- nodeStart to nodeEnd) {
          val insIdx: Int = this.nodeToIns(i)
          trainDataStore.preds(insIdx) += param.learningRate * weight
        }
      }
    }
    LOG.info(s"Update instance predictions cost ${System.currentTimeMillis() - start} ms")
  }

  // push leaf predictions to PS
  def updateLeafPreds() {
    LOG.info("------Update leaf node predictions------")
    val start = System.currentTimeMillis()
    val nodePreds = model.getPSModel(FPGBDTModel.NODE_PRED_MAT)
    if (ctx.getTaskIndex == 0) {
      val vec: DenseFloatVector = new DenseFloatVector(this.maxNodeNum)
      for (nid <- 0 until this.maxNodeNum) {
        val node: TNode = this.forest(this.currentTree).nodes.get(nid)
        if (node != null && node.isLeaf) {
          val weight: Float = node.getLeafValue
          vec.set(nid, weight)
        }
      }
      val nodePreds = model.getPSModel(FPGBDTModel.NODE_PRED_MAT)
      nodePreds.increment(this.currentTree, vec)
    }
    nodePreds.clock().get
    LOG.info(s"Update leaf node predictions cost ${System.currentTimeMillis() - start} ms")
  }

  // evaluate
  def eval(dataStore: TrainDataStore): Tuple1[Float] = {
    LOG.info("------Evaluation------")
    val start = System.currentTimeMillis()
    val evalMetric = new LogErrorMetric
    val error: Float = evalMetric.eval(dataStore.getPreds, dataStore.getLabels)
    LOG.info(s"Error on training dataset after tree[${this.currentTree}]: $error")
    LOG.info(s"Evaluation cost ${System.currentTimeMillis() - start} ms")
    new Tuple1[Float](error)
  }

  // predict
  def predict(dataStore: TestDataStore): Tuple1[Float] = {
    LOG.info("------Predict------")
    val start = System.currentTimeMillis()
    // 1. predict
    val numValid: Int = dataStore.getNumInstances
    for (i <- 0 until numValid) {
      val x: SparseDoubleSortedVector = dataStore.getInstance(i)
      var nid: Int = 0
      var node: TNode = this.forest(this.currentTree).nodes.get(nid)
      while (node != null && !node.isLeaf) {
        val stat: RegTNodeStat = this.forest(this.currentTree).stats.get(nid)
        val splitFid: Int = stat.splitEntry.getFid
        val splitFvalue: Float = stat.splitEntry.getFvalue
        if (x.get(splitFid) <= splitFvalue)
          nid = nid * 2 + 1
        else
          nid = nid * 2 + 2
        node = this.forest(this.currentTree).nodes.get(nid)
      }
      if (node.isLeaf) {
        dataStore.preds(i) += node.getLeafValue
      }
      else {
        LOG.error("Test instance gets into null node")
      }
    }
    // 2. evaluation
    val evalMetric = new LogErrorMetric
    val error: Float = evalMetric.eval(dataStore.getPreds, dataStore.getLabels)
    LOG.info(s"Error on validation dataset after tree[${this.currentTree}]: $error")
    LOG.info(s"Predict cost ${System.currentTimeMillis() - start} ms")
    new Tuple1[Float](error)
  }

  // set the tree phase
  def setPhase(phase: Int): Unit = {
    this.phase = phase
  }

  // check if there is active node
  def hasActiveTNode: Boolean = {
    LOG.debug("Check actice nodes: [" + this.activeNode.mkString(", ") + "]")
    var hasActive: Boolean = false
    var nid: Int = 0
    while (nid < this.maxNodeNum && !hasActive) {
      if (this.activeNode(nid) == 1)
        hasActive = true
      nid += 1
    }
    hasActive
  }

  // check if finish all the trees
  def isFinished: Boolean = this.currentTree >= param.numTree

  def sync(): Unit = {
    FPGBDTLearner.sync(model)
  }

  def clockAllMatrices(needFlushMatrices: util.Set[String], wait: Boolean): Unit = {
    FPGBDTLearner.clockAllMatrices(needFlushMatrices, model, wait)
  }

  def clearAllMatrices(needClearMatrices: util.Set[String]): Unit = {
    FPGBDTLearner.clearAllMatrices(needClearMatrices, model)
  }

}
