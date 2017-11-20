package com.tencent.angel.ml.FPGBDT.algo

import java.util
import java.util.concurrent.{ExecutorService, Executors}

import com.tencent.angel.ml.FPGBDT.{FPGBDTLearner, FPGBDTModel}
import com.tencent.angel.ml.FPGBDT.algo.FPRegTree.FPRegTDataStore
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
                       trainDataStore: FPRegTDataStore, validDataStore: FPRegTDataStore) {
  val LOG = LogFactory.getLog(classOf[FPGBDTController])

  val forest: Array[RegTree] = new Array[RegTree](param.numTree)
  var currentTree: Int = 0
  val maxNodeNum: Int = Maths.pow(2, param.maxDepth) - 1
  var phase: Int = FPGBDTPhase.CREATE_SKETCH

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

  val nodePosStart: Array[Int] = new Array[Int](maxNodeNum)
  val nodePosEnd: Array[Int] = new Array[Int](maxNodeNum)
  val instancePos: Array[Int] = (0 until trainDataStore.numInstance).toArray
  val histograms: util.List[util.Map[Int, DenseFloatVector]] =
    new util.ArrayList[util.Map[Int, DenseFloatVector]](param.numTree)

  val threadPool: ExecutorService = Executors.newFixedThreadPool(param.numThread)

  // find splits for each feature, try to do this during dataset transpose process
  def createSketch(): Unit = {
    trainDataStore.createSketch(param.numSplit, 1)
    this.phase = FPGBDTPhase.NEW_TREE
  }

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
    // 4. reset active tree nodes, set all nodes to be inactive
    for (nid <- 0 until this.maxNodeNum) {
      resetActiveTNode(nid)
    }
    // 5. set root node to be active
    addReadyNode(0)
    // 6. reset instance position, set the root node's span
    this.nodePosStart(0) = 0
    this.nodePosEnd(0) = trainDataStore.numInstance
    for (nid <- 1 until this.maxNodeNum) {
      this.nodePosStart(nid) = -1
      this.nodePosEnd(nid) = -1
    }
    // 7. set phase
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
          // currently we use level-wise training, so we active it anyway
          addActiveNode(nid)
          hasActive = true
          activeList.add(nid)
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
  // TODO: multi-threading, batch building
  def runActiveNodes(): Unit = {
    LOG.info("------Run active node------")
    val start = System.currentTimeMillis()
    // 1. for each active node, cal grad and build hist
    for (nid <- 0 until this.maxNodeNum) {
      if (this.activeNode(nid) == 1) {
        // 1.1. cal grad for instances on current node
        calGradPairs(nid)
        // 1.2. set status to batch num
        // TODO: multi-batch
        this.activeNodeStat(nid) = 1
        this.histograms.get(this.currentTree)
          .put(nid, buildHistogram(nid))
      }
    }
    // 2. check if all nodes (threads) finished
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

  // calculate gradients of instances on specific node
  def calGradPairs(nid: Int): Unit = {
    LOG.info(s"------Calculate grad pairs of node[$nid]------")
    val nodeStart: Int = this.nodePosStart(nid)
    val nodeEnd: Int = this.nodePosEnd(nid)
    for (posIdx <- nodeStart until nodeEnd) {
      val insIdx: Int = instancePos(posIdx)
      val pred: Float = trainDataStore.getPred(insIdx)
      val label: Float = trainDataStore.getLabel(insIdx)
      val weight: Float = trainDataStore.getWeight(insIdx)
      val prob: Float = objective.transPred(pred)
      val gradPair: GradPair = new GradPair(objective.firOrderGrad(prob, label) * weight,
        objective.secOrderGrad(prob, label) * weight)
      gradPairs(insIdx) = gradPair
    }
  }

  // calculate gradients of all instances, used for level-wise training
  def calGradPairs(): Unit = {
    LOG.info(s"------Calculate grad pairs------")
    for (insIdx <- 0 until trainDataStore.numInstance) {
      val pred: Float = trainDataStore.getPred(insIdx)
      val label: Float = trainDataStore.getLabel(insIdx)
      val weight: Float = trainDataStore.getWeight(insIdx)
      val prob: Float = objective.transPred(pred)
      val gradPair: GradPair = new GradPair(objective.firOrderGrad(prob, label) * weight,
        objective.secOrderGrad(prob, label) * weight)
      gradPairs(insIdx) = gradPair
    }
  }

  // build histogram for a node
  def buildHistogram(nid: Int): DenseFloatVector = {
    LOG.info(s"------Build histogram of node[$nid]------")
    val buildStart = System.currentTimeMillis()
    // 1. allocate histogram
    val sampleFeats: Array[Int] = this.fset.get(currentTree)
    val numSampleFeats: Int = sampleFeats.length
    val numSplit: Int = this.param.numSplit
    val nodeStart: Int = nodePosStart(nid)
    val nodeEnd: Int = nodePosEnd(nid)
    val histogram: DenseFloatVector = new DenseFloatVector(numSampleFeats * numSplit * 2)
    // 2. sum up gradients
    var gradSum: Float = 0.0f
    var hessSum: Float = 0.0f
    for (posIdx <- nodeStart until nodeEnd) {
      val insIdx: Int = instancePos(posIdx)
      val gradPair: GradPair = gradPairs(insIdx)
      gradSum += gradPair.getGrad
      hessSum += gradPair.getHess
    }
    // 3. build histograms
    for (i <- 0 until numSampleFeats) {
      // 3.1. get info of current feature
      val fid: Int = sampleFeats(i)
      val indices: Array[Int] = trainDataStore.getFeatRow(fid).getIndices
      val bins: Array[Int] = trainDataStore.getfeatBins(fid)
      val nnz: Int = indices.length
      val gradOffset: Int = i * numSplit * 2
      val hessOffset: Int = gradOffset + numSplit
      var gradTaken: Float = 0
      var hessTaken: Float = 0
      // 3.2. loop non-zero instances, add to histogram, and record the gradients taken
      for (j <- 0 until nnz) {
        val insIdx: Int = indices(j)
        val binIdx: Int = bins(j)
        val gradIdx: Int = gradOffset + binIdx
        val hessIdx: Int = hessOffset + binIdx
        val gradPair: GradPair = gradPairs(insIdx)
        histogram.set(gradIdx, histogram.get(gradIdx) + gradPair.getGrad)
        histogram.set(hessIdx, histogram.get(hessIdx) + gradPair.getHess)
        gradTaken += gradPair.getGrad
        hessTaken += gradPair.getHess
      }
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
    FPGBDTLearner.clockAllMatrices(needFlushMatrixSet, model, true)
    // 3. leader worker pull all local best splits, find the global best split
    // TODO: find global best on server
    // do not clear node grad stats
    needFlushMatrixSet.remove(FPGBDTModel.NODE_GRAD_MAT)
    if (ctx.getTaskIndex == 0) {
      // 3.1. pull local best splits
      val workers: Array[Int] = (0 until param.numWorker).toArray
      // com.tencent.angel.psagent.matrix.transport.MatrixTransportClient: RequestDispatcher is failed?
      // NullPointerException at com.tencent.angel.ml.matrix.transport.GetRowsSplitRequest.getEstimizeDataSize(GetRowsSplitRequest.java:86)
      //val splitFidMat = splitFid.getRows(workers).asInstanceOf[util.ArrayList[SparseIntVector]]
      //LOG.info(s"Get split feature id matrix successfully, #row=${splitFidMat.size()}")
      //val splitFvalueMat = splitFvalue.getRows(workers).asInstanceOf[util.ArrayList[SparseFloatVector]]
      //LOG.info(s"Get split value matrix successfully, #row=${splitFvalueMat.size()}")
      //val splitGainMat = splitGain.getRows(workers).asInstanceOf[util.ArrayList[SparseFloatVector]]
      //LOG.info(s"Get split gain matrix successfully, #row=${splitGainMat.size()}")
      val splitFidMat: util.List[SparseIntVector] = new util.ArrayList[SparseIntVector](param.numWorker)
      val splitFvalueMat: util.List[SparseFloatVector] = new util.ArrayList[SparseFloatVector](param.numWorker)
      val splitGainMat: util.List[SparseFloatVector] = new util.ArrayList[SparseFloatVector](param.numWorker)
      for (wid <- 0 until param.numWorker) {
        splitFidMat.add(splitFid.getRow(wid).asInstanceOf[SparseIntVector])
        splitFvalueMat.add(splitFvalue.getRow(wid).asInstanceOf[SparseFloatVector])
        splitGainMat.add(splitGain.getRow(wid).asInstanceOf[SparseFloatVector])
      }
      // 3.2. clear local best split matrices for next update
      FPGBDTLearner.clearAllMatrices(needFlushMatrixSet, model)
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
    // 3.5. clock
    needFlushMatrixSet.add(FPGBDTModel.GLOBAL_FEAT_MAT)
    needFlushMatrixSet.add(FPGBDTModel.GLOBAL_VALUE_MAT)
    needFlushMatrixSet.add(FPGBDTModel.GLOBAL_GAIN_MAT)
    FPGBDTLearner.clockAllMatrices(needFlushMatrixSet, model, true)
    // 4. finish current phase
    this.phase = FPGBDTPhase.AFTER_SPLIT
    LOG.info(s"Find split cost ${System.currentTimeMillis() - start} ms")
  }

  // find local best split of one node
  def findLocalBestSplit(nid: Int): SplitEntry = {
    LOG.info(s"------To find the best split of node[$nid]------")
    val splitEntry: SplitEntry = new SplitEntry()
    // 1. calculate the gradStats of the node
    val hist: DenseFloatVector = histograms.get(this.currentTree).get(nid)
    if (hist == null) {
      LOG.error(s"null histogram on node[$nid]")
      return splitEntry
    }
    var gradSum: Float = 0.0f
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
    }
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
    LOG.info(s"Get split result from PS cost ${System.currentTimeMillis() - start} ms")
    // 2. reset instance position
    val resetPosStart = System.currentTimeMillis()
    // 2.1. get split result of responsible nodes
    // TODO: use BitSet
    //val splitResult: MyBitSet = new MyBitSet(trainDataStore.numInstance)
    var splitResult: Array[Int] = new Array[Int](trainDataStore.numInstance)
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
          val (leftChildGradStats, rightChildGradStats) =
            getSplitResult(nid, splitEntry, nodeStat, splitResult)
          updateNodeGradStats(2 * nid + 1, leftChildGradStats)
          updateNodeGradStats(2 * nid + 2, rightChildGradStats)
        }
      }
    }
    // 2.2. clear previous split result
    val resultModel = model.getPSModel(FPGBDTModel.SPLIT_RESULT_MAT)
    resultModel.zero()
    resultModel.clock().get
    // 2.3. push local split result & children grad stats
    val splitResultVec = new DenseIntVector(trainDataStore.numInstance, splitResult)
    resultModel.increment(0, splitResultVec)
    resultModel.clock().get
    nodeGradStats.clock().get
    // 2.4. pull global split result & children grad stats
    splitResult = resultModel.getRow(0).asInstanceOf[DenseIntVector].getValues
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
          resetInstancePos(nid, splitResult)
          // 3.7. set children as ready
          addReadyNode(2 * nid + 1)
          addReadyNode(2 * nid + 2)
        }
        // 3.8. deactivate active node
        resetActiveTNode(nid)
      }
    }
    LOG.info(s"After split cost: ${System.currentTimeMillis() - start} ms")
  }

  // get split result of responsible nodes
  def getSplitResult(nid: Int, splitEntry: SplitEntry,
                     nodeStat: GradStats, splitResult: Array[Int]): (GradStats, GradStats) = {
    val splitFid: Int = splitEntry.getFid
    val splitFvalue: Float = splitEntry.getFvalue
    LOG.info(s"------Get split result of node[$nid]: fid[$splitFid] fvalue[$splitFvalue]")
    val nodeStart: Int = this.nodePosStart(nid)
    val nodeEnd: Int = this.nodePosEnd(nid)
    LOG.info(s"Node[$nid] span: [$nodeStart-$nodeEnd]")
    // if no instance on this node, we do nothing
    var leftChildSumGrad: Float = 0.0f
    var rightChildSumGrad: Float = 0.0f
    var leftChildSumHess: Float = 0.0f
    var rightChildSumHess: Float = 0.0f
    if (nodeStart <= nodeEnd) {
      // find out instances that should be in right child
      // TODO: use bins rather than values
      val indices: Array[Int] = trainDataStore.getFeatRow(splitFid).getIndices
      val values: Array[Double] = trainDataStore.getFeatRow(splitFid).getValues
      val nnz: Int = indices.length
      // if split value >= 0, default left, otherwise default right
      if (splitFvalue >= 0.0f) {
        for (i <- 0 until nnz) {
          if (values(i) > splitFvalue) {
            //splitResult.set(indices(i))
            splitResult(indices(i)) = 1
            rightChildSumGrad += this.gradPairs(indices(i)).getGrad
            rightChildSumHess += this.gradPairs(indices(i)).getHess
          }
        }
        leftChildSumGrad = nodeStat.getSumGrad - rightChildSumGrad
        leftChildSumHess = nodeStat.getSumHess - rightChildSumHess
      }
      else {
        for (i <- nodeStart until nodeEnd) {
          //splitResult.set(this.instancePos(i))
          splitResult(this.instancePos(i)) = 1
        }
        for (i <- 0 until nnz) {
          if (values(i) < splitFvalue) {
            //splitResult.clear(indices(i))
            splitResult(indices(i)) = 0
            leftChildSumGrad += this.gradPairs(indices(i)).getGrad
            leftChildSumHess += this.gradPairs(indices(i)).getHess
          }
        }
        rightChildSumGrad = nodeStat.getSumGrad - leftChildSumGrad
        rightChildSumHess = nodeStat.getSumHess - leftChildSumHess
      }
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
  def resetInstancePos(nid: Int, splitResult: Array[Int]): Unit = {
    val nodeStart: Int = this.nodePosStart(nid)
    val nodeEnd: Int = this.nodePosEnd(nid)
    LOG.info(s"------Reset instance position of node[$nid] with span [$nodeStart-$nodeEnd]")
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
      var left: Int = nodeStart
      var right: Int = nodeEnd
      while (left < right) {
        // 1. left to right, find the first instance that should be in the right child
        var leftInsIdx: Int = this.instancePos(left)
        while (left < right && splitResult(leftInsIdx) == 0) {
          left += 1
          leftInsIdx = this.instancePos(left)
        }
        // 2. right to left, find the first instance that should be in the left child
        var rightInsIdx: Int = this.instancePos(right)
        while (left < right && splitResult(right) == 1) {
          right -= 1
          rightInsIdx = this.instancePos(right)
        }
        // 3. swap two instances
        if (left < right) {
          this.instancePos(left) = rightInsIdx
          this.instancePos(right) = leftInsIdx
        }
      }
      // 4. find the cut pos
      val curInsIdx: Int = this.instancePos(left)
      val cutPos = if (splitResult(curInsIdx) == 1) left else left + 1
      // 5. set the span of children
      this.nodePosStart(2 * nid + 1) = nodeStart
      this.nodePosStart(2 * nid + 2) = cutPos
      this.nodePosEnd(2 * nid + 1) = cutPos - 1
      this.nodePosEnd(2 * nid + 2) = nodeEnd
    }
    LOG.info(s"Left child[${2*nid+1} span: [${this.nodePosStart(2*nid+1)}-${this.nodePosEnd(2*nid+1)}]]")
    LOG.info(s"Right child[${2*nid+2} span: [${this.nodePosStart(2*nid+2)}-${this.nodePosEnd(2*nid+2)}]]")
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
    LOG.debug(s"Set node[$nid] as leaf node, leaf weight=$nodeWeight")
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
    val train_error: Tuple1[Float] = eval(trainDataStore, true)
    val valid: Tuple1[Float] = predict(validDataStore)
    globalMetrics.metric(MLConf.TRAIN_ERROR, train_error._1)
    globalMetrics.metric(MLConf.VALID_ERROR, valid._1)
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
      if (node.isLeaf) {
        val weight: Float = node.getLeafValue
        LOG.info(s"Leaf[$nid] weight: $weight")
        val nodeStart: Int = this.nodePosStart(nid)
        val nodeEnd: Int = this.nodePosEnd(nid)
        for (i <- nodeStart until nodeEnd) {
          val insIdx: Int = this.instancePos(i)
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
        if (node.isLeaf) {
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
  def eval(dataStore: FPRegTDataStore, isTrainSet: Boolean): Tuple1[Float] = {
    val descrip: String = if (isTrainSet) "training" else "validation"
    LOG.info("------Evaluation------")
    val start = System.currentTimeMillis()
    val evalMetric = new LogErrorMetric
    val error: Float = evalMetric.eval(dataStore.getPreds, dataStore.getLabels)
    LOG.info(s"Error on $descrip set after tree[${this.currentTree}]: $error")
    LOG.info(s"Evaluation cost ${System.currentTimeMillis() - start} ms")
    new Tuple1[Float](error)
  }

  // predict
  def predict(dataStore: FPRegTDataStore): Tuple1[Float] = {
    LOG.info("------Predict------")
    val start = System.currentTimeMillis()

    val splitFeat = model.getPSModel(FPGBDTModel.GLOBAL_FEAT_MAT)
    val splitValue = model.getPSModel(FPGBDTModel.GLOBAL_VALUE_MAT)
    val nodePreds = model.getPSModel(FPGBDTModel.NODE_PRED_MAT)

    val splitFeatVec = splitFeat.getRow(this.currentTree).asInstanceOf[TIntVector]
    val splitValueVec = splitValue.getRow(this.currentTree).asInstanceOf[TIntVector]
    val nodePredsVec = nodePreds.getRow(this.currentTree).asInstanceOf[TIntVector]
    LOG.info(s"Prediction of tree[${this.currentTree}]: ["
      + nodePredsVec.getValues.mkString(", ") + "]")

    val validInsPos: Array[Int] = new Array[Int](dataStore.numInstance)

    new Tuple1[Float](0.0f)
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

}
