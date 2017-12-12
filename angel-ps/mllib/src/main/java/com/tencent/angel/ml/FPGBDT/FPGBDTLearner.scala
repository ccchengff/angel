package com.tencent.angel.ml.FPGBDT

import java.nio.ByteBuffer
import java.util

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ml.FPGBDT.algo.{FPGBDTController, FPGBDTPhase}
import com.tencent.angel.ml.FPGBDT.algo.FPRegTreeDataStore.{TestDataStore, TrainDataStore}
import com.tencent.angel.ml.FPGBDT.algo.QuantileSketch.HeapQuantileSketch
import com.tencent.angel.ml.FPGBDT.psf._
import com.tencent.angel.ml.FPGBDT.psf.ClearUpdate.ClearUpdateParam
import com.tencent.angel.ml.FPGBDT.psf.QSketchGetFunc.QSketchGetParam
import com.tencent.angel.ml.FPGBDT.psf.QSketchMergeFunc.QSketchMergeParam
import com.tencent.angel.ml.FPGBDT.psf.QSketchesGetFunc.QSketchesGetParam
import com.tencent.angel.ml.FPGBDT.psf.QSketchesMergeFunc.QSketchesMergeParam
import com.tencent.angel.ml.MLLearner
import com.tencent.angel.ml.conf.MLConf
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.math.vector._
import com.tencent.angel.ml.matrix.MatrixContext
import com.tencent.angel.ml.matrix.psf.update.enhance.VoidResult
import com.tencent.angel.ml.metric.ErrorMetric
import com.tencent.angel.ml.model.MLModel
import com.tencent.angel.ml.param.FPGBDTParam
import com.tencent.angel.ml.utils.Maths
import com.tencent.angel.worker.storage.DataBlock
import com.tencent.angel.worker.task.TaskContext
import com.yahoo.sketches.quantiles.DoublesSketch
import org.apache.commons.logging.LogFactory
import com.tencent.angel.protobuf.generated.MLProtos.RowType


/**
  * Created by ccchengff on 2017/11/16.
  */
object FPGBDTLearner {
  val LOG = LogFactory.getLog(classOf[FPGBDTLearner])

  def sync(model: MLModel): Unit = {
    val syncModel = model.getPSModel(FPGBDTModel.SYNC)
    syncModel.clock().get
    syncModel.getRow(0)
    LOG.info("SYNCHRONIZATION")
  }

  def clearAllMatrices(needClearMatrices: util.Set[String], model: MLModel): Unit = {
    val iterator = needClearMatrices.iterator()
    while (iterator.hasNext) {
      val entry = iterator.next()
      val psModel = model.getPSModel(entry)
      val matrixId = psModel.getMatrixId()
      val clearUpdate = new ClearUpdate(new ClearUpdateParam(matrixId, false))
      psModel.update(clearUpdate).get
    }
  }

  def clockAllMatrices(needFlushMatrices: util.Set[String], model: MLModel, wait: Boolean): Unit = {
    val start = System.currentTimeMillis()

    val clockFutures = new util.ArrayList[util.concurrent.Future[VoidResult]]()
    val iter = model.getPSModels.entrySet().iterator()
    while (iter.hasNext) {
      val entry = iter.next()
      if (needFlushMatrices.contains(entry.getKey)) {
        clockFutures.add(entry.getValue.clock(true))
      }
      else {
        clockFutures.add(entry.getValue.clock(false))
      }
    }

    if (wait) {
      val size = clockFutures.size()
      for (i <- 0 until size) {
        clockFutures.get(i).get
      }
    }

    LOG.info(s"clock and flush matrices $needFlushMatrices cost "
      + s"${System.currentTimeMillis() - start} ms")
  }
}


class FPGBDTLearner(override val ctx: TaskContext) extends MLLearner(ctx) {
  val LOG = LogFactory.getLog(classOf[FPGBDTLearner])

  val param = new FPGBDTParam

  param.numFeature = conf.getInt(MLConf.ML_FEATURE_NUM, MLConf.DEFAULT_ML_FEATURE_NUM)
  param.numSplit = conf.getInt(MLConf.ML_GBDT_SPLIT_NUM, MLConf.DEFAULT_ML_GBDT_SPLIT_NUM)
  param.numTree = conf.getInt(MLConf.ML_GBDT_TREE_NUM, MLConf.DEFAULT_ML_GBDT_TREE_NUM)
  param.numThread = conf.getInt(MLConf.ML_GBDT_THREAD_NUM, MLConf.DEFAULT_ML_GBDT_THREAD_NUM)
  param.maxDepth = conf.getInt(MLConf.ML_GBDT_TREE_DEPTH, MLConf.DEFAULT_ML_GBDT_TREE_DEPTH)
  param.colSample = conf.getFloat(MLConf.ML_GBDT_SAMPLE_RATIO, MLConf.DEFAULT_ML_GBDT_SAMPLE_RATIO)
  param.learningRate = conf.getFloat(MLConf.ML_LEARN_RATE, MLConf.DEFAULT_ML_LEAR_RATE.asInstanceOf[Float])

  param.numWorker = conf.getInt(AngelConf.ANGEL_WORKERGROUP_NUMBER, AngelConf.DEFAULT_ANGEL_WORKERGROUP_NUMBER)
  param.featLo = ctx.getTaskIndex * (param.numFeature / param.numWorker)
  param.featHi = if (ctx.getTaskIndex + 1 == param.numWorker)
    param.numFeature else param.featLo + param.numFeature / param.numWorker
  LOG.info(s"Worker[${ctx.getTaskIndex}] responsible for feature range [${param.featLo}, ${param.featHi})")

  val model = new FPGBDTModel(conf, ctx)

  /*def initFeatureMetaInfo(dataStorage: DataBlock[LabeledData], model: FPGBDTModel): FPRegTDataStore = {
    val start = System.currentTimeMillis()

    val numFeature: Int = this.numFeature
    LOG.info(s"Create data meta, numFeature=$numFeature")

    // 1. clear vectors on server
    val matrices = new util.HashSet[String]()
    matrices.add(FPGBDTModel.TOTAL_SAMPLE_NUM)
    matrices.add(FPGBDTModel.FEAT_NNZ_MAT)
    matrices.add(FPGBDTModel.LABEL_MAT)
    if (ctx.getTaskIndex == 0) {
      FPGBDTLearner.clearAllMatrices(matrices, model)
    }
    FPGBDTLearner.clockAllMatrices(matrices, model, true)

    var tmpModel = model.getPSModel(FPGBDTModel.TOTAL_SAMPLE_NUM)
    val tmpVec = tmpModel.getRow(0).asInstanceOf[DenseIntVector]
    LOG.info(FPGBDTModel.TOTAL_SAMPLE_NUM + ": " + tmpVec.getValues.mkString(","))
    tmpModel = model.getPSModel(FPGBDTModel.FEAT_NNZ_MAT)
    for (i <- 0 until numWorker) {
      val tmpVec = tmpModel.getRow(i).asInstanceOf[SparseDoubleVector]
      LOG.info("size of " + FPGBDTModel.FEAT_NNZ_MAT + s" row[$i]=${tmpVec.getIndexToValueMap.size()}")
    }

    // 2. read local data partition
    val featIndices = new Array[util.ArrayList[Int]](numFeature)
    val featValues = new Array[util.ArrayList[Double]](numFeature)
    for (i <- 0 until numFeature) {
      featIndices(i) = new util.ArrayList[Int]()
      featValues(i) = new util.ArrayList[Double]()
    }
    val labelsList = new util.ArrayList[Float]()

    var numLocalSample: Int = 0
    dataStorage.resetReadIndex()
    var data: LabeledData = dataStorage.read()
    while (data != null) {
      val x: SparseDoubleSortedVector = data.getX.asInstanceOf[SparseDoubleSortedVector]
      var y: Float = data.getY.toFloat
      if (y != 1.0f)
        y = 0.0f
      val indices: Array[Int] = x.getIndices
      val values: Array[Double] = x.getValues
      val length: Int = indices.length
      for (i <- 0 until length) {
        val fid: Int = indices(i)
        val fvalue: Double = values(i)
        featIndices(fid).add(numLocalSample)
        featValues(fid).add(fvalue)
      }
      labelsList.add(y)
      numLocalSample += 1
      data = dataStorage.read()
    }
    LOG.info(s"Read data storage cost ${System.currentTimeMillis() - start} ms"
      + s", local sample number=$numLocalSample")

    // 2. sum up #sample
    val sumUpStart = System.currentTimeMillis()
    var sampleNumVec = new DenseIntVector(numWorker)
    sampleNumVec.set(ctx.getTaskIndex, numLocalSample)
    val sampleNumModel = model.getPSModel(FPGBDTModel.TOTAL_SAMPLE_NUM)
    sampleNumModel.increment(0, sampleNumVec)
    sampleNumModel.clock().get
    sampleNumVec = sampleNumModel.getRow(0).asInstanceOf[DenseIntVector]
    var numTotalSample: Int = 0
    var from: Int = 0
    for (i <- 0 until numWorker) {
      numTotalSample += sampleNumVec.get(i)
      if (i < ctx.getTaskIndex)
        from += sampleNumVec.get(i)
    }
    LOG.info(s"Sum up sample number cost ${System.currentTimeMillis() - sumUpStart} ms, "
      + s"total sample number=$numTotalSample, Task[${ctx.getTaskIndex}] from=$from")

    // 3. push features & create feature data storage
    val createStart = System.currentTimeMillis()
    // 3.1. create data storage
    val fpDataStore = new FPRegTDataStore(numFeature, numTotalSample)
    // 3.2. push each feature row and pull global feature row
    val featModel = model.getPSModel(FPGBDTModel.FEAT_NNZ_MAT)
    val featPerWorker = numFeature / numWorker
    matrices.clear()
    matrices.add(FPGBDTModel.FEAT_NNZ_MAT)
    for (round <- 0 to featPerWorker) {
      // 3.2.1. push #numWorker feature rows
      for (wid <- 0 until numWorker) {
        val fid = wid * featPerWorker + round
        if (fid < numFeature) {
          val nnz: Int = featIndices(fid).size()
          LOG.info(s"Task[${ctx.getTaskIndex}] feature[$fid] nnz=$nnz")
          val featRow = new SparseDoubleVector(numTotalSample, nnz)
          for (i <- 0 until nnz) {
            val k = featIndices(fid).get(i) + from
            val v = featValues(fid).get(i)
            featRow.set(k, v)
          }
          featModel.increment(wid, featRow)
        }
      }
      //FPGBDTLearner.clockAllMatrices(matrices, model, true)
      featModel.clock().get
      // 3.2.2. pull the feature row current worker responds to
      val fid = ctx.getTaskIndex * featPerWorker + round
      if (featLo <= fid && fid < featHi) {
        val featRow = featModel.getRow(ctx.getTaskIndex).asInstanceOf[SparseDoubleVector]
        fpDataStore.setFeatureRow(fid, featRow)
        val totalNnz = featRow.getIndexToValueMap.size()
        LOG.info(s"Task[${ctx.getTaskIndex}] feature[$fid] total nnz=$totalNnz")
      }
      // 3.2.3. clear matrix for next round
      if (ctx.getTaskIndex == 0) {
        FPGBDTLearner.clearAllMatrices(matrices, model)
      }
      //FPGBDTLearner.clockAllMatrices(matrices, model, true)
      featModel.clock().get
      for (wid <- 0 until numWorker) {
        val t = featModel.getRow(wid).asInstanceOf[SparseDoubleVector]
        val nnz = t.getIndexToValueMap.size()
        LOG.info(s"row[$wid] nnz after clear=$nnz")
      }
    }
    LOG.info(s"Set feature rows cost ${System.currentTimeMillis() - createStart} ms")
    // 3.3. set info for each instance
    var labelsVec = new DenseFloatVector(numTotalSample)
    for (i <- 0 until numLocalSample) {
      val k = i + from
      val v = labelsList.get(i)
      labelsVec.set(k, v)
    }
    val labelsModel = model.getPSModel(FPGBDTModel.LABEL_MAT)
    labelsModel.increment(0, labelsVec)
    labelsModel.clock().get
    labelsVec = labelsModel.getRow(0).asInstanceOf[DenseFloatVector]
    val labelsStore = new Array[Float](numTotalSample)
    for (i <- 0 until numTotalSample) {
      labelsStore(i) = labelsVec.get(i)
    }
    fpDataStore.setLabels(labelsStore)
    val preds = new Array[Float](numTotalSample)
    val weigths = new Array[Float](numTotalSample)
    util.Arrays.fill(preds, 0.0f)
    util.Arrays.fill(weigths, 1.0f)
    fpDataStore.setPreds(preds)
    fpDataStore.setWeights(weigths)

    LOG.info(s"Crete data meta info cost ${System.currentTimeMillis() - start} ms")
    fpDataStore
  }
  */

  def initFeatureMetaInfo(dataStorage: DataBlock[LabeledData], model: FPGBDTModel): TrainDataStore = {
    val start = System.currentTimeMillis()

    val numFeature: Int = param.numFeature
    val numWorker: Int = param.numWorker
    LOG.info(s"Create data meta, numFeature=$numFeature, numWorker=$numWorker")

    val featIndices = new Array[util.ArrayList[Int]](numFeature)
    val featValues = new Array[util.ArrayList[Double]](numFeature)
    for (i <- 0 until numFeature) {
      featIndices(i) = new util.ArrayList[Int]()
      featValues(i) = new util.ArrayList[Double]()
    }
    val labels = new util.ArrayList[Float]()

    // 1. read local data partition
    var numLocalSample: Int = 0
    dataStorage.resetReadIndex()
    var data: LabeledData = dataStorage.read()
    while (data != null) {
      val x: SparseDoubleSortedVector = data.getX.asInstanceOf[SparseDoubleSortedVector]
      var y: Float = data.getY.toFloat
      if (y != 1.0f)
        y = 0.0f
      val indices: Array[Int] = x.getIndices
      val values: Array[Double] = x.getValues
      val length: Int = indices.length
      for (i <- 0 until length) {
        val fid: Int = indices(i)
        val fvalue: Double = values(i)
        featIndices(fid).add(numLocalSample)
        featValues(fid).add(fvalue)
      }
      labels.add(y)
      numLocalSample += 1
      data = dataStorage.read()
    }
    LOG.info(s"Read data storage cost ${System.currentTimeMillis() - start} ms"
      + s", local sample number=$numLocalSample")
    // 2. sum up #sample
    val sumUpStart = System.currentTimeMillis()
    var sampleNumVec = new DenseIntVector(numWorker)
    sampleNumVec.set(ctx.getTaskIndex, numLocalSample)
    val sampleNumModel = model.getPSModel(FPGBDTModel.TOTAL_SAMPLE_NUM)
    sampleNumModel.increment(0, sampleNumVec)
    sampleNumModel.clock().get
    sampleNumVec = sampleNumModel.getRow(0).asInstanceOf[DenseIntVector]
    var numTotalSample: Int = 0
    for (i <- 0 until numWorker) {
      numTotalSample += sampleNumVec.get(i)
    }
    var offset: Int = 0
    for (i <- 0 until ctx.getTaskIndex) {
      offset += sampleNumVec.get(i)
    }
    LOG.info(s"Sum up sample number cost ${System.currentTimeMillis() - sumUpStart} ms, "
      + s"total sample number=$numTotalSample, Task[${ctx.getTaskIndex}] first instance index=$offset")
    // 3. push features & create feature data storage
    val createStart = System.currentTimeMillis()
    // 3.1. create data storage
    val fpDataStore = new TrainDataStore(param, numTotalSample)
    // 3.2. push each feature row and pull global feature row
    val featModel = model.getPSModel(FPGBDTModel.FEAT_NNZ_MAT)
    for (fid <- 0 until numFeature) {
      val nnz: Int = featIndices(fid).size()
      //val fStart = System.currentTimeMillis()
      //LOG.info(s"Task[${ctx.getTaskIndex}] feature[$fid] nnz=$nnz")
      /*val featIndicesArr: Array[Int] = new Array[Int](nnz)
      val featValuesArr: Array[Double] = new Array[Double](nnz)
      for (i <- 0 until nnz) {
        featIndicesArr(i) = featIndices(fid).get(i) + from
        featValuesArr(i) = featValues(fid).get(i)
      }
      var featRow = new SparseDoubleSortedVector(numTotalSample, featIndicesArr, featValuesArr)*/
      val featRow = new SparseDoubleVector(numTotalSample, nnz)
      for (i <- 0 until nnz) {
        val k = featIndices(fid).get(i) + offset
        val v = featValues(fid).get(i)
        featRow.set(k, v)
      }
      featModel.increment(fid, featRow)
      //featModel.clock().get
      if ((fid + 1) % 1000 == 0 || (fid + 1) == numFeature) {
        featModel.clock().get()
        LOG.info(s"Pushed ${fid + 1} feature rows")
      }
      /*if (featLo <= fid && fid < featHi) {
        val temp = featModel.getRow(fid).asInstanceOf[SparseDoubleVector]
        val indices = temp.getIndices
        val totalNnz = indices.length
        util.Arrays.sort(indices)
        val values = new Array[Double](totalNnz)
        for (i <- 0 until totalNnz) {
          values(i) = temp.get(indices(i))
        }
        val featRow = new SparseDoubleSortedVector(numTotalSample, indices, values)
        fpDataStore.setFeatureRow(fid, featRow)
        LOG.info(s"Task[${ctx.getTaskIndex}] feature[$fid] total nnz=$totalNnz")
      }
      LOG.info(s"Sum up feature[$fid] cost ${System.currentTimeMillis() - fStart} ms")*/
    }

    //val rows = featModel.getRows((param.featLo until param.featHi).toArray)
    for (fid <- param.featLo until param.featHi) {
      val temp = featModel.getRow(fid).asInstanceOf[SparseDoubleVector]
      //val temp = rows.get(fid - param.featLo).asInstanceOf[SparseDoubleVector]
      val indices = temp.getIndices
      val totalNnz = indices.length
      util.Arrays.sort(indices)
      val values = new Array[Double](totalNnz)
      for (i <- 0 until totalNnz) {
        values(i) = temp.get(indices(i))
      }
      val featRow = new SparseDoubleSortedVector(numTotalSample, indices, values)
      fpDataStore.setFeatureRow(fid, featRow)
      if ((fid - param.featLo + 1) % 1000 == 0) LOG.info(s"Pulled ${fid - param.featLo + 1} feature rows")
      // LOG.info(s"Task[${ctx.getTaskIndex}] feature[$fid] total nnz=$totalNnz")
    }
    LOG.info(s"Set feature rows cost ${System.currentTimeMillis() - createStart} ms")
    // 3.3. set info for each instance
    var labelsVec = new DenseFloatVector(numTotalSample)
    for (i <- 0 until numLocalSample) {
      val k = i + offset
      val v = labels.get(i)
      labelsVec.set(k, v)
    }
    val labelsModel = model.getPSModel(FPGBDTModel.LABEL_MAT)
    labelsModel.increment(0, labelsVec)
    labelsModel.clock().get
    labelsVec = labelsModel.getRow(0).asInstanceOf[DenseFloatVector]
    val labelsStore = new Array[Float](numTotalSample)
    for (i <- 0 until numTotalSample) {
      labelsStore(i) = labelsVec.get(i)
    }
    fpDataStore.setLabels(labelsStore)
    val preds = new Array[Float](numTotalSample)
    val weigths = new Array[Float](numTotalSample)
    for (i <- 0 until numTotalSample) {
      preds(i) = 0.0f
      weigths(i) = 1.0f
    }
    fpDataStore.setPreds(preds)
    fpDataStore.setWeights(weigths)

    LOG.info(s"Crete data meta info cost ${System.currentTimeMillis() - start} ms")
    fpDataStore
  }

  def initDataMetaInfo(dataStorage: DataBlock[LabeledData], model: FPGBDTModel): TestDataStore = {
    val start = System.currentTimeMillis()

    val numFeature: Int = param.numFeature
    LOG.info(s"Create data meta, numFeature=$numFeature")

    val instances: util.List[SparseDoubleSortedVector] = new util.ArrayList[SparseDoubleSortedVector]()
    val labelsList: util.List[Float] = new util.ArrayList[Float]()

    // 1. read local data partition
    var numInstances: Int = 0
    dataStorage.resetReadIndex()
    var data: LabeledData = dataStorage.read()
    while (data != null) {
      val x: SparseDoubleSortedVector = data.getX.asInstanceOf[SparseDoubleSortedVector]
      var y: Float = data.getY.toFloat
      if (y != 1.0f)
        y = 0.0f
      instances.add(x)
      labelsList.add(y)
      numInstances += 1
      data = dataStorage.read()
    }

    val dataStore: TestDataStore = new TestDataStore(param, numInstances)
    dataStore.setInstances(instances)
    val labels: Array[Float] = new Array[Float](numInstances)
    val preds: Array[Float] = new Array[Float](numInstances)
    val weights: Array[Float] = new Array[Float](numInstances)
    for (i <- 0 until numInstances) {
      labels(i) = labelsList.get(i)
      preds(i) = 0.0f
      weights(i) = 1.0f
    }
    dataStore.setLabels(labels)
    dataStore.setPreds(preds)
    dataStore.setWeights(weights)

    LOG.info(s"Read data storage cost ${System.currentTimeMillis() - start} ms"
      + s", sample number=$numInstances")
    dataStore
  }

  def transpose(dataStorage: DataBlock[LabeledData], model: FPGBDTModel): TrainDataStore = {
    val start = System.currentTimeMillis()

    val numFeature: Int = param.numFeature
    val numWorker: Int = param.numWorker
    val numSplit: Int = param.numSplit
    LOG.info(s"Transpose dataset, numFeature=$numFeature, numWorker=$numWorker")

    val featIndices = new Array[util.ArrayList[Int]](numFeature)
    val featValues = new Array[util.ArrayList[Float]](numFeature)
    for (i <- 0 until numFeature) {
      featIndices(i) = new util.ArrayList[Int]()
      featValues(i) = new util.ArrayList[Float]()
    }
    val labels = new util.ArrayList[Float]()

    // 1. read local data partition
    var numLocalSample: Int = 0
    dataStorage.resetReadIndex()
    var data: LabeledData = dataStorage.read()
    while (data != null) {
      val x: SparseDoubleSortedVector = data.getX.asInstanceOf[SparseDoubleSortedVector]
      var y: Float = data.getY.toFloat
      if (y != 1.0f)
        y = 0.0f
      val indices: Array[Int] = x.getIndices
      val values: Array[Double] = x.getValues
      val length: Int = indices.length
      for (i <- 0 until length) {
        val fid: Int = indices(i)
        val fvalue: Float = values(i).toFloat
        featIndices(fid).add(numLocalSample)
        featValues(fid).add(fvalue)
      }
      labels.add(y)
      numLocalSample += 1
      data = dataStorage.read()
    }
    LOG.info(s"Read data storage cost ${System.currentTimeMillis() - start} ms"
      + s", local sample number=$numLocalSample")

    // 2. sum up #sample and #nnz
    val sampleNumModel = model.getPSModel(FPGBDTModel.TOTAL_SAMPLE_NUM)
    val nnzNumModel = model.getPSModel(FPGBDTModel.NNZ_NUM_MAT)
    val needFlushMatrices = new util.HashSet[String]()
    // 2.1. push #sample
    val sumUpStart = System.currentTimeMillis()
    var sampleNumVec = new DenseIntVector(numWorker)
    sampleNumVec.set(ctx.getTaskIndex, numLocalSample)
    sampleNumModel.increment(0, sampleNumVec)
    needFlushMatrices.add(FPGBDTModel.TOTAL_SAMPLE_NUM)
    // 2.2. push #nnz for each feature
    var nnzNumVec = new DenseIntVector(numFeature)
    for (i <- 0 until numFeature) {
      nnzNumVec.set(i, featIndices(i).size())
    }
    nnzNumModel.increment(0, nnzNumVec)
    needFlushMatrices.add(FPGBDTModel.NNZ_NUM_MAT)
    FPGBDTLearner.clockAllMatrices(needFlushMatrices, model, true)
    // 2.3. pull #sample and starting index
    sampleNumVec = sampleNumModel.getRow(0).asInstanceOf[DenseIntVector]
    val numTotalSample = sampleNumVec.getValues.sum
    val offset = sampleNumVec.getValues.slice(0, ctx.getTaskIndex).sum
    // 2.4. pull #nnz for each feature
    nnzNumVec = nnzNumModel.getRow(0).asInstanceOf[DenseIntVector]
    LOG.info(s"Sum up sample number cost ${System.currentTimeMillis() - sumUpStart} ms, "
      + s"total sample number=$numTotalSample, Task[${ctx.getTaskIndex}] first instance index=$offset")

    // 3. push features & create feature data storage
    val createStart = System.currentTimeMillis()
    // 3.1. create data storage
    val fpDataStore = new TrainDataStore(param, numTotalSample)
    // 3.2. get candidate splits
    val sketchModel = model.getPSModel(FPGBDTModel.SKETCH_MAT)
    val matrixId = sketchModel.getMatrixId()
    needFlushMatrices.clear()
    needFlushMatrices.add(FPGBDTModel.SKETCH_MAT)

    var transposeBatchSize: Int = 1000
    if (transposeBatchSize == -1)
      transposeBatchSize = numFeature
    var fid: Int = 0
    var rowIndexes = (0 until transposeBatchSize).toArray
    while (fid < numFeature) {
      if (fid + transposeBatchSize > numFeature) {
        transposeBatchSize = numFeature - fid
        rowIndexes = (0 until transposeBatchSize).toArray
      }
      val start: Int = fid
      val stop: Int = fid + transposeBatchSize
      LOG.info(s"Range[$start-$stop)")
      val sketches = new Array[HeapQuantileSketch](transposeBatchSize)
      val estimateNs = new Array[Long](transposeBatchSize)
      while (fid < stop) {
        // 3.2.1. create local quantile sketch
        val nnz: Int = featValues(fid).size()
        val sketch: HeapQuantileSketch = new HeapQuantileSketch(nnz.toLong)
        for (i <- 0 until nnz) {
          sketch.update(featValues(fid).get(i))
        }
        sketches(fid - start) = sketch
        estimateNs(fid - start) = nnzNumVec.get(fid)
        fid += 1
      }
      // 3.2.2. push to PS and merge on PS
      sketchModel.update(new QSketchesMergeFunc(new QSketchesMergeParam(
        matrixId, true, rowIndexes, numWorker, numSplit, sketches, estimateNs)))
      FPGBDTLearner.sync(model)
      // 3.2.3. pull quantiles
      var getRowIndexes: Array[Int] = rowIndexes.clone()
      while (getRowIndexes != null) {
        val getResult = sketchModel.get(new QSketchesGetFunc(
          matrixId, rowIndexes, numWorker, numSplit)).asInstanceOf[QSketchesGetResult]
        val retryList: util.List[Int] = new util.ArrayList[Int]()
        for (rowId <- rowIndexes) {
          val quantiles: Array[Float] = getResult.getQuantiles(rowId)
          if (quantiles == null) {
            LOG.info(s"Feature[${rowId + start} need to try again]")
            retryList.add(rowId)
          }
          else {
            LOG.info(s"Feature[${rowId + start}] quantiles: [" + quantiles.mkString(", ") + "]")
          }
        }
        if (retryList.size() > 0) {
          getRowIndexes = new Array[Int](retryList.size())
          for (i <- 0 until retryList.size())
            getRowIndexes(i) = retryList.get(i)
        }
        else {
          getRowIndexes = null
        }
      }
      FPGBDTLearner.sync(model)
    }
    LOG.info(s"Get feature rows cost ${System.currentTimeMillis() - createStart} ms")
    if (1 == 1) return fpDataStore

    for (fid <- 0 until numFeature) {
      if (nnzNumVec.get(fid) > 0) {
        // 3.2.1. create local quantile sketch
        val nnz: Int = featIndices(fid).size()
        val sketch: HeapQuantileSketch = new HeapQuantileSketch(nnz.toLong)
        for (i <- 0 until nnz) {
          sketch.update(featValues(fid).get(i))
        }
        // 3.2.2. push and merge on PS
        sketchModel.update(new QSketchMergeFunc(new QSketchMergeParam(
          matrixId, true, fid, numWorker, numSplit, sketch, nnzNumVec.get(fid))))
        //FPGBDTLearner.clockAllMatrices(needFlushMatrices, model, false)
        //FPGBDTLearner.sync(model)
        //val quantiles = sketchModel.get(new PartialGetRowFunc(
        //  matrixId, fid, 0, numQuantiles)).asInstanceOf[PartialGetRowResult].getData
        //LOG.info("Feature[" + fid + "] quantiles: [" + quantiles.mkString(", ") + "]")
      }
    }
    FPGBDTLearner.clockAllMatrices(needFlushMatrices, model, true)

    for (fid <- 0 until numFeature) {
      if (nnzNumVec.get(fid) > 0) {
        var quantilesResult = sketchModel.get(new QSketchGetFunc(new QSketchGetParam(
          matrixId, fid, numWorker, numSplit))).asInstanceOf[QSketchGetResult]
        while (!quantilesResult.isSuccess) {
          quantilesResult = sketchModel.get(new QSketchGetFunc(new QSketchGetParam(
            matrixId, fid, numWorker, numSplit))).asInstanceOf[QSketchGetResult]
        }
        val quantiles = quantilesResult.getQuantiles
        LOG.info("Feature[" + fid + "] quantiles: [" + quantiles.mkString(", ") + "]")
      }
    }
    LOG.info(s"Get quantiles cost ${System.currentTimeMillis() - createStart} ms")

    /*val fracs = new Array[Double](param.numQuantiles)
    for (i <- 0 until param.numQuantiles) {
      fracs(i) = i.toDouble / param.numQuantiles.toDouble
    }
    for (fid <- 0 until numFeature) {
      if (ctx.getTaskIndex == 0) {
        // 3.2.1. create local quantile sketch
        val nnz: Int = featIndices(fid).size()
        val sketch = DoublesSketch.builder().build()
        //val sketch = new HeapQuantileSketch(nnz.toLong)
        for (i <- 0 until nnz) {
          sketch.update(featValues(fid).get(i))
        }
        // 3.2.2. push and merge on PS
        // blablabla
        val quantiles = Maths.double2Float(sketch.getQuantiles(fracs))
        val candidatesVec = new DenseFloatVector(param.numQuantiles, quantiles)
        candidatesModel.increment(fid, candidatesVec)
      }
      // 3.2.3. pull candidate splits from PS sketch model
      FPGBDTLearner.clockAllMatrices(needFlushMatrices, model, true)
      val quantiles = candidatesModel.getRow(fid).asInstanceOf[DenseFloatVector]
      if (fid >= param.featLo && fid < param.featHi) {
        fpDataStore.setSplits(fid, quantiles.getValues)
      }
      LOG.info(s"Candidate splits of feature[$fid]: [" + quantiles.getValues.mkString(", ") + "]")
      // 3.2.4. push candidate splits to PS candidate model
      // blablabla
    }
    // 3.2. push each feature row and pull global feature row
    // blablabla
    */
    // 3.3. set info for each instance
    var labelsVec = new DenseFloatVector(numTotalSample)
    for (i <- 0 until numLocalSample) {
      val k = i + offset
      val v = labels.get(i)
      labelsVec.set(k, v)
    }
    val labelsModel = model.getPSModel(FPGBDTModel.LABEL_MAT)
    labelsModel.increment(0, labelsVec)
    labelsModel.clock().get
    labelsVec = labelsModel.getRow(0).asInstanceOf[DenseFloatVector]
    val labelsStore = new Array[Float](numTotalSample)
    for (i <- 0 until numTotalSample) {
      labelsStore(i) = labelsVec.get(i)
    }
    fpDataStore.setLabels(labelsStore)
    val preds = new Array[Float](numTotalSample)
    val weights = new Array[Float](numTotalSample)
    for (i <- 0 until numTotalSample) {
      preds(i) = 0.0f
      weights(i) = 1.0f
    }
    fpDataStore.setPreds(preds)
    fpDataStore.setWeights(weights)

    LOG.info(s"Create data meta info cost ${System.currentTimeMillis() - start} ms")
    fpDataStore
  }

  /**
    * Train a ML Model
    *
    * @param train : input train data storage
    * @param vali  : validate data storage
    * @return : a learned model
    */
  override def train(train: DataBlock[LabeledData], vali: DataBlock[LabeledData]): MLModel = {
    transpose(train, model)
    if (1 == 1) return model

    val trainDataStore: TrainDataStore = initFeatureMetaInfo(train, model)
    train.clean()
    val validDataStore: TestDataStore = initDataMetaInfo(vali, model)
    vali.clean()

    val trainStart = System.currentTimeMillis()
    LOG.info("Start to train")
    val controller: FPGBDTController = new FPGBDTController(ctx,
      model, param, trainDataStore, validDataStore)
    var clock: Int = 0

    globalMetrics.addMetric(MLConf.TRAIN_ERROR, ErrorMetric(trainDataStore.numInstance))
    globalMetrics.addMetric(MLConf.VALID_ERROR, ErrorMetric(1))

    while (controller.phase != FPGBDTPhase.FINISHED) {
      if (controller.phase == FPGBDTPhase.CREATE_SKETCH) {
        LOG.info(s"******Current phase: CREATE_SKETCH, clock[$clock]******")
        controller.createSketch()
      }
      else if (controller.phase == FPGBDTPhase.NEW_TREE) {
        LOG.info(s"******Current phase: NEW_TREE, clock[$clock]******")
        controller.createNewTree()
      }
      else if (controller.phase == FPGBDTPhase.CHOOSE_ACTIVE) {
        LOG.info(s"******Current phase: CHOOSE_ACTIVE, clock[$clock]******")
        controller.chooseActive()
      }
      else if (controller.phase == FPGBDTPhase.RUN_ACTIVE) {
        LOG.info(s"******Current phase: RUN_ACTIVE, clock[$clock]******")
        controller.runActiveNodes()
      }
      else if (controller.phase == FPGBDTPhase.FIND_SPLIT) {
        LOG.info(s"******Current phase: FIND_SPLIT, clock[$clock]******")
        controller.findSplit()
      }
      else if (controller.phase == FPGBDTPhase.AFTER_SPLIT) {
        LOG.info(s"******Current phase: AFTER_SPLIT, clock[$clock]******")
        controller.afterSplit()
      }
      else if (controller.phase == FPGBDTPhase.FINISH_TREE) {
        LOG.info(s"******Current phase: FINISH_TREE, clock[$clock]******")
        controller.finishCurrentTree(globalMetrics)
      }
      clock += 1
    }

    LOG.info(s"Task[${ctx.getTaskIndex}] finishes training, " +
      s"train phase cost ${System.currentTimeMillis - trainStart} ms, " +
      s"total clock $clock")

    model
  }
}
