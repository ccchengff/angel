package com.tencent.angel.ml.FPGBDT

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ml.FPGBDT.FPGBDTModel._
import com.tencent.angel.ml.FPGBDT.algo.QuantileSketch.{HeapQuantileSketch, SketchUtils}
import com.tencent.angel.ml.conf.MLConf
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.math.vector.{TIntDoubleVector, TIntVector}
import com.tencent.angel.ml.model.{MLModel, PSModel}
import com.tencent.angel.ml.predict.PredictResult
import com.tencent.angel.ml.utils.Maths
import com.tencent.angel.protobuf.generated.MLProtos.RowType
import com.tencent.angel.worker.storage.{DataBlock, MemoryDataBlock}
import com.tencent.angel.worker.task.TaskContext
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.conf.Configuration

/**
  * Created by ccchengff on 2017/11/16.
  */
object FPGBDTModel {
  val SYNC: String = "fgbdt.sync"
  val NNZ_NUM_MAT: String = "fpgbdt.nnz.num"
  val FEAT_ROW_MAT: String = "fpgbdt.feature.rows"
  val TOTAL_SAMPLE_NUM: String = "fpgbdt.total.sample.num"
  val FEAT_NNZ_MAT: String = "fpgbdt.feature.nnz"
  val LABEL_MAT: String = "fpgbdt.label"
  val FEAT_SAMPLE_MAT: String = "fpgbdt.feature.sample."
  val ACTIVE_NODE_MAT: String = "fpgbdt.active.nodes"
  val LOCAL_FEAT_MAT: String = "fpgbdt.local.best.split.feature"
  val LOCAL_VALUE_MAT: String = "fpgbdt.local.best.split.value"
  val LOCAL_GAIN_MAT: String = "fpgbdt.local.best.split.gain"
  val GLOBAL_FEAT_MAT: String = "fpgbdt.global.best.split.feature"
  val GLOBAL_VALUE_MAT: String = "fpgbdt.global.best.split.value"
  val GLOBAL_GAIN_MAT: String = "fpgbdt.global.best.split.gain"
  val NODE_GRAD_MAT: String = "fpgbdt.node.grad.stats"
  val NODE_PRED_MAT: String = "fpgbdt.node.predict"
  val SPLIT_RESULT_MAT: String = "fpgbdt.split.result"


  def apply(conf: Configuration) = {
    new FPGBDTModel(conf)
  }

  def apply(ctx:TaskContext, conf: Configuration) = {
    new FPGBDTModel(conf, ctx)
  }
}

class FPGBDTModel(conf: Configuration, _ctx: TaskContext = null) extends MLModel(conf, _ctx) {
  var LOG = LogFactory.getLog(classOf[FPGBDTModel])

  var featNum = conf.getInt(MLConf.ML_FEATURE_NUM, 10000)
  val maxTreeNum = conf.getInt(MLConf.ML_GBDT_TREE_NUM, MLConf.DEFAULT_ML_GBDT_TREE_NUM)
  val maxTreeDepth = conf.getInt(MLConf.ML_GBDT_TREE_DEPTH, MLConf.DEFAULT_ML_GBDT_TREE_DEPTH)
  val splitNum = conf.getInt(MLConf.ML_GBDT_SPLIT_NUM, MLConf.DEFAULT_ML_GBDT_SPLIT_NUM)
  //val featSampleRatio = conf.getFloat(MLConf.ML_GBDT_SAMPLE_RATIO, MLConf.DEFAULT_ML_GBDT_SAMPLE_RATIO)

  val maxTNodeNum: Int = Maths.pow(2, maxTreeDepth) - 1

  val psNumber = conf.getInt(AngelConf.ANGEL_PS_NUMBER, AngelConf.DEFAULT_ANGEL_PS_NUMBER)
  val wokergroupNumber = conf.getInt(AngelConf.ANGEL_WORKERGROUP_NUMBER, AngelConf.DEFAULT_ANGEL_WORKERGROUP_NUMBER)

  // adjust feature number to ensure the parameter partition
  //if (featNum % psNumber != 0) {
  //  featNum = (featNum / psNumber + 1) * psNumber
  //  conf.setInt(MLConf.ML_FEATURE_NUM, featNum)
  //  LOG.info(s"PS num: ${psNumber}, true feat num: ${featNum}")
  //}
  //val sampleFeatNum: Int = (featNum * featSampleRatio).toInt
  val trainDataNum = conf.getInt("angel.ml.train.data.num", 100)

  // Matrix xxx: synchronization
  val sync = PSModel(SYNC, 1, 1)
    .setRowType(RowType.T_INT_DENSE)
    .setOplogType("DENSE_INT")
    .setNeedSave(false)
  addPSModel(SYNC, sync)

  // Matrix 0-0: total sample number
  val totalSampleVec = PSModel(TOTAL_SAMPLE_NUM, 1, wokergroupNumber)
    .setRowType(RowType.T_INT_DENSE)
    .setOplogType("DENSE_INT")
    .setNeedSave(false)
  addPSModel(TOTAL_SAMPLE_NUM, totalSampleVec)

  // Matrix xxx: #nonzero for each feature of each worker
  val nonzeroNumVec = PSModel(NNZ_NUM_MAT, wokergroupNumber, featNum, 1, featNum)
    .setRowType(RowType.T_INT_DENSE)
    .setOplogType("DENSE_INT")
    .setNeedSave(false)
  addPSModel(NNZ_NUM_MAT, nonzeroNumVec)

  // Matrix xxx: feature rows
  var transposeBatchSize: Int = 1024
  if (transposeBatchSize == -1 || transposeBatchSize > featNum)
    transposeBatchSize = featNum
  val bufCapacity = 1 + SketchUtils.needBufferCapacity(HeapQuantileSketch.DEFAULT_K, trainDataNum.toLong)
  val featureRowBufSize = 1 + 2 * wokergroupNumber + Math.ceil(trainDataNum * 5 / 4).toInt
  val tmp = Math.max(bufCapacity, featureRowBufSize)
  val sketch = PSModel(FEAT_ROW_MAT, transposeBatchSize, tmp, transposeBatchSize / psNumber, tmp)
    .setRowType(RowType.T_FLOAT_DENSE)
    .setOplogType("DENSE_FLOAT")
    .setNeedSave(false)
  addPSModel(FEAT_ROW_MAT, sketch)

  // Matrix 0-1: feature to instance
  /*val featRow = PSModel(FEAT_NNZ_MAT, 1, trainDataNum, 1, trainDataNum / psNumber)
    .setRowType(RowType.T_DOUBLE_SPARSE)
    .setOplogType("SPARSE_DOUBLE")
    .setHogwild(true)
    .setNeedSave(false)
  addPSModel(FEAT_NNZ_MAT, featRow)*/

  // Matrix 0-2: labels
  val labels = PSModel(LABEL_MAT, 1, trainDataNum, 1, trainDataNum / psNumber)
    .setRowType(RowType.T_FLOAT_DENSE)
    .setOplogType("DENSE_FLOAT")
    .setHogwild(true)
    .setNeedSave(false)
  addPSModel(LABEL_MAT, labels)

  // Matrix 1: active tree nodes
  val activeTNodes = PSModel(ACTIVE_NODE_MAT, 1, maxTNodeNum, 1, maxTNodeNum / psNumber)
    .setRowType(RowType.T_INT_DENSE)
    .setOplogType("DENSE_INT")
    .setNeedSave(false)
  addPSModel(ACTIVE_NODE_MAT, activeTNodes)

  // Matrix 2: local best split feature id
  val localBestFeat = PSModel(LOCAL_FEAT_MAT, wokergroupNumber, maxTNodeNum)
    .setRowType(RowType.T_INT_SPARSE)
    .setOplogType("SPARSE_INT")
    .setNeedSave(false)
  addPSModel(LOCAL_FEAT_MAT, localBestFeat)

  // Matrix 3: local best split feature value
  val localBestValue = PSModel(LOCAL_VALUE_MAT, wokergroupNumber, maxTNodeNum)
    .setRowType(RowType.T_FLOAT_SPARSE)
    .setOplogType("SPARSE_FLOAT")
    .setNeedSave(false)
  addPSModel(LOCAL_VALUE_MAT, localBestValue)

  // Matrix 4: local best split feature gain
  val localBestGain = PSModel(LOCAL_GAIN_MAT, wokergroupNumber, maxTNodeNum)
    .setRowType(RowType.T_FLOAT_SPARSE)
    .setOplogType("SPARSE_FLOAT")
    .setNeedSave(false)
  addPSModel(LOCAL_GAIN_MAT, localBestGain)

  // Matrix 5: global best split feature
  val globalBestFeature = PSModel(GLOBAL_FEAT_MAT, maxTreeNum, maxTNodeNum, maxTreeNum, maxTNodeNum / psNumber)
    .setRowType(RowType.T_INT_DENSE)
    .setOplogType("DENSE_INT")
  addPSModel(GLOBAL_FEAT_MAT, globalBestFeature)

  // Matrix 6: global best split value
  val globalBestValue = PSModel(GLOBAL_VALUE_MAT, maxTreeNum, maxTNodeNum, maxTreeNum, maxTNodeNum / psNumber)
    .setRowType(RowType.T_FLOAT_DENSE)
    .setOplogType("DENSE_FLOAT")
  addPSModel(GLOBAL_VALUE_MAT, globalBestValue)

  // Matrix 7: global best split gain
  val globalBestGain = PSModel(GLOBAL_GAIN_MAT, maxTreeNum, maxTNodeNum, maxTreeNum, maxTNodeNum / psNumber)
    .setRowType(RowType.T_FLOAT_DENSE)
    .setOplogType("DENSE_FLOAT")
  addPSModel(GLOBAL_GAIN_MAT, globalBestGain)

  // Matrix 8: node's grad status
  val nodeGradStats = PSModel(NODE_GRAD_MAT, maxTreeNum, 2 * maxTNodeNum, maxTreeNum, 2 * maxTNodeNum / psNumber)
    .setRowType(RowType.T_FLOAT_DENSE)
    .setOplogType("DENSE_FLOAT")
    .setNeedSave(false)
  addPSModel(NODE_GRAD_MAT, nodeGradStats)

  // Matrix 9: node's predict value
  val nodePred = PSModel(NODE_PRED_MAT, maxTreeNum, maxTNodeNum, maxTreeNum, maxTNodeNum / psNumber)
    .setRowType(RowType.T_FLOAT_DENSE)
    .setOplogType("DENSE_FLOAT")
    .setAverage(true)
    .setNeedSave(false)
  addPSModel(NODE_PRED_MAT, nodePred)

  // Matrix 10: split result
  val colNum: Int = Math.ceil(trainDataNum / 32.0).toInt
  val splitResult = PSModel(SPLIT_RESULT_MAT, 1, colNum, 1, colNum)
    .setRowType(RowType.T_INT_DENSE)
    .setOplogType("DENSE_INT")
    .setNeedSave(false)
  addPSModel(SPLIT_RESULT_MAT, splitResult)

  super.setSavePath(conf)
  super.setLoadPath(conf)


  /**
    * Predict use the PSModels and predict data
    *
    * @param storage predict data
    * @return predict result
    */
  override def predict(storage: DataBlock[LabeledData]): DataBlock[PredictResult] = ???
}
