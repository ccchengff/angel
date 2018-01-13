package com.tencent.angel.ml.treemodels.gbdt

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.exception.AngelException
import com.tencent.angel.ml.MLLearner
import com.tencent.angel.ml.conf.MLConf
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.model.MLModel
import com.tencent.angel.ml.treemodels.gbdt.dp.DPGBDTController
import com.tencent.angel.ml.treemodels.gbdt.fp.{FPGBDTController, FPGBDTModel}
import com.tencent.angel.ml.treemodels.param.GBDTParam
import com.tencent.angel.ml.treemodels.storage.{DPDataStore, DataStore, FPDataStore}
import com.tencent.angel.ml.utils.Maths
import com.tencent.angel.worker.storage.DataBlock
import com.tencent.angel.worker.task.TaskContext
import org.apache.commons.logging.{Log, LogFactory}

object GBDTLearner {
  private val LOG: Log = LogFactory.getLog(classOf[GBDTLearner])
}

class GBDTLearner(override val ctx: TaskContext) extends MLLearner(ctx) {
  private val LOG: Log = GBDTLearner.LOG

  val param = new GBDTParam

  // tree params
  param.numFeature = conf.getInt(MLConf.ML_FEATURE_NUM, MLConf.DEFAULT_ML_FEATURE_NUM)
  param.numSplit = conf.getInt(MLConf.ML_GBDT_SPLIT_NUM, MLConf.DEFAULT_ML_GBDT_SPLIT_NUM)
  param.numWorker = conf.getInt(AngelConf.ANGEL_WORKERGROUP_NUMBER, AngelConf.DEFAULT_ANGEL_WORKERGROUP_NUMBER)
  param.maxDepth = conf.getInt(MLConf.ML_GBDT_TREE_DEPTH, MLConf.DEFAULT_ML_GBDT_TREE_DEPTH)
  param.maxNodeNum = conf.getInt(MLConf.ML_GBDT_MAX_NODE_NUM, Maths.pow(2, param.maxDepth) - 1)
  param.insSampleRatio = conf.getFloat(MLConf.ML_GBDT_ROW_SAMPLE_RATIO, MLConf.DEFAULT_ML_GBDT_ROW_SAMPLE_RATIO)
  param.featSampleRatio = conf.getFloat(MLConf.ML_GBDT_SAMPLE_RATIO, MLConf.DEFAULT_ML_GBDT_SAMPLE_RATIO)
  // GBDT params
  param.numTree = conf.getInt(MLConf.ML_GBDT_TREE_NUM, MLConf.DEFAULT_ML_GBDT_TREE_NUM)
  param.numClass = conf.getInt(MLConf.ML_GBDT_CLASS_NUM, MLConf.DEFAULT_ML_GBDT_CLASS_NUM)
  param.numThread = conf.getInt(MLConf.ML_GBDT_THREAD_NUM, MLConf.DEFAULT_ML_GBDT_THREAD_NUM)
  param.batchNum = conf.getInt(MLConf.ML_GBDT_BATCH_NUM, MLConf.DEFAULT_ML_GBDT_BATCH_NUM)
  param.learningRate = conf.getFloat(MLConf.ML_LEARN_RATE, MLConf.DEFAULT_ML_LEAR_RATE.asInstanceOf[Float])
  param.minSplitGain = 0
  param.minChildWeight = conf.getFloat(MLConf.ML_GBDT_MIN_CHILD_WEIGHT, MLConf.DEFAULT_ML_GBDT_MIN_CHILD_WEIGHT.asInstanceOf[Float])
  param.regAlpha = conf.getFloat(MLConf.ML_GBDT_REG_ALPHA, MLConf.DEFAULT_ML_GBDT_REG_ALPHA)
  param.regLambda = conf.getFloat(MLConf.ML_GBDT_REG_LAMBDA, MLConf.DEFAULT_ML_GBDT_REG_LAMBDA.asInstanceOf[Float])
  param.maxLeafWeight = 0
  // parallel mode
  private val parallelMode = conf.get("ml.gbdt.parallel.mode", "FeatureParallel")
  // GBDT model
  var model: GBDTModel = _
  parallelMode match {
    case "FeatureParallel" => model = new FPGBDTModel(conf, ctx)
    case "DataParallel" => throw new AngelException("DataParallel not implemented")
    case _ => throw new AngelException("No such parallel mode: " + parallelMode)
  }

  /**
    * Train a ML Model
    *
    * @param train : input train data storage
    * @param vali  : validate data storage
    * @return : a learned model
    */
  override def train(train: DataBlock[LabeledData], vali: DataBlock[LabeledData]): MLModel = {
    val initStart = System.currentTimeMillis()
    var controller: GBDTController[_, _] = _
    parallelMode match {
      case "FeatureParallel" => controller = dataParallelTrain(train, vali)
      case "DataParallel" => controller = featureParallelTrain(train, vali)
      case _ => throw new AngelException("No such parallel mode: " + parallelMode)
    }
    LOG.info(s"Initialize data info and controller " +
      s"cost ${System.currentTimeMillis() - initStart} ms")

    val trainStart = System.currentTimeMillis
    LOG.info("Start to train")
    while (controller.phase != GBDTPhase.FINISHED) {
      LOG.info(s"******Current phase: ${controller.phase}******")
      controller.phase match {
        case GBDTPhase.NEW_TREE => controller.createNewTree()
        case GBDTPhase.CHOOSE_ACTIVE => controller.chooseActive()
        case GBDTPhase.RUN_ACTIVE => controller.runActiveNodes()
        case GBDTPhase.FIND_SPLIT => controller.findSplit()
        case GBDTPhase.AFTER_SPLIT => controller.afterSplit()
        case GBDTPhase.FINISH_TREE => controller.finishCurrentTree(globalMetrics)
        case GBDTPhase.FINISHED => throw new AngelException("GBDT train procedure does not terminate properly")
        case _ => throw new AngelException("Unrecognizable GBDT phase: " + controller.phase)
      }
    }
    LOG.info(s"Task[${ctx.getTaskIndex}] finishes training, " +
      s"train phase cost ${System.currentTimeMillis - trainStart} ms")

    model
  }

  def dataParallelTrain(train: DataBlock[LabeledData], vali: DataBlock[LabeledData]): GBDTController[_, _] = {
    LOG.info("Initialize data meta info")
    val trainDataStore = new DPDataStore(ctx, param)
    trainDataStore.init(train, model)
    train.clean()
    val validDataStore = new DPDataStore(ctx, param)
    validDataStore.setSplits(trainDataStore.getSplits)
    validDataStore.setZeroBins(trainDataStore.getZeroBins)
    validDataStore.init(vali, model)
    vali.clean()

    LOG.info("Initialize data-parallel GBDT controller")
    val controller = new DPGBDTController(ctx, param, model, trainDataStore, validDataStore)
    controller.init()
    controller
  }

  def featureParallelTrain(train: DataBlock[LabeledData], vali: DataBlock[LabeledData]): GBDTController[_, _] = {
    LOG.info("Initialize data meta info")
    val trainDataStore = new FPDataStore(ctx, param)
    trainDataStore.init(train, model)
    train.clean()
    val validDataStore = new DPDataStore(ctx, param)
    validDataStore.setSplits(trainDataStore.getSplits)
    validDataStore.setZeroBins(trainDataStore.getZeroBins)
    validDataStore.init(vali, model)
    vali.clean()

    LOG.info("Initialize feature-parallel GBDT controller")
    val controller = new FPGBDTController(ctx, param, model, trainDataStore, validDataStore)
    controller.init()
    controller
  }


}
