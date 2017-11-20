package com.tencent.angel.ml.FPGBDT

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ml.MLRunner
import com.tencent.angel.ml.conf.MLConf
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.conf.Configuration

/**
  * Created by ccchengff on 2017/11/16.
  */
class FPGBDTRunner extends MLRunner {
  val LOG = LogFactory.getLog(classOf[FPGBDTRunner])

  /**
    * Training job to obtain a model
    */
  override def train(conf: Configuration): Unit = {
    conf.setInt("angel.worker.matrixtransfer.request.timeout.ms", 60000)

    var featNum = conf.getInt(MLConf.ML_FEATURE_NUM, 10000)

    val psNumber = conf.getInt(AngelConf.ANGEL_PS_NUMBER, 1)
    if (featNum % psNumber != 0) {
      featNum = (featNum / psNumber + 1) * psNumber
      LOG.info(s"PS num: ${psNumber}, true feat num: ${featNum}")
    }
    conf.setInt(MLConf.ML_FEATURE_NUM, featNum)

    train(conf, FPGBDTModel(conf), classOf[FPGBDTTrainTask])
  }

  /**
    * Incremental training job to obtain a model based on a trained model
    */
  override def incTrain(conf: Configuration): Unit = {
    conf.setInt("angel.worker.matrix.transfer.request.timeout.ms", 60000)
    super.predict(conf, FPGBDTModel(conf), classOf[FPGBDTPredictTask])
  }

  /**
    * Using a model to predict with unobserved samples
    */
  override def predict(conf: Configuration): Unit = {
    conf.setInt("angel.worker.matrix.transfer.request.timeout.ms", 60000)
    train(conf)
  }
}
