package com.tencent.angel.ml.FPGBDT

import com.tencent.angel.ml.conf.MLConf
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.task.TrainTask
import com.tencent.angel.ml.utils.DataParser
import com.tencent.angel.worker.storage.MemoryDataBlock
import com.tencent.angel.worker.task.TaskContext
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.io.{LongWritable, Text}

/**
  * Created by ccchengff on 2017/11/16.
  */
class FPGBDTTrainTask(val ctx: TaskContext) extends TrainTask[LongWritable, Text](ctx) {
  private val LOG = LogFactory.getLog(classOf[FPGBDTTrainTask])

  private val feaNum = conf.getInt(MLConf.ML_FEATURE_NUM, MLConf.DEFAULT_ML_FEATURE_NUM)
  private val dataFormat = conf.get(MLConf.ML_DATA_FORMAT, "libsvm")
  private val dataParser = DataParser(dataFormat, feaNum, true)

  // validation data storage
  private val validRatio = conf.getDouble(MLConf.ML_VALIDATE_RATIO, 0.05)
  private val validDataStorage = new MemoryDataBlock[LabeledData](-1)

  override def train(taskContext: TaskContext): Unit = {
    val trainer = new FPGBDTLearner(ctx)
    trainer.train(taskDataBlock, validDataStorage)
  }

  override def parse(key: LongWritable, value: Text): LabeledData = {
    dataParser.parse(value.toString)
  }

  override def preProcess(taskContext: TaskContext): Unit = {
    var count: Int = 0
    val valid: Int = Math.ceil(1.0 / validRatio).asInstanceOf[Int]

    val reader = taskContext.getReader
    while (reader.nextKeyValue()) {
      val out = parse(reader.getCurrentKey, reader.getCurrentValue)
      if (out != null) {
        if (count % valid == 0)
          validDataStorage.put(out)
        else
          taskDataBlock.put(out)
        count += 1
      }
    }
    taskDataBlock.flush()
    validDataStorage.flush()
  }
}
