package com.tencent.angel.ml.FPGBDT

import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.task.PredictTask
import com.tencent.angel.worker.task.TaskContext
import org.apache.hadoop.io.{LongWritable, Text}

/**
  * Created by ccchengff on 2017/11/16.
  */
class FPGBDTPredictTask(ctx: TaskContext) extends PredictTask[LongWritable, Text](ctx) {
  override def predict(taskContext: TaskContext): Unit = {
    predict(ctx, FPGBDTModel(ctx, conf), taskDataBlock)
  }

  override def parse(key: LongWritable, value: Text): LabeledData = {
    dataParser.parse(value.toString)
  }
}
