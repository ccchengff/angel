package com.tencent.angel.ml.treemodels.gbdt.dp

import com.tencent.angel.ml.treemodels.gbdt.GBDTModel
import com.tencent.angel.worker.task.TaskContext
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.hadoop.conf.Configuration

object DPGBDTModel {
  private val LOG: Log = LogFactory.getLog(classOf[DPGBDTModel])

  def apply(conf: Configuration, _ctx: TaskContext = null): DPGBDTModel = new DPGBDTModel(conf, _ctx)
}

class DPGBDTModel(conf: Configuration, _ctx: TaskContext = null) extends GBDTModel(conf, _ctx) {

}
