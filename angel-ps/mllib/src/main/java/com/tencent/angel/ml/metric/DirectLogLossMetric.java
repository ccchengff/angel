package com.tencent.angel.ml.metric;

import com.tencent.angel.ml.utils.Maths;

public class DirectLogLossMetric extends LogLossMetric {
    /**
     * evaluate a specific metric for instances
     *
     * @param preds the probability predictions
     * @param labels the labels
     * @return the eval metric
     */
    @Override
    public float eval(float[] preds, float[] labels) {
        float errSum = 0.0f;
        for (int i = 0; i < preds.length; i++) {
            float predProbs = Maths.sigmoid(preds[i]);
            errSum += evalOne(predProbs, labels[i]);
        }
        return errSum / preds.length;
    }
}
