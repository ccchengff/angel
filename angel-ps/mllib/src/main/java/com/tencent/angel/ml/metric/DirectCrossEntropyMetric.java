package com.tencent.angel.ml.metric;

public class DirectCrossEntropyMetric extends CrossEntropyMetric {
    /**
     * evaluate a specific metric for one instance
     *
     * @param preds the prediction
     * @param label the label
     * @return the eval metric
     */
    public float evalOne(float[] preds, float label) {
        float eps = 1e-16f;
        int y = (int) label;
        float t = 0.0f;
        for (float pred : preds) {
            t += Math.exp(pred);
        }
        t = Math.max((float) Math.log(t), eps);
        return -(preds[y] - t);
    }
}
