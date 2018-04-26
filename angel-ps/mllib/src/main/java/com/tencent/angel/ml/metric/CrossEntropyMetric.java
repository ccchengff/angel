package com.tencent.angel.ml.metric;

public class CrossEntropyMetric implements EvalMetric {
    /**
     * return name of metric
     *
     * @return the name
     */
    @Override
    public String getName() {
        return "cross-entropy loss";
    }

    /**
     * evaluate a specific metric for instances
     *
     * @param predProbs  the predictions
     * @param labels the labels
     * @return the eval metric
     */
    @Override
    public float eval(float[] predProbs, float[] labels) {
        int insNum = labels.length;
        int classNum = predProbs.length / insNum;
        float err = 0.0f;
        float[] temp = new float[classNum];
        for (int insIdx = 0; insIdx < insNum; insIdx++) {
            System.arraycopy(predProbs, insIdx * classNum, temp, 0, classNum);
            err += evalOne(temp, labels[insIdx]);
        }
        return err / labels.length;
    }

    /**
     * evaluate a specific metric for one instance
     *
     * @param predProbs the prediction
     * @param label the label
     * @return the eval metric
     */
    public float evalOne(float[] predProbs, float label) {
        float eps = 1e-16f;
        int y = (int) label;
        float p = Math.max(predProbs[y], eps);
        return (float) -Math.log(p);
    }

    /**
     * evaluate a specific metric for one instance
     *
     * @param predProbs  the prediction
     * @param label the label
     * @return the eval metric
     */
    @Override
    public float evalOne(float predProbs, float label) {
        return 0f;
    }
}
