package org.deeplearning4j.examples.feedforward;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.feedforward.classification.PlotUtil;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

/**
 * Created by Administrator on 2016/11/9.
 */
public class LogAnalysisMLP {
    private static Logger log = LoggerFactory.getLogger(LogAnalysisMLP.class);

    public static void main(String[] args) throws Exception {
        int seed = 1234;
        double learningRate = 0.00015;//0.1
        int batchSize = 1;
        int nEpochs = 1;

        int numInputs = 60;
        int numOutputs = 19;
        int numHiddenNodes = 41;

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("src/main/resources/logAnalysis/binary_dataset_training.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,60,numOutputs);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("src/main/resources/logAnalysis/binary_dataset_testing.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,60,numOutputs);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
            .iterations(1)
            .activation("sigmoid")
            .weightInit(WeightInit.XAVIER)
            .learningRate(learningRate)
            //.regularization(true).l2(1e-4).dropOut(0.00015)
            //.updater(Updater.NESTEROVS)
            .momentum(0.9)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).activation("relu")
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation("softmax")
                .nIn(numHiddenNodes).nOut(numOutputs).build())
            .backprop(true).pretrain(false)
            .build();



        NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();
        preProcessor.fit(trainIter);
        trainIter.setPreProcessor(preProcessor);

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new HistogramIterationListener(1));
        model.setListeners(new ScoreIterationListener(5));  //Print score every 10 parameter updates

        log.info("Train model....");
        for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIter );
        }

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        //model.output(testIter);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
//            preProcessor.transform(t);
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);
        }

        //Print the evaluation statistics
        System.out.println(eval.stats());

        System.out.println("****************Example finished********************");
    }
}
