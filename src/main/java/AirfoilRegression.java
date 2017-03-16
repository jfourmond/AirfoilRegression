import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class AirfoilRegression {
    private static Logger Log = LoggerFactory.getLogger(AirfoilRegression.class);

    private static final String FILENAME = "airfoil_self_noise.csv";

    private static UIServer uiServer;
    private static StatsStorage statsStorage;

    private static DataSet allData;
    private static DataSet trainingData;
    private static DataSet testData;

    private static DataNormalization normalizer;

    private static final int numClasses = 1;
    private static final int numInputs = 5;
    private static final int numHidden = 5;
    private static final int numOutputs = 1;
    private static final double learningRate = 0.1;
    private static final int iterations = 10000;
    private static final long seed = 13;

    private static MultiLayerNetwork net;

    private static void loadData() throws IOException, InterruptedException {
        int numLinesToSkip = 0;
        String delimiter = ";";

        int labelIndex = 5;
        int batchSize = 1503;

        Log.info("Loading dataset");

        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource(FILENAME).getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 5, 5, true);
        allData = iterator.next();
        allData.shuffle();

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.75);
        trainingData = testAndTrain.getTrain();
        testData = testAndTrain.getTest();
    }

    private static void normalizeData() {
        Log.info("Normalizing data");

        normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);
        normalizer.transform(testData);
    }

    private static void buildNetwork() {
        Log.info("Building model...");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(learningRate)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHidden)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHidden).nOut(numHidden)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHidden).nOut(numOutputs).build())
                .backprop(true).pretrain(false)
                .build();

        Log.info("Building neural network...");
        net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1), new StatsListener(statsStorage));
    }

    private static void train() {
        Log.info("Training neural network...");
        net.fit(trainingData);
    }

    private static void evaluate() {
        Log.info("Evaluate neural network...");

        // TODO EVALUATION
        RegressionEvaluation re = new RegressionEvaluation(5);
        INDArray output = net.output(testData.getFeatureMatrix());
        re.eval(testData.getLabels(), output);

        Log.info(re.stats());

//        Evaluation eval = new Evaluation(0);
//        INDArray output = net.output(testData.getFeatureMatrix());
//        eval.eval(testData.getLabels(), output);

//        Log.info(eval.stats());
    }

    public static void main(String args[]) throws IOException, InterruptedException {
        uiServer = UIServer.getInstance();
        statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        loadData();
        normalizeData();
        buildNetwork();
        train();
        evaluate();
    }
}
