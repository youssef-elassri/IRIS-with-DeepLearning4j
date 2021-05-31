import java.io.File;

import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class IrisApp {
	public static void main(String[] args) throws Exception {
		
		double learningRate = 0.001;
		int numInput = 4;
		int numHidden = 10;
		int numOut = 3;
		System.out.println("-------------------Model Creation-----------------------------");

		MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
				.seed(4451)
				.updater(new Adam(learningRate))
				.list()
					.layer(0, new DenseLayer.Builder()
							.nIn(numInput)
							.nOut(numHidden)
							.activation(Activation.SIGMOID)
							.build()
					)
					.layer(1, new OutputLayer.Builder()
							.nIn(numHidden)
							.nOut(numOut)
							.activation(Activation.SOFTMAX)
							.lossFunction(LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
							.build()
					)
				.build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(multiLayerConfiguration);
		model.init();
		
		//System.out.println("-------------------Monitor Creation-----------------------------");
		
		//System.out.println(multiLayerConfiguration.toJson());
		UIServer uiServer = UIServer.getInstance();
		InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
		uiServer.attach(inMemoryStatsStorage);
		
		model.setListeners(new StatsListener(inMemoryStatsStorage));
		
		System.out.println("-------------------Model Trainig-----------------------------");
		
		File fileTrain = new ClassPathResource("iris-train.csv").getFile();
		RecordReader recordReaderTrain = new CSVRecordReader();
		recordReaderTrain.initialize(new FileSplit(fileTrain));
		
		int batchSize = 1;
		int classIndex = 4;
		DataSetIterator dataSetIteratorTrain = 
				new RecordReaderDataSetIterator(recordReaderTrain, batchSize, classIndex, numOut);
		
		/*while(dataSetIteratorTrain.hasNext()) {
			DataSet dataSet = dataSetIteratorTrain.next();
			System.out.println(dataSet.getFeatures());
			System.out.println(dataSet.getLabels());
		}*/
		int nbEpochs = 50;
		
		for (int i=0; i<nbEpochs; i++) {
			System.out.println("\nEpoch: "+i+"\n");
			model.fit(dataSetIteratorTrain);	
		}
		System.out.println("Fitting is completed");
		
		System.out.println("-------------------Model Evaluation-----------------------------");
		File fileTest = new ClassPathResource("iris-test.csv").getFile();
		RecordReader recordReaderTest = new CSVRecordReader();
		recordReaderTest.initialize(new FileSplit(fileTest));
		DataSetIterator dataSetIteratorTest = 
				new RecordReaderDataSetIterator(recordReaderTest, batchSize, classIndex, numOut);
		
		Evaluation evaluation = new Evaluation();
		
		while(dataSetIteratorTest.hasNext()) {
			DataSet dataset = dataSetIteratorTest.next();
			INDArray features = dataset.getFeatures();
			INDArray labels = dataset.getLabels();
			INDArray predictions = model.output(features);
			evaluation.eval(labels, predictions);
		}
		System.out.println(evaluation.stats());
		ModelSerializer.writeModel(model, "irisModel.zip", true);


	}

}
