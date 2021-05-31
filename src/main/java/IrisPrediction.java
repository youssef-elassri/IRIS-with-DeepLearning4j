import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class IrisPrediction {
    public static void main(String[] args) throws Exception {
        System.out.println("-------------------Model Deployment-----------------------------");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("irisModel.zip"));

        INDArray inputData = Nd4j.create(new double[][]{
                {6.0, 1.5, 3.9, 2.8},{14.1, 13.5, 0.8, 0.2},
                {6.1, 1.1, 3.8, 2.1},{15.0, 17.5, 1.8, 0.2},
        });
        INDArray outputData = model.output(inputData);
        INDArray classes=outputData.argMax(1);
        System.out.println(outputData);
        System.out.println(classes);

    }
}
