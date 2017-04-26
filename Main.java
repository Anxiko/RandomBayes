
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.RandomBayes;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Kindo
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try {
            //Name of the data file
            String filename = "iris.arff";
            DataSource source = new DataSource(filename);
            Instances data = source.getDataSet();
            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
            
            RandomBayes classifier = new RandomBayes();//Create the classifier (default config)
            
            classifier.buildClassifier(data);//Build it using the read dataset
            
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
    
}
