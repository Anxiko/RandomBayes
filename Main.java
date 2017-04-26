
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.RandomBayes;
import weka.core.Attribute;
import weka.core.Instance;

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
            
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new Random(1));
            
            System.out.println(eval.toSummaryString());
            
            Enumeration<Attribute> atts = data.enumerateAttributes();
            
            while(atts.hasMoreElements()){
                System.out.println(atts.nextElement());
            }
            
            classifier.buildClassifier(data);
            
            System.out.println(classifier.randomCFS(data));
            
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
    
}
