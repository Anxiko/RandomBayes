
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Attribute;
import weka.core.Instance;
import weka.classifiers.AbstractClassifier;
import weka.core.Randomizable;

import weka.classifiers.bayes.*;
import weka.classifiers.bayes.net.search.local.TAN;
import weka.classifiers.meta.Bagging;

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
            
            //Seed to run the experiments
            int SEED = 9943;
            
            String[] files ={"cylinder_bands.arff", "hypothyroid.arff",  "ionosphere.arff", "kr-vs-kp.arff", "optdigits.arff", "risk_factors_cervical_cancer.arff", "soybean.arff", "spambase.arff", "supermarket.arff", "unbalanced.arff"};
            
            for (String file : files)
                testWithFile(file,SEED);
            
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
    
    //Perform tests on a given .arrf file, with a seed
    static void testWithFile(String filename, int seed){
        try {
            
            System.out.println("File "+filename);
            
            //Configuration shared by the models
            final int PERC_INSTANCES = 100;//% of instances (with bootstrap)
            final int PERC_FEATURES = 50;//% of features
            final int NUM_CLASSIFIERS = 10;//Number of classifiers
            
            
            DataSource source = new DataSource(filename);
            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well
            Instances data = source.getDataSet();
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
            
            //NaiveBayes
            NaiveBayes nb = new NaiveBayes();
            testModel(data,seed,nb);
            
            //TAN
            BayesNet tan = new BayesNet();
            tan.setSearchAlgorithm(new TAN());
            testModel(data,seed,tan);
            
            //Bagging - NaiveBayes
            Bagging bagging_nb = new Bagging();
            bagging_nb.setClassifier(new NaiveBayes());
            bagging_nb.setBagSizePercent(PERC_INSTANCES);
            bagging_nb.setNumIterations(NUM_CLASSIFIERS);
            testModel(data,seed,bagging_nb);
            
            //Bagging - TAN
            Bagging bagging_tan = new Bagging();
            BayesNet base_bagging_tan = new BayesNet();
            base_bagging_tan.setSearchAlgorithm(new TAN());
            bagging_tan.setClassifier(bagging_nb);
            testModel(data,seed,bagging_tan);
            
            //RandomBayes
            RandomBayes rb = new RandomBayes();
            rb.set_feat_perc(PERC_FEATURES);
            rb.set_instances_perc(PERC_INSTANCES);
            rb.set_n_classifiers(NUM_CLASSIFIERS);
            testModel(data,seed,rb);
            
            //RandomTAN
            RandomTAN rt = new RandomTAN();
            rt.set_feat_perc(PERC_FEATURES);
            rt.set_instances_perc(PERC_INSTANCES);
            rt.set_n_classifiers(NUM_CLASSIFIERS);
            testModel(data,seed,rt);
            
            RandomBayes.reset_folds();
            RandomTAN.reset_folds();
            
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
    //Performs tests on a single model
    static void testModel(Instances data, int seed, AbstractClassifier c){
        try {
            
            System.out.println(c.getClass().toString()+ " " + ((c instanceof weka.classifiers.meta.Bagging)? ((weka.classifiers.meta.Bagging)c).getClassifier().getClass() : ""));
            
            final int CV_FOLDS = 10;
            
            //Train the model on a copy of the data
            Instances dataCopy = new Instances(data);
            
            //Set the seed, if possible
            if (c instanceof Randomizable){
                ((Randomizable)c).setSeed(seed);
            }
            
            Evaluation eval = new Evaluation(dataCopy);
            eval.crossValidateModel(c, dataCopy, CV_FOLDS, new Random(seed));
            System.out.println(eval.toSummaryString());
            
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
