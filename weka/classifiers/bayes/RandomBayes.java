package weka.classifiers.bayes;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.core.Aggregateable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.estimators.DiscreteEstimator;
import weka.estimators.Estimator;
import weka.estimators.KernelEstimator;
import weka.estimators.NormalEstimator;

import weka.classifiers.bayes.NaiveBayes;
import weka.filters.unsupervised.instance.Resample;
import weka.filters.Filter;

/**
 *
 * @author Kindo
 */
public class RandomBayes extends AbstractClassifier{
    
    /* Config */
    
    /*Default options*/
    
    //Number of classifiers
    public static final int DEF_N_CLASSIFIERS = 10;
    
    //Percentage of instances used in each classifier
    public static final float DEF_PERC_INSTANCES = 0.6f;
    
    //Percentage of features used in each classifier
    public static final float DEF_PERC_FEAT = 0.6f;
    
    /* Data */
    
    /*Parameters*/
    
    //Number of classifiers
    int n_classifiers;
    
    //Percentage of instances
    float perc_instances;
    
    //Percentages of features
    float perc_feat;
    
    /*Bagging*/
    
    //Bag of classifiers
    NaiveBayes[] bag;
    
    
    /* Methods */
    
    /*Constructors*/
    
    public RandomBayes(){
        //Call parent constructor
        super();
        
        //Set default parameters
        n_classifiers = DEF_N_CLASSIFIERS;
        perc_instances = DEF_PERC_INSTANCES;
        perc_feat = DEF_PERC_FEAT;
        
        //Bag of classifiers
        bag = new NaiveBayes[n_classifiers];
    }
    
    /*Classifier*/
    
    //Train the classifier with the given instances
    @Override
    public void buildClassifier(Instances data) throws Exception{
        //Train all the NaiveBayes
        for (int i  = 0;i<n_classifiers;++i){
            bag[i] = new weka.classifiers.bayes.NaiveBayes();//Create the classifier (untrained)
            
            //Create and configure the bootsrap filter, to get a random sample of the data
            Resample bootstrap = new Resample();
            bootstrap.setInputFormat(data);//Configure the filter tp work on data
            bootstrap.setNoReplacement(false);//Using replacement
            bootstrap.setSampleSizePercent(perc_instances*100);//Set the percentage
            
            Instances sample = Filter.useFilter(data, bootstrap);//Use the filter to get a sample
            
            bag[i].buildClassifier(sample);//Train the classifier with the sample
        }
    }
    
}
