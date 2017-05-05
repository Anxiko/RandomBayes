package weka.classifiers.bayes;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;
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
import weka.core.Randomizable;
import weka.attributeSelection.CfsSubsetEval;

import java.util.Random;
import java.util.Set;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Kindo
 */
public class RandomBayes extends AbstractClassifier implements Randomizable{
    
    /* Config */
    
    /*Default options*/
    
    //Number of classifiers
    public static final int DEF_N_CLASSIFIERS = 10;
    
    //Percentage of instances used in each classifier
    public static final float DEF_PERC_INSTANCES = 0.6f;
    
    //Percentage of features used in each classifier
    public static final float DEF_PERC_FEAT = 0.6f;
    
    //Default seed
    public static final int DEF_SEED = 0;
    
    /* Data */
    
    /*Parameters*/
    
    //Number of classifiers
    int n_classifiers=DEF_N_CLASSIFIERS;
    
    //Percentage of instances
    float perc_instances=DEF_PERC_INSTANCES;
    
    //Percentages of features
    float perc_feat=DEF_PERC_FEAT;
    
    /*Random*/
    
    //Seed used by the RNG
    int seed=DEF_SEED;
    
    //RNG to be used by the classifier
    Random rng;
    
    /*Bagging*/
    
    //Bag of classifiers
    NaiveBayes[] bag;
    
    //Indices of columns used in each classifier
    int indices_used[][];
    
    //Instances used to train this classifier
    Instances instances_used;
    
    
    /* Methods */
    
    /*Constructors*/
    
    public RandomBayes(){
        //Call parent constructor
        super();
    }
    
    /*Classifier*/
    
    //Train the classifier with the given instances
    @Override
    public void buildClassifier(Instances data) throws Exception{
        //Build the RNG
        rng = new Random(getSeed());
        
		//Bag of classifiers
        bag = new NaiveBayes[n_classifiers];
        
        //Indices of attributes used in each classifier
        indices_used = new int [n_classifiers][];
        
        //Instances used to train this classifier
        instances_used = new Instances(data);
        instances_used.delete();//Keep the format, not the actual instances!
        
        //Train all the NaiveBayes
        for (int i  = 0;i<n_classifiers;++i){
            bag[i] = new weka.classifiers.bayes.NaiveBayes();//Create the classifier (untrained)
            
            //Create and configure the bootsrap filter, to get a random sample of the data
            Resample bootstrap = new Resample();
            bootstrap.setNoReplacement(false);//Using replacement
            bootstrap.setSampleSizePercent(perc_instances*100);//Set the percentage
            bootstrap.setRandomSeed(rng.nextInt());//Set the random seed
            bootstrap.setInputFormat(data);//Call this last! Respect calling convention: https://weka.wikispaces.com/Use+WEKA+in+your+Java+code#Filter-Calling%20conventions
            
            Instances sample = Filter.useFilter(data, bootstrap);//Use the filter to get a sample
            
            List<Integer> chosen_atts = randomCFS(sample);//Indices of attributes to be kept
            indices_used[i] = chosen_atts.stream().mapToInt(x->x).toArray();//save this to reapply later
            
            
            if (data.classIndex()>=0)//If the class index is known, keep it
                chosen_atts.add(data.classIndex());
            
            //Filter to remove the features
            Remove rem = new Remove();
            rem.setInvertSelection(true);//Keep the columns in the indices
            rem.setAttributeIndicesArray(chosen_atts.stream().mapToInt(x->x).toArray());//Columns to keep
            rem.setInputFormat(sample);//Call this last! Respect calling convention: https://weka.wikispaces.com/Use+WEKA+in+your+Java+code#Filter-Calling%20conventions
            sample = Filter.useFilter(sample, rem);//Remove the unselected features from the sample
            
            System.out.println(sample);
            
            bag[i].buildClassifier(sample);//Train the classifier with the sample
        }
    }
    
    //Give the probability of an instance to belong to each possible class
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        
        double[] prob = null;//Contains the probability to belong to each class
        
        for (int i = 0; i<n_classifiers;++i){//Classify with each NaiveBayes in the bag
            
            //Remove unused attributes
            
            List<Integer> lista = new ArrayList<>();
            
            for (int j = 0; j<indices_used[i].length;++j){
                lista.add(indices_used[i][j]);
            }
            lista.add(instances_used.classIndex());
            int[] indices_with_class = lista.stream().mapToInt(x->x).toArray();
            
            Remove rem = new Remove();
            rem.setInvertSelection(true);
            rem.setAttributeIndicesArray(indices_with_class);
            rem.setInputFormat(instances_used);
            rem.input(instance);
            rem.batchFinished();
            instance = rem.output();
            
            double[] new_prob = bag[i].distributionForInstance(instance);//Distribution for this classifier
            
            if (prob==null){//First result
                prob=new_prob;//Copy it directly
            }
            else{//Not the first result
                
                for (int cnt = 0; cnt<prob.length; ++cnt){//Add the probabilities together
                    prob[cnt]+=new_prob[cnt];
                }
            }
        }
        
        if (bag.length>=2){//Need to normalize?
            for (int cnt = 0; cnt<prob.length; ++cnt){
                prob[cnt]/=bag.length;//Normalize the probability, so the sum of all is 1
            }
        }
        
        return prob;
    }
    
    /*
        Perform feature selection, using CFS's score as the random probability for a feature to be picked.
        Process is constructive, starting with 0 features and adding one in each iteration until the percentage of features is reached.
        The CFS score is the probability of that feature to be added
    */
    
    
    private class RatedAttribute{
        private final double score;//CFS rating when adding this attribute
        private final int att;//Index of the attribute
        
        public RatedAttribute(double score, int att){
            this.score = score;
            this.att = att;
        }
        
        double getScore(){
            return score;
        }
        
        int getAtt(){
            return att;
        }
    }
    
    private List<Integer> randomCFS(Instances instances) throws Exception{
        
        CfsSubsetEval cfs = new CfsSubsetEval();//Create the cfs
        cfs.buildEvaluator(instances);
        
        List<Attribute> all_atts = Collections.list(instances.enumerateAttributes());//List of all attributes
        BitSet picked_atts = new BitSet(all_atts.size());//Attributes ready to be picked
        picked_atts.clear();//Set them all to false, none is picked at the start
        final int goal = (int) Math.ceil(all_atts.size()*this.perc_feat);//Number of features to reach
        int n_picked_atts = 0;//Number of picked atts
        
        while(n_picked_atts<goal){//Add features until the goal is reached
            double totalScore = 0.0;//Total CFS score in this iteration
            List<RatedAttribute> ranking = new ArrayList<>();//Stores the CFS score when adding an attribute
            for(int i = 0; i < all_atts.size(); ++i){//Iterate over the possible attributes
                if (picked_atts.get(i))//Skip the element if it's already there
                    continue;
                
                picked_atts.set(i);//Add this attribute
                double score = cfs.evaluateSubset(picked_atts);//Eval the new subset
                totalScore+=score;//Add it to the total
                ranking.add(new RatedAttribute(score,i));
                picked_atts.clear(i);//Turn it off again
            }
            
            double random_att = rng.nextDouble()*totalScore;//Attribute will be picked when the accumulative probability reaches or exceeds this value
            Integer picked_att = null;//Attribute to be picked
            
            for (RatedAttribute rated_att : ranking){
                random_att-=rated_att.getScore();//Decrease the random number by the probability
                if(random_att<=0){//This is the selected attribute
                    picked_att = rated_att.getAtt();
                    break;
                }
            }
            
            if (picked_att == null){//If none was picked, pick the last one
                picked_att = ranking.size()-1;
            }
            
            picked_atts.set(picked_att);//Set the picked attribute
            ++n_picked_atts;
        }
        
        BitSet features_bit = picked_atts;//Bits of the features to keep
        List<Integer> indices = new ArrayList<>();//Get them to a list
        for (int feat_index = features_bit.nextSetBit(0); feat_index >= 0; feat_index = features_bit.nextSetBit(feat_index + 1)) {
            indices.add(feat_index);
        }
        
        return indices;
    }
    
    /*Randomizable*/
    
    //Set the random seed used by the RNG (has to be called before buildClassifier
    @Override
    public void setSeed(int seed) {
        this.seed=seed;
    }
    
    //Get the random seed used by the RNG
    @Override
    public int getSeed() {
        return this.seed;
    }
    
}
