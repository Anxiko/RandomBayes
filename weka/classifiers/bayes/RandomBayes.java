package weka.classifiers.bayes;

import java.util.Collections;
import java.util.Enumeration;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedList;
import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

import weka.filters.unsupervised.instance.Resample;
import weka.filters.Filter;
import weka.core.Randomizable;
import weka.attributeSelection.CfsSubsetEval;

import java.util.Random;
import weka.core.DenseInstance;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Kindo
 */
public class RandomBayes extends AbstractClassifier implements Randomizable, OptionHandler{
    
    /* Config */
    
    /*Default options*/
    
    //Number of classifiers
    public static final int DEF_N_CLASSIFIERS = 10;
    
    //Percentage of instances used in each classifier
    public static final float DEF_PERC_INSTANCES = 1.0f;
    
    //Percentage of features used in each classifier
    public static final float DEF_PERC_FEAT = 0.5f;
    
    public static final boolean DEF_K_FLAG=false,DEF_D_FLAG=false,DEF_O_FLAG=false;
    
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
    
    //NaiveBayes parameters
    
    boolean k_flag=DEF_K_FLAG,d_flag=DEF_D_FLAG,o_flag=DEF_O_FLAG;
    
    /*Random*/
    
    //Seed used by the RNG
    int seed=DEF_SEED;
    
    //RNG to be used by the classifier
    Random rng;
    
    /*Bagging*/
    
    //Bag of classifiers
    NaiveBayes[] bag;
    
    //Filter used for each classifier
    Filter[] filters;
    
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
        filters = new Filter[n_classifiers];
        
        //Train all the NaiveBayes
        for (int i  = 0;i<n_classifiers;++i){
            bag[i] = new weka.classifiers.bayes.NaiveBayes();//Create the classifier (untrained)
            bag[i].setDisplayModelInOldFormat(o_flag);
            bag[i].setUseKernelEstimator(k_flag);
            bag[i].setUseSupervisedDiscretization(d_flag);
            
            //Create and configure the bootsrap filter, to get a random sample of the data
            Resample bootstrap = new Resample();
            bootstrap.setNoReplacement(false);//Using replacement
            bootstrap.setSampleSizePercent(perc_instances*100);//Set the percentage
            bootstrap.setRandomSeed(getSeed()+1);//Set the random seed
            bootstrap.setInputFormat(data);//Call this last! Respect calling convention: https://weka.wikispaces.com/Use+WEKA+in+your+Java+code#Filter-Calling%20conventions
            
            Instances sample = Filter.useFilter(data, bootstrap);//Use the filter to get a sample
            
            List<Integer> chosen_atts = randomCFS(sample);//Indices of attributes to be kept
            
            if (data.classIndex()>=0)//If the class index is known, keep it
                chosen_atts.add(data.classIndex());
            
            int[] array_atts = new int[chosen_atts.size()];
            for (int index = 0;index<array_atts.length; ++index)
                array_atts[index] += chosen_atts.get(index);
            
            //Filter to remove the features
            Remove rem = new Remove();
            rem.setInvertSelection(true);//Keep the columns in the indices
            rem.setAttributeIndicesArray(array_atts);//Columns to keep
            rem.setInputFormat(sample);//Call this last! Respect calling convention: https://weka.wikispaces.com/Use+WEKA+in+your+Java+code#Filter-Calling%20conventions
            filters[i] = rem;//Save the filter to reapply later
            sample = Filter.useFilter(sample, rem);//Remove the unselected features from the sample
            
            bag[i].buildClassifier(sample);//Train the classifier with the sample
        }
    }
    
    //Give the probability of an instance to belong to each possible class
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        
        double[] prob = null;//Contains the probability to belong to each class
        
        for (int i = 0; i<n_classifiers;++i){//Classify with each NaiveBayes in the bag
            
            //Make a copy of the instance, and reapply the filter
            Instance copy = new DenseInstance(instance);
            
            Filter filter = filters[i];
            
            if (!filter.isNewBatch())
                filter.batchFinished();
            
            filter.input(copy);
            filter.batchFinished();
            copy = filter.output();
            
            double[] new_prob = bag[i].distributionForInstance(copy);//Distribution for this classifier
            
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
        double currentScore = 0.0;//CFS score of the currently selected subset of attributes (0 at the start, because we start with none)
        
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
            double newScore = 0.0;//CFS score adding the new picked attribute
            
            for (RatedAttribute rated_att : ranking){
                random_att-=rated_att.getScore();//Decrease the random number by the probability
                if(random_att<=0){//This is the selected attribute
                    picked_att = rated_att.getAtt();
                    newScore = rated_att.getScore();
                    break;
                }
            }
            
            if (picked_att == null){//If none was picked, pick the last one
                picked_att = ranking.get(ranking.size()-1).getAtt();
                newScore = ranking.get(ranking.size()-1).getScore();
            }
            
            //If the attribute picked improves the score, add it to the set
            if (newScore > currentScore)
            {
                currentScore = newScore;
                picked_atts.set(picked_att);//Set the picked attribute
                ++n_picked_atts;
            }
            else
            {
                break;
            }
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
    
    /*OptionHandler*/
    
    public void set_instances_perc(float new_perc){
        if (bag==null)
            perc_instances = new_perc/100.0f;
    }
    
    public float get_instances_perc(){
        return 100.0f*perc_instances;
    }
    
    public void set_feat_perc(float new_perc){
        if (bag==null)
            perc_feat = new_perc/100.0f;
    }
    
    public float get_feat_perc(){
        return 100.0f*perc_feat;
    }
    
    public void set_n_classifiers(int new_n){
        if (bag==null)
            n_classifiers=new_n;
    }
    
    public int get_n_classifiers(){
        return n_classifiers;
    }
    
    public void set_k_flag(boolean new_flag){
        if (bag==null){
            k_flag=new_flag;
            if (k_flag)
                set_d_flag(false);
        }
    }
    
    public boolean get_k_flag(){
        return k_flag;
    }
    
    public void set_d_flag(boolean new_flag){
        if (bag==null){
            d_flag=new_flag;
            if (d_flag)
                set_k_flag(false);
        }
    }
    
    public boolean get_d_flag(){
        return d_flag;
    }
    
    public void set_o_flag(boolean new_flag){
        if (bag==null){
            o_flag=new_flag;
        }
    }
    
    public boolean get_o_flag(){
        return o_flag;
    }
    
    @Override
    public String[] getOptions() {
        List<String> result = new LinkedList<>();

        result.add("-P");//Percentage of samples
        result.add(""+(perc_instances*100));

        result.add("-F");//Percentage of features
        result.add(""+(perc_feat*100));

        result.add("-N");//Number of classifiers
        result.add(""+n_classifiers);
        
        if (k_flag)
            result.add("-K");
        
        if(d_flag)
            result.add("-D");
        
        if(o_flag)
            result.add("-O");

        return result.toArray(new String[result.size()]);
  }
    
    @Override
    public void setOptions(String[] options) throws Exception {
        set_instances_perc(Float.parseFloat(Utils.getOption('P', options)));
        set_feat_perc(Float.parseFloat(Utils.getOption('F', options)));
        set_n_classifiers(Integer.parseInt(Utils.getOption('N', options)));
        
        boolean k = Utils.getFlag('K', options);
        boolean d = Utils.getFlag('D', options);
        if (k && d) {
          throw new IllegalArgumentException("Can't use both kernel density "
            + "estimation and discretization!");
        }
        
        set_k_flag(k);
        set_d_flag(d);
        set_o_flag(Utils.getFlag('O', options));
        Utils.checkForRemainingOptions(options);
    }
    
    @Override
    public Enumeration<Option> listOptions(){
        List<Option> options = new LinkedList<>();
        
        options.add(new Option("Number of classifiers","N",1,"-N"));
        options.add(new Option("Percentage of instances to train each classifier with", "P",1,"-P"));
        options.add(new Option("Percentafe of features to train each classifier with", "F",1,"-F"));
        options.add(new Option("\tUse kernel density estimator rather than normal\n"+"\tdistribution for numeric attributes", "K", 0, "-K"));
        options.add(new Option("\tUse supervised discretization to process numeric attributes\n", "D",0, "-D"));
        options.add(new Option("\tDisplay model in old format (good when there are "+ "many classes)\n", "O", 0, "-O"));
        
        return Collections.enumeration(options);
    }
    
    /*Capabalities*/
    
    @Override
    public Capabilities getCapabilities(){
        
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable( Capability.MISSING_VALUES );

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // instances
    result.setMinimumNumberInstances(0);

    return result;
    }
}
