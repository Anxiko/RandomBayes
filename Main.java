
import java.util.logging.Level;
import java.util.logging.Logger;

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
        
        String s = "CLASSIFIER weka.classifiers.trees.J48 -U FILTER weka.filters.unsupervised.instance.Randomize DATASET iris.arff";
        
        String[] weka_args= s.split(" ");
        
        try {
            //Call the demo
            WekaDemo.main(weka_args);
        } catch (Exception ex) {
        }
    }
    
}
