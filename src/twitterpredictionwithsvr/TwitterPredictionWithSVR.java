/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package twitterpredictionwithsvr;

import com.google.gson.Gson;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.List;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Instances;

/**
 *
 * @author irfannurhakim
 */
public class TwitterPredictionWithSVR {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        
        String[] arrayModelFileName = {"bebaslembab.arff", "cumannanya_pestakeluargaertiga.arff","jkw4p.arff","raisa6690-data1.arff","happinest.arff"};
        Integer idxModel = Integer.parseInt(args[0]);
        String fieldToForecast = args[1];
        String timeStampField = args[2];
        Double[] pred = new Double[6];
        Double[] actual = new Double[6];
        Gson gson = new Gson();
        
        try {
            // path to the Australian data data included with the time series forecasting
            // package
                        
            String absolutePathToData = weka.core.WekaPackageManager.PACKAGES_DIR.toString()
                    + File.separator + "timeseriesForecasting" + File.separator + "sample-data"
                    + File.separator + arrayModelFileName[idxModel];

            // load the data data
            Instances data = new Instances(new BufferedReader(new FileReader(absolutePathToData)));

            // new forecaster
            WekaForecaster forecaster = new WekaForecaster();

            // set the targets we want to forecast. This method calls
            // setFieldsToLag() on the lag maker object for us
            forecaster.setFieldsToForecast(fieldToForecast);

            // default underlying classifier is SMOreg (SVM) - we'll use
            // gaussian processes for regression instead
            forecaster.setBaseForecaster(new SMOreg());

            forecaster.getTSLagMaker().setTimeStampField(timeStampField); // date time stamp
            forecaster.getTSLagMaker().setMinLag(1);
            forecaster.getTSLagMaker().setMaxLag(12); //
            
            // build the model
            forecaster.buildForecaster(data);

            // prime the forecaster with enough recent historical data
            // to cover up to the maximum lag. In our case, we could just supply
            // the 12 most recent historical instances, as this covers our maximum
            // lag period
            forecaster.primeForecaster(data);

            // forecast for 12 units (months) beyond the end of the
            // training data
            List<List<NumericPrediction>> forecast = forecaster.forecast(6);

            // output the predictions. Outer list is over the steps; inner list is over
            // the targets
            for (int i = 0; i < 6; i++) {
                List<NumericPrediction> predsAtStep = forecast.get(i);
                NumericPrediction predForTarget = predsAtStep.get(0);
                //System.out.print("" + predForTarget.predicted() + " ");
                //System.out.println();
                pred[i] = predForTarget.predicted();
                
            }
            //System.out.println(gson.toJson(actual));
            System.out.print(gson.toJson(pred));

            // we can continue to use the trained forecaster for further forecasting
            // by priming with the most recent historical data (as it becomes available).
            // At some stage it becomes prudent to re-build the model using current
            // historical data.

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
