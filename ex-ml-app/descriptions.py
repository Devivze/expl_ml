titles = {
    
'perm_importance' : 'Permutation Importance',
                     
'pdp' : 'Partial Dependence Plots',
       
'shap' : 'SHAP',

'shap_sum' : 'SHAP Summary plot',

'shap_single' : 'SHAP single prediction',
          
'whatif' : 'What if ... ?',

'whatif-shap' : 'SHAP',
    
'whatif-trust' : 'Trust RETRO Score',

'whatif-range' : 'Range PDP'
    
}


text = {
        
'perm_importance' : "Method provides a basic understanding of the impact of functions on the target. The essence of the method is to change the positions of each separated feature, thereby breaking the link between the feature and the true result. Next, the increase in the prediction error of the model is measured. It is assumed that if the error from the permutation is large, then the feature is important from the point of view of the model."

"On the graph, the objects are sorted in descending order of importance from top to bottom."

"Permutation importance also allows detecting data artefacts. For example, if the model identifies only one feature as important, it is highly likely that the model has been overfitted. \n",

'pdp' : "The partial dependency plot (PDP), shows the main trends and dependencies between features and target. It displays the average prediction for a certain feature value, provided that all other data points are entered into this feature value"
"This method reveals the type of dependence, i.e. whether the relationship between the goal and the object is linear, monotonous or more complex"
"There are two types of PDP:"

"Isolate - These graphs depict the range of values in the dataset, as well as how a specific value affects the result at the output of the model;"
"Interact - Such graphs show the combined effect of two features on the plot target the average target value for various combinations of object values.",

'shap' : "In SHAP, Shapley values are calculated to assess the importance of features." 
"To assess the importance of the feature, the predictions of the model with and without this feature are evaluated. The method uses extensions of the concepts of game theory to evaluate the operation of the ML algorithm. The game here is understood not as a confrontation between two or more sides, but as a team process in which each participant contributes to the overall result. Shapley value can be applied in machine learning if the players consider the presence of individual features, and the result of the game is the response of the model to a specific example. An important note is that this method does not consider the contribution of each feature to the accuracy of the model, but the contribution of each feature to the magnitude of the prediction of the model on a specific test example, which helps to interpret this prediction.",

'shap_sum' : 'Summary plot gives a general view of the impact of all functions taken together on the goal. There are all model features on the graph. SHAP values on the horizontal axis shows impact on the model. The color interpretate values of the feature in the data set.',

'shap_single' : "SHAP also allows seeing the impact for one selected prediction. The blue bars interpret the values of the functions that cause an increase in prediction, the red one shows an increase. The visual size reflects the magnitude of the function's impact. The difference in the total length of the bars is equal to the deviation of the predicted value from the base value.",

'whatif' : '<What If> is a tool that allows you to find out what will happen to the model if any specific feature values are input to it.',

'whatif-shap' : "The method is similar to SHAP for single prediction. Allows you to set specific values of features and evaluate their impact from the point of view of the machine learning model.",
    
'whatif-trust' : "Allows you to set specific feature values and evaluate how well the model is able to make accurate predictions for such values.",

'whatif-range' : 'Allows you to change the range of values for one of the features of the model, and then build a PDP interact for the other two features and evaluate the change in prediction behavior.'

}

source = {
 
    'perm_importance' : 'source :  Alexis Perrier "Feature Importance in Random Forests" (2015)',
                         
    'pdp' : 'source :  Goldstein A., Kapelner A., Bleich J., and Pitkin E. "Peeking Inside the Black Box: Visualizing Statistical Learning With Plots of Individual Conditional Expectation. (2015)',
           
    'shap' : 'source :  Scott Lundberg and Su-In Lee "A Unified Approach to Interpreting Model Predictions". (2017)',

    'shap_sum' : '',

    'shap_single' : '',
              
    'whatif' : '',

    'whatif-shap' : '',
        
    'whatif-trust' : '',

    'whatif-range' : ''
    
    }



