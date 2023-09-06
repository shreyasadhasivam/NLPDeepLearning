# NLPDeepLearning
Notes on NLP Preprocessing:<a href="https://creative-owl-18d.notion.site/Preprocessing-644150dd16fb40618f7db31e2716601f?pvs=4"> Preprocessing</a>

Once preprocessing is done, feature extraction is performed.
```python
freqs = build_freqs(tweets, labels) #build frequencies dictionary
X = np.zeros((m,3)) #initialise matrix X
for i in range(m): #for every tweet
  p_tweet = process_tweet(tweets[i]) #process tweet
  X[i,:] = extract_features(p_tweet,freqs) #extract features
```
Notes on 'Building and Visualizing word frequencies': <a href="https://creative-owl-18d.notion.site/Building-and-Visualizing-word-frequencies-49ca5fbd6ad1409c800aae25d402a1e9?pvs=4"> Building and Visualizing word frequencies </a>

<h5>Overview of Logistic Regression</h5>
<p>
  We have features X and labels Y, we use a function with some parameters to map the features to output labels. To get an optimum mapping we minimize the cost by comparing how closely output Yhat is to the true labels Y from our data. We repeat the process till the cost is minimized.

<br>
The sigmoid function approaches zero as the dot product of theta transpose X approaches minus infinity and one as it approaches infinity.
For classification, a threshold is needed and that is usually 0.5. So, if dot product is less than zero, the prediction is negative, and if greater the prediction is positive.
</p>
<h5>Logistic Regression:Training & Testing</h5>
Initialize parameter Theta, that we can use in sigmoid then compute the gradient that we will use to update theta, then calculate the cost. We continue doing so, until this is good enough.
To test a model, a subset of the data would be run, known as the validation set, on our model to get predictions.
TO compute accuracy:
We go over all the training samples(m of them), and then for every prediction, if it was right we add a one and then we divide by m.
