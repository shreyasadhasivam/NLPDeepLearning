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
