from flask import Flask , render_template ,request
from preprocessing_pipeline import preprocessing_pipeline
import pickle
model =  pickle.load(open('model.pkl' , 'rb'))
vectorizer =  pickle.load(open('vectorizer.pkl' , 'rb'))

app = Flask(__name__)

@app.route('/result' , methods = ['POST' , 'GET'])
def check_result():
    Text = request.form(['text'])
    clean_text = preprocessing_pipeline(Text)
    token = vectorizer.transform(clean_text)
    pred = model.predict(token)
    if pred == 1:
        result = 'spam'
    else:
        result = 'not spam'
        
    return result        
        
    
    
    
    
    









if  __name__ =='__main__':
    app.run(debug=True)