from flask import Flask , render_template ,request
from preprocessing_pipeline import preprocessing_pipeline
import pickle
model =  pickle.load(open('model.pkl' , 'rb'))
vectorizer =  pickle.load(open('vectorizer.pkl' , 'rb'))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('Input.html')



@app.route("/Result" , methods = ['POST'])
def check_result():
    Text = request.form['Text']
    clean_text = preprocessing_pipeline(Text)
    token = vectorizer.transform([clean_text])
    pred = model.predict(token)
    if pred == 1:
        output = 'Spam'
    else:
        output = 'Not Spam'
        
    return render_template('result.html', result = output)        
        
    
    
    
    
    









if  __name__ =='__main__':
    app.run(debug=True)