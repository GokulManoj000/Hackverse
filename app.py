from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your new ML model
model = pickle.load(open('final.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def details():
    if request.method == 'POST':
        # Get the input features from the form and convert to integers
        feature1 = int(request.form['school'])if request.form.get('school') == 'yes' else 0
        feature2 = int(request.form['study'])
        feature3 = int(request.form['failure'])
        feature4 = int(request.form['higher'])if request.form.get('higher') == 'yes' else 0
        feature5 = int(request.form['absence'])
        feature6 = int(request.form['g1'])
        feature7 = int(request.form['g2'])
        feature8 = int(request.form['g3'])
        feature9 = int(request.form['feducation'])
        feature10 = int(request.form['fdalc'])
        feature11 = int(request.form['fwalc'])
        feature12 = int(request.form['meducation'])
        feature13 = int(request.form['mdalc'])
        feature14 = int(request.form['mwalc'])

        # Convert the features to a NumPy array
        input_features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14]])

        # Make predictions using your ML model
        prediction = model.predict(input_features)

        # Pass the prediction to the result template
        return render_template('result.html', prediction=prediction[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
