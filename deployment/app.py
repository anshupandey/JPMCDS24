from flask import Flask, request
import json
import mlflow
import numpy as np
import pandas as pd

mlflow.set_tracking_uri("http://3.25.126.82:5000/")

model_name="insurance"
model_version=1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")


app = Flask(__name__)

@app.route("/predict",methods=['GET','POST'])
def main():
    data = request.data
    print(data)
    data = data.decode()
    data = json.loads(data)
    outdata = data.copy()
    for key in ['age','bmi','children','smoker']:
        if key in data:
            pass
        else:
            return f"missing key {key}"
    data['age'] = np.array(data['age'],dtype='int64')
    data['bmi'] = np.array(data['bmi'],dtype='float64')
    data['children'] = np.array(data['children'],dtype='int64')
    data['smoker'] = np.array(data['smoker'],dtype='int32')
    data2 = pd.DataFrame(data)
    print(data2.info())
    print(data2)
    pred = model.predict(data2)
    outdata['prediction'] = pred[0]
    return json.dumps(outdata)

if __name__=="__main__":
    app.run(port=5001,debug=False,host="0.0.0.0")