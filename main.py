
import streamlit as st
import pickle
import numpy as np
from sklearn import metrics

model=pickle.load(open("model3.pkl",'rb'))


def predict_forest(y,red,po):
    input=np.array([[y,red,po]]).astype(np.float64)
    prediction=model.fit_predict(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Forest Fire Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    y = st.text_input("y","Type Here")
    red = st.text_input("red","Type Here")
    po = st.text_input("po","Type Here")
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_forest(y,red,po)
        print(output)

if __name__=='__main__':
    main()
