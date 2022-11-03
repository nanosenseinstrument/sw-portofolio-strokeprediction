import base64
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# import phik
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import shap
import streamlit as st
from streamlit_echarts import st_echarts
from streamlit_option_menu import option_menu
from streamlit_shap import st_shap

from bin.predictor import Predictor


@st.cache
def load_data():
    data = pd.read_csv("data/stroke_dataset.csv")
    str_only = data[data["stroke"] == 1]
    no_str_only = data[data["stroke"] == 0]
    no_str_only = no_str_only[(no_str_only["gender"] != 255)]
    return data, str_only, no_str_only


home = os.getcwd()
model = Predictor()
saved_model = joblib.load("model/model.sav")
explainer = shap.TreeExplainer(saved_model)
colors = ["#fe346e", "#512b58"]

colname = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]


# set settings for streamlit page
st.set_page_config(layout="wide", page_title="Stroke Prediction", page_icon="pill")
# st.set_page_config(page_title="Stroke Prediction", page_icon="pill")

# load database
data, str_only, no_str_only = load_data()

# hide streamlit menu bar
hide_streamlit_style = """
<style>
MainMenu {visibility: hidden;}
footer{visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes("header.png")
)
st.sidebar.markdown(
    header_html,
    unsafe_allow_html=True,
)

# Option menu from streamlit component streamlit_option_menu
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "Stroke Prediction", "Stroke Database"],
        icons=["house", "list-task", "kanban"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        # orientation="horizontal",
    )


# def predict_sidebar(inputs, shap_value=True):
#     with st.spinner(text="In Progress"):
#         prediction = model.predict(inputs)
#         proba = model.predict_proba(inputs)

#         if shap_value:
#             st.subheader(" ")
#             st.subheader("Prediction Explainer")
#             input_shap = np.asarray(inputs).reshape(1, -1)
#             input_shap = pd.DataFrame(input_shap, columns=colname)
#             shap_values = explainer.shap_values(input_shap)
#             st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :], input_shap))

#     # sidebar
#     st.sidebar.markdown(
#         "<h2 style='text-align: center; color: black;'>Stroke Prediction and Confidence</h2>",
#         unsafe_allow_html=True,
#     )

#     if prediction == 1:
#         html_temp = """
#             <div style="background-color:{};padding:10px;border-radius:10px 10px 10px 10px">
#             <h2 style="color:{};text-align:center;">High Stroke Risk</h2>
#             </div>
#             """

#         bg_results = ["#F1E1E6", "#7D363B"]

#         persen_proba = float(round((proba - 0.3) / 0.7, 2))

#         liquidfill_option = {
#             "series": [{"type": "liquidFill", "data": [persen_proba], "color": [colors[0]]}]
#         }

#     else:

#         html_temp = """
#             <div style="background-color:{};padding:10px;border-radius:10px 10px 10px 10px">
#             <h2 style="color:{};text-align:center;">Low Stroke Risk</h2>
#             </div>
#             """

#         bg_results = ["#D3E0EA", "#276678"]

#         persen_proba = float(round((0.3 - proba) / 0.3, 2))

#         liquidfill_option = {
#             "series": [{"type": "liquidFill", "data": [persen_proba], "color": [colors[1]]}]
#         }

#     st.sidebar.markdown(html_temp.format(bg_results[0], bg_results[1]), unsafe_allow_html=True)

#     with st.sidebar:
#         st_echarts(liquidfill_option)


if "inputs" not in st.session_state:
    st.session_state.inputs = [1, 0, 0, 0, 0, 2, 0, 50, 20.0, 2]


# selected = option_menu(
#     None,
#     ["Home", "Stroke Prediction", "Stroke Database"],
#     icons=["house", "list-task", "kanban"],
#     menu_icon="cast",
#     default_index=0,
#     orientation="horizontal",
# )

# Login page
if selected == "Home":

    # predict_sidebar(inputs=st.session_state.inputs, shap_value=False)

    if "user_name" not in st.session_state:
        st.session_state.user_name = "User Name"
    user_name = st.text_input("User Name", value=st.session_state.user_name)
    st.session_state.user_name = user_name

    password = st.text_input("Password", type="password")
    st.session_state.password = password
    if st.session_state.password:
        sc = "Welcome " + st.session_state.user_name + " please proceed to Stroke Prediction"
        st.success(sc)


if selected == "Stroke Prediction":

    with st.sidebar:
        st.header("Input Variable:")
        age = st.number_input("Age", step=1)
        glucose = st.number_input("Glucose level", step=1, min_value=50)
        weight = st.number_input("Weight (Kg)", step=1, min_value=20)
        height = st.number_input("Height (cm)", step=1, min_value=100)
        smoking = st.selectbox(
            "Smoking Status", ("Smokes", "Formely smoked", "Never smoked", "Unknown")
        )
        gender = st.selectbox("Gender", ("Male", "Female"))
        hypertension = st.selectbox("Hypertention", ("No History", "Hystory"))
        work_type = st.selectbox(
            "Work Type",
            ("Goverment Job", "Self-employed", "Private", "Never Worked", "Children"),
        )
        Residence_type = st.selectbox("Residence Type", ("Rural", "Urban"))
        ever_married = st.selectbox("Martial Status", ("No", "Yes"))
        heart_history = st.selectbox("Heart Disease", ("No History", "History"))

        inputs = [
            1 if gender == "Male" else 0,
            age,
            1 if hypertension == "Hystory" else 0,
            1 if heart_history == "History" else 0,
            1 if ever_married == "Yes" else 0,
            2
            if work_type == "Goverment Job"
            else 1
            if work_type == "Self-employed"
            else 0
            if work_type == "Private"
            else -1
            if work_type == "Children"
            else -2,
            1 if Residence_type == "Urban" else 0,
            glucose,
            float(weight / (height * 0.01)),
            2
            if smoking == "Smokes"
            else 1
            if smoking == "Formely smoked"
            else 0
            if smoking == "Never smoked"
            else -1,
        ]
        print(inputs)
        st.session_state.inputs = inputs

    prediction = model.predict(inputs)
    proba = model.predict_proba(inputs)

    # sidebar
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Stroke Prediction and Confidence</h1>",
        unsafe_allow_html=True,
    )

    if prediction == 1:
        html_temp = """
            <div style="background-color:{};padding:10px;border-radius:10px 10px 10px 10px">
            <h2 style="color:{};text-align:center;">High Stroke Risk</h2>
            </div>
            """

        bg_results = ["#F1E1E6", "#7D363B"]

        persen_proba = float(round((proba - 0.3) / 0.7, 2))

        liquidfill_option = {
            "series": [{"type": "liquidFill", "data": [persen_proba], "color": [colors[0]]}]
        }

    else:

        html_temp = """
            <div style="background-color:{};padding:10px;border-radius:10px 10px 10px 10px">
            <h2 style="color:{};text-align:center;">Low Stroke Risk</h2>
            </div>
            """

        bg_results = ["#D3E0EA", "#276678"]

        persen_proba = float(round((0.3 - proba) / 0.3, 2))

        liquidfill_option = {
            "series": [{"type": "liquidFill", "data": [persen_proba], "color": [colors[1]]}]
        }

    st.markdown(html_temp.format(bg_results[0], bg_results[1]), unsafe_allow_html=True)

    input_shap = np.asarray(inputs).reshape(1, -1)
    input_shap = pd.DataFrame(input_shap, columns=colname)
    shap_values = explainer.shap_values(input_shap)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :], input_shap))

    st_echarts(liquidfill_option)

    # input variables
    # st.subheader("Input Varibles")
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     age = st.number_input("Age", step=1)
    # with col2:
    #     glucose = st.number_input("Glucose level", step=1, min_value=50)
    # with col3:
    #     weight = st.number_input("Weight (Kg)", step=1, min_value=20)
    # with col4:
    #     height = st.number_input("Height (cm)", step=1, min_value=100)

    # ccol1, ccol2, ccol3, ccol4 = st.columns(4)
    # with ccol1:
    #     smoking = st.selectbox(
    #         "Smoking Status", ("Smokes", "Formely smoked", "Never smoked", "Unknown")
    #     )
    # with ccol2:
    #     gender = st.selectbox("Gender", ("Male", "Female"))
    # with ccol3:
    #     hypertension = st.selectbox("Hypertention", ("No History", "Hystory"))
    # with ccol4:
    #     work_type = st.selectbox(
    #         "Work Type",
    #         ("Goverment Job", "Self-employed", "Private", "Never Worked", "Children"),
    #     )

    # ccol5, ccol6, ccol7, ccol8 = st.columns(4)
    # with ccol5:
    #     Residence_type = st.selectbox("Residence Type", ("Rural", "Urban"))
    # with ccol6:
    #     ever_married = st.selectbox("Martial Status", ("No", "Yes"))
    # with ccol7:
    #     heart_history = st.selectbox("Heart Disease", ("No History", "History"))

    # inputs = [
    #     1 if gender == "Male" else 0,
    #     age,
    #     1 if hypertension == "Hystory" else 0,
    #     1 if heart_history == "History" else 0,
    #     1 if ever_married == "Yes" else 0,
    #     2
    #     if work_type == "Goverment Job"
    #     else 1
    #     if work_type == "Self-employed"
    #     else 0
    #     if work_type == "Private"
    #     else -1
    #     if work_type == "Children"
    #     else -2,
    #     1 if Residence_type == "Urban" else 0,
    #     glucose,
    #     float(weight / (height * 0.01)),
    #     2
    #     if smoking == "Smokes"
    #     else 1
    #     if smoking == "Formely smoked"
    #     else 0
    #     if smoking == "Never smoked"
    #     else -1,
    # ]
    # print(inputs)
    # st.session_state.inputs = inputs

    # predict_sidebar(inputs=inputs)

    # res1, res2 = st.columns((7, 3))

    # with res1:
    #     with st.spinner(text="In Progress"):
    #         prediction = model.predict(inputs)
    #         proba = model.predict_proba(inputs)

    #         st.subheader(" ")
    #         st.subheader("Prediction Explainer")
    #         input_shap = np.asarray(inputs).reshape(1, -1)
    #         input_shap = pd.DataFrame(input_shap, columns=colname)
    #         shap_values = explainer.shap_values(input_shap)
    #         st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :], input_shap))

    # with res2:
    #     # sidebar
    #     st.markdown(
    #         "<h2 style='text-align: center; color: black;'>Stroke Prediction and Confidence</h2>",
    #         unsafe_allow_html=True,
    #     )

    #     if prediction == 1:
    #         html_temp = """
    #             <div style="background-color:{};padding:10px;border-radius:10px 10px 10px 10px">
    #             <h2 style="color:{};text-align:center;">High Stroke Risk</h2>
    #             </div>
    #             """

    #         bg_results = ["#F1E1E6", "#7D363B"]

    #         persen_proba = float(round((proba - 0.3) / 0.7, 2))

    #         liquidfill_option = {
    #             "series": [{"type": "liquidFill", "data": [persen_proba], "color": [colors[0]]}]
    #         }

    #     else:

    #         html_temp = """
    #             <div style="background-color:{};padding:10px;border-radius:10px 10px 10px 10px">
    #             <h2 style="color:{};text-align:center;">Low Stroke Risk</h2>
    #             </div>
    #             """

    #         bg_results = ["#D3E0EA", "#276678"]

    #         persen_proba = float(round((0.3 - proba) / 0.3, 2))

    #         liquidfill_option = {
    #             "series": [{"type": "liquidFill", "data": [persen_proba], "color": [colors[1]]}]
    #         }

    #     st.markdown(html_temp.format(bg_results[0], bg_results[1]), unsafe_allow_html=True)

    #     st_echarts(liquidfill_option)


if selected == "Stroke Database":
    # predict_sidebar(inputs=st.session_state.inputs, shap_value=False)

    # Database distribution
    st.subheader("Stroke Database Distribution")

    # tab plot
    (
        tab_age,
        tab_smoking,
        tab_gender,
        tab_heart,
        tab_glucose,
        tab_bmi,
        tab_hypertension,
        tab_work,
        tab_importance,
    ) = st.tabs(
        [
            "Age",
            "Smoking",
            "Gender",
            "Heart Disease",
            "Glucose",
            "BMI",
            "Hypertension",
            "Work Type",
            "Shap Importance",
        ]
    )

    group_labels = ["Positive", "Negative"]

    with tab_age:

        fig = ff.create_distplot(
            [str_only["age"].values, no_str_only["age"].values],
            group_labels=group_labels,
            colors=colors,
        )

        fig.update_layout(
            width=800,
            height=600,
            yaxis=dict(tickfont=dict(size=20), showticklabels=False),
            xaxis=dict(tickfont=dict(size=20)),
        )

        st.plotly_chart(fig)

        # left, middle, right = st.columns((2, 5, 2))
        # with middle:
        #     st.plotly_chart(fig)

    with tab_smoking:
        positive = pd.DataFrame(str_only["smoking_status"].value_counts())
        positive["Percentage"] = positive["smoking_status"].apply(
            lambda x: x / sum(positive["smoking_status"])
        )
        negative = pd.DataFrame(no_str_only["smoking_status"].value_counts())
        negative["Percentage"] = negative["smoking_status"].apply(
            lambda x: x / sum(negative["smoking_status"])
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=negative["Percentage"],
                y=negative.index,
                orientation="h",
                name="Negative",
                marker_color=colors[1],
                text=negative["Percentage"],
            )
        )
        fig.add_trace(
            go.Bar(
                x=positive["Percentage"],
                y=positive.index,
                orientation="h",
                name="Positive",
                marker_color=colors[0],
                text=positive["Percentage"],
            )
        )
        fig.update_layout(
            width=800,
            height=600,
            xaxis=dict(tickformat=",.0%", range=[0, 0.4], tickfont=dict(size=20)),
            yaxis=dict(tickfont=dict(size=20)),
        )
        fig.update_traces(texttemplate="%{text:.2%}", textposition="inside", textfont_size=16)
        st.plotly_chart(fig)

    with tab_gender:
        positive = pd.DataFrame(str_only["gender"].value_counts())
        positive["Percentage"] = positive["gender"].apply(lambda x: x / sum(positive["gender"]))
        negative = pd.DataFrame(no_str_only["gender"].value_counts())
        negative["Percentage"] = negative["gender"].apply(lambda x: x / sum(negative["gender"]))

        negative_ind = ["Male" if i == 1 else "Female" for i in negative.index]
        positive_ind = ["Male" if i == 1 else "Female" for i in positive.index]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=negative["Percentage"],
                x=negative_ind,
                name="Negative",
                marker_color=colors[1],
                text=negative["Percentage"],
            )
        )
        fig.add_trace(
            go.Bar(
                y=positive["Percentage"],
                x=positive_ind,
                name="Positive",
                marker_color=colors[0],
                text=positive["Percentage"],
            )
        )
        fig.update_layout(
            width=800,
            height=600,
            yaxis=dict(
                tickformat=",.0%",
                range=[0, 0.7],
                tickfont=dict(size=20),
                tickmode="linear",
                tick0=0.0,
                dtick=0.1,
            ),
            xaxis=dict(tickfont=dict(size=20)),
        )

        fig.update_traces(texttemplate="%{text:.2%}", textposition="inside", textfont_size=19)
        st.plotly_chart(fig)

    with tab_heart:
        positive = pd.DataFrame(str_only["heart_disease"].value_counts())
        positive["Percentage"] = positive["heart_disease"].apply(
            lambda x: x / sum(positive["heart_disease"])
        )
        negative = pd.DataFrame(no_str_only["heart_disease"].value_counts())
        negative["Percentage"] = negative["heart_disease"].apply(
            lambda x: x / sum(negative["heart_disease"])
        )

        negative_ind = ["History" if i == 1 else "No History" for i in negative.index]
        positive_ind = ["History" if i == 1 else "No History" for i in positive.index]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=negative["Percentage"],
                x=negative_ind,
                name="Negative",
                marker_color=colors[1],
                text=negative["Percentage"],
            )
        )
        fig.add_trace(
            go.Bar(
                y=positive["Percentage"],
                x=positive_ind,
                name="Positive",
                marker_color=colors[0],
                text=positive["Percentage"],
            )
        )
        fig.update_layout(
            width=800,
            height=600,
            yaxis=dict(
                tickformat=",.0%",
                range=[0, 1],
                tickfont=dict(size=20),
                tickmode="linear",
                tick0=0.0,
                dtick=0.2,
            ),
            xaxis=dict(tickfont=dict(size=20)),
        )

        fig.update_traces(texttemplate="%{text:.2%}", textposition="inside", textfont_size=19)
        st.plotly_chart(fig)

    with tab_glucose:

        fig = ff.create_distplot(
            [
                str_only["avg_glucose_level"].values,
                no_str_only["avg_glucose_level"].values,
            ],
            group_labels=group_labels,
            colors=colors,
        )

        fig.update_layout(
            width=800,
            height=600,
            yaxis=dict(tickfont=dict(size=20), showticklabels=False),
            xaxis=dict(tickfont=dict(size=20)),
        )

        st.plotly_chart(fig)

    with tab_bmi:

        fig = ff.create_distplot(
            [
                str_only["bmi"].values,
                no_str_only["bmi"].values,
            ],
            group_labels=group_labels,
            colors=colors,
        )

        fig.update_layout(
            width=800,
            height=600,
            yaxis=dict(tickfont=dict(size=20), showticklabels=False),
            xaxis=dict(tickfont=dict(size=20)),
        )

        st.plotly_chart(fig)

    with tab_hypertension:
        positive = pd.DataFrame(str_only["hypertension"].value_counts())
        positive["Percentage"] = positive["hypertension"].apply(
            lambda x: x / sum(positive["hypertension"])
        )
        negative = pd.DataFrame(no_str_only["hypertension"].value_counts())
        negative["Percentage"] = negative["hypertension"].apply(
            lambda x: x / sum(negative["hypertension"])
        )

        negative_ind = ["History" if i == 1 else "No History" for i in negative.index]
        positive_ind = ["History" if i == 1 else "No History" for i in positive.index]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=negative["Percentage"],
                x=negative_ind,
                name="Negative",
                marker_color=colors[1],
                text=negative["Percentage"],
            )
        )
        fig.add_trace(
            go.Bar(
                y=positive["Percentage"],
                x=positive_ind,
                name="Positive",
                marker_color=colors[0],
                text=positive["Percentage"],
            )
        )
        fig.update_layout(
            width=800,
            height=600,
            yaxis=dict(
                tickformat=",.0%",
                range=[0, 1],
                tickfont=dict(size=20),
                tickmode="linear",
                tick0=0.0,
                dtick=0.2,
            ),
            xaxis=dict(tickfont=dict(size=20)),
        )

        fig.update_traces(texttemplate="%{text:.2%}", textposition="inside", textfont_size=19)
        st.plotly_chart(fig)

    with tab_work:
        positive = pd.DataFrame(str_only["work_type"].value_counts())
        positive.rename(
            index={0: "Private", 1: "Self-employed", 2: "Govt_job", 255: "children"},
            inplace=True,
        )

        positive["Percentage"] = positive["work_type"].apply(
            lambda x: x / sum(positive["work_type"])
        )
        positive = positive.sort_index()

        negative = pd.DataFrame(no_str_only["work_type"].value_counts())
        negative.rename(
            index={
                0: "Private",
                1: "Self-employed",
                2: "Govt_job",
                255: "children",
                254: "Never Worked",
            },
            inplace=True,
        )

        negative["Percentage"] = negative["work_type"].apply(
            lambda x: x / sum(negative["work_type"])
        )
        negative = negative.sort_index()

        negative_ind = negative.index
        positive_ind = positive.index

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=negative["Percentage"],
                x=negative_ind,
                name="Negative",
                marker_color=colors[1],
                text=negative["Percentage"],
            )
        )
        fig.add_trace(
            go.Bar(
                y=positive["Percentage"],
                x=positive_ind,
                name="Positive",
                marker_color=colors[0],
                text=positive["Percentage"],
            )
        )
        fig.update_layout(
            width=800,
            height=600,
            yaxis=dict(
                tickformat=",.0%",
                range=[0, 0.7],
                tickfont=dict(size=20),
                tickmode="linear",
                tick0=0.0,
                dtick=0.2,
            ),
            xaxis=dict(tickfont=dict(size=20)),
        )

        fig.update_traces(texttemplate="%{text:.2%}", textposition="inside", textfont_size=19)
        st.plotly_chart(fig)

    with tab_importance:
        df_summary = pd.read_csv("data/shap_summary.csv")

        fig = px.bar(
            df_summary,
            x="feature_importance_vals",
            y="col_name",
            category_orders={"col_name": df_summary.col_name.values},
        )
        fig.update_layout(
            width=800,
            height=600,
            xaxis=dict(tickformat=",.2f", range=[0, 2.5], tickfont=dict(size=20)),
            yaxis=dict(tickfont=dict(size=20)),
            yaxis_title=None,
            xaxis_title=None,
        )

        st.plotly_chart(fig)


footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p><b>Stroke Dashboard V1</b><br>© 2022 Data Science Division<a style='display: block; text-align: center;' href="https://www.nanosense-id.com/" target="_blank">PT Nanosense Instrument Indonesia</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
