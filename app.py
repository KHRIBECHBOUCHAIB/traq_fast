import streamlit as st

import pandas as pd
import numpy as np
#import base64
import datetime
import hmac
#import time
import xlsxwriter
#from bson import ObjectId
from datetime import datetime, timedelta, date
from streamlit_date_picker import date_range_picker, date_picker, PickerType
from io import BytesIO
import pickle
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

st.set_page_config(layout="wide")
st.title("Traq Fast Questionnaire")




sex_mapping = {'male': 0, 'female': 1}
answers = {}




st.markdown(
        """<style>
        div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
                }
        </style>
                """, unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .centered_button {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)





st.sidebar.header('Informations')




def stringify(i:int = 0) -> str:
    return slider_strings[i-1]



def read_data_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    all_lists = [item.strip() for item in file_content.split("***")]
    #print(all_lists)

    return all_lists

all_data = read_data_file("questionnaire.txt")

#st.write(f"""# {all_data[0]}""")

all_questions = all_data[1].split("\n\n")

Comp = []
loop = 0
if not st.checkbox("Avec Fast"):
    loop = 1
    Comp = all_questions[0].split("\n")
else:
    loop = 2




slider_strings = all_data[2].split("\n")
slider_values = [i for i in range(int(all_data[3]), len(slider_strings) + 1)]
slider_values_inv = [i for i in range(len(slider_strings), int(all_data[3])-1, -1)]

def user_input_features(Comp):
        #current_date = datetime.date.today()
        #code = st.sidebar.text_input("Code")
        lastName = st.sidebar.text_input("Nom")
        firstName = st.sidebar.text_input("Prénom")
        default_value = datetime.now()
        with st.sidebar.container():
            st.write("Date de Naissance")
            birthDate = date_picker(picker_type=PickerType.date, value=default_value, key='date_picker')
        #select_date = date_picker(picker_type=PickerType.date, value=default_value, key='date_picker')
        #date = st.sidebar.date_input("Date de naissance", datetime.date(2010, 1, 1))
        #age = current_date.year - date.year - ((current_date.month, current_date.day) < (date.month, date.day))
        sex = st.sidebar.selectbox('Genre',('Homme','Femme'))
        #age = st.sidebar.number_input('Age:', min_value=0, max_value=120, step=1)
        educationLevel = st.sidebar.selectbox("Niveau d'etude",('CAP/BEP','Baccalauréat professionnel','Baccalauréat général', 'Bac +2 (DUT/BTS)', 'Bac +3 (Licence)',
                                                       'Bac +5 (Master)', 'Bac +7 (Doctorat, écoles supérieurs)'))
        #questionnaire = st.sidebar.selectbox('Questionnaire',('TRAQ','FAST','TRAQ+FAST'))
        #st.write("""## Concernant mon utilisation de la planche de transfert:""")
        if (loop == 1):
            param = Comp[0]
            Comp = Comp[1:]
            for i, question in enumerate(Comp, start=1):
                if(question[0] == "-"):
                    slider_output = st.select_slider(
                    #f":red[{question}]",
                    f"{question[1:]}",
                    options=slider_values_inv,
                    value=slider_values_inv[0],
                    format_func=stringify
                    )
                else:
                    slider_output = st.select_slider(
                    #f":red[{question}]",
                    f"{question}",
                    options=slider_values,
                    value=slider_values[0],
                    format_func=stringify
                    )
                answers[f"{param}{i}"] = slider_output
        else:
            for j in range(len(all_questions)):
                Comp = all_questions[j].split("\n")
                param = Comp[0]
                Comp = Comp[1:]
                for i, question in enumerate(Comp, start=1):
                    if(question[0] == "-"):
                        slider_output = st.select_slider(
                        f"{question[1:]}",
                        options=slider_values_inv,
                        value=slider_values_inv[0],
                        format_func=stringify
                        )
                    else:
                        slider_output = st.select_slider(
                        f"{question}",
                        options=slider_values,
                        value=1,
                        format_func=stringify
                        )
                    answers[f"{param}{i}"] = slider_output


        user_data = {#'Questionnaire': all_data[0],
                     #'Code': code,
                     'lastName': lastName,
                     'firstName': firstName,
                     'birthDate': birthDate,
                     'sex': sex,
                     'educationLevel': educationLevel}
                     #'Age': age}
                     #'Qdate': datetime.date.today().strftime('%Y-%m-%d')}
                     #'educationalLevel': study}
        answers_data = answers

        document = {
        #"_id": ObjectId(),  # Generate a new ObjectId
        "user": user_data,
        "answers": answers_data
        #"__v": 0
        }
                
        return document



document = user_input_features(Comp)
#result = check_data(code)

if "disabled" not in st.session_state:
    st.session_state.disabled = False
     
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    button = st.button('Enregistrer', disabled=st.session_state.disabled)
    st.image("clinicogImg.png", width=200)
    


def calc_traq(answers: pd.DataFrame) -> pd.DataFrame:
    """TRAQ"""
    attention = ['TRAQ1', 'TRAQ2', 'TRAQ4', 'TRAQ8', 'TRAQ10']
    inhibition = ['TRAQ3', 'TRAQ5', 'TRAQ6', 'TRAQ7', 'TRAQ9']
    total = ['traq_attention', 'traq_inhibition']

    answers['traq_attention'] = answers[attention].sum(axis=1)
    answers['traq_inhibition'] = answers[inhibition].sum(axis=1)
    answers['traq_total'] = answers[total].sum(axis=1)

    return answers

def calc_fast(answers: pd.DataFrame) -> pd.DataFrame:
    """FAST"""
    soc_int = ['FAST01', 'FAST02', 'FAST03', 'FAST04']
    stereotypie = ['FAST07', 'FAST08']
    total = ['fast_soc-int', 'fast_stereotypie']

    answers["fast_soc-int"] = answers[soc_int].sum(axis=1)
    answers["fast_stereotypie"] = answers[stereotypie].sum(axis=1)
    answers["fast_total"] = answers[total].sum(axis=1)

    return answers

def calc_centile(groups: pd.DataFrame, type: str):
    """CENTILE"""
    excels = get_centiles(type=type)
    centiles = pd.DataFrame({'CENTILES': ["/"]})

    columns = {
        "traq": ["traq_attention", "traq_inhibition", "traq_total"],
        "fast": ["fast_soc-int", "fast_stereotypie", "fast_total"]
    }

    # 1 centile for every group
    for column in columns.get(type):
        # get selected data centiles
        data = excels.get(column.split("_")[1])
        names = list(data.columns[1:])

        # get value from each calc
        value = groups[column].values[0]
        # find closest
        adhd = min(data[names[0]].to_list(), key=lambda x: abs(x-value))
        control = min(data[names[1]].to_list(), key=lambda x: abs(x-value))
        # find centile
        adhd = data["Centile"][data[names[0]] == adhd].values
        control = data["Centile"][data[names[1]] == control].values

        # Does have multiple lines ? surround with min < x < max
        if(len(adhd) > 1):
            adhd = f"{adhd[0]} < x < {adhd[-1]}"
        if(len(control) > 1):
            control = f"{control[0]} < x < {control[-1]}"

        centiles[f"{column}_{names[0]}_centile"] = adhd
        centiles[f"{column}_{names[1]}_centile"] = control

    return centiles

def get_centiles(type: str):
    if(type == "traq"):
        xlsx = pd.ExcelFile("traq_centiles.xlsx")

        return {
            "attention": pd.read_excel(xlsx, "Attention"),
            "inhibition": pd.read_excel(xlsx, "Inhibition"),
            "total": pd.read_excel(xlsx, "Total")
        }
    elif (type == "fast"):
        xlsx = pd.ExcelFile("fast_centiles.xlsx")

        return {
            "soc-int": pd.read_excel(xlsx, "soc-int"),
            "stereotypie": pd.read_excel(xlsx, "stereotypie"),
            "total": pd.read_excel(xlsx, "total")
        }

if button:
    if loop == 1:
        answers  = pd.DataFrame([document["answers"].values()], columns=document["answers"].keys())
        #st.write(answers)

        user = document["user"]
        #writer = pd.ExcelWriter("./Patients/" + user['lastName'] + ".xlsx", engine='xlsxwriter')
        user_df = pd.DataFrame([document["user"].values()], columns=document["user"].keys())
        #user_df.to_excel(writer, sheet_name='patient', index=False)
        #st.write(user_df)

        filter_col = [col for col in answers if col.startswith('TRAQ')]
        traq = calc_traq(answers[filter_col])
        centiles = calc_centile(traq, "traq")
        result = traq.join(centiles)
        #st.write(result)

        traq = result
        gender = user['sex']
        today = date.today()
        birthdate = datetime.strptime(user['birthDate'], "%Y-%m-%d").date()
        age = [today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))]

        #traq = pd.read_excel("./Patients/" + user['lastName']+".xlsx",sheet_name="TRAQ")
        df_traq = pd.DataFrame(columns = ["gender","age","traq1","traq2","traq3","traq4","traq5","traq6","traq7","traq8","traq9","traq10"])
        df_traq["gender"] = ["Homme"]
        df_traq["gender"] = gender
        df_traq["age"] = age
        df_traq["traq1"] = [answers["TRAQ1"][0]]
        df_traq["traq2"] = [answers["TRAQ2"][0]]
        df_traq["traq3"] = [answers["TRAQ3"][0]]
        df_traq["traq4"] = [answers["TRAQ4"][0]]
        df_traq["traq5"] = [answers["TRAQ5"][0]]
        df_traq["traq6"] = [answers["TRAQ6"][0]]
        df_traq["traq7"] = [answers["TRAQ7"][0]]
        df_traq["traq8"] = [answers["TRAQ8"][0]]
        df_traq["traq9"] = [answers["TRAQ9"][0]]
        df_traq["traq10"] = [answers["TRAQ10"][0]]

        

        df_traq.loc[(df_traq["gender"] == "Homme") , "gender"] = "1"
        df_traq.loc[(df_traq["gender"] == "Femme") , "gender"] = "0"

        df_traq['gender'] = df_traq['gender'].astype('int')

        #st.write(df_traq)



        params_filename = 'traq_class.pkl'
        with open(params_filename, 'rb') as f:
            bst = pickle.load(f)

        y_pred_proba = bst.predict_proba(df_traq.values)



        if type(traq["traq_total_ADHD_centile"].iloc[0]) != str:
            traqTotAD = traq["traq_total_ADHD_centile"].iloc[0]
        else:
            traqTotAD = (int(traq["traq_total_ADHD_centile"].iloc[0][-2:])+int(traq["traq_total_ADHD_centile"].iloc[0][0:2]))/2

        if type(traq["traq_attention_ADHD_centile"].iloc[0]) != str:
            traqAttAD = traq["traq_attention_ADHD_centile"].iloc[0]
        else:
            traqAttAD = (int(traq["traq_attention_ADHD_centile"].iloc[0][-2:])+int(traq["traq_attention_ADHD_centile"].iloc[0][0:2]))/2

        if type(traq["traq_inhibition_ADHD_centile"].iloc[0]) != str:
            traqInhAD = traq["traq_inhibition_ADHD_centile"].iloc[0]
        else:
            traqInhAD = (int(traq["traq_inhibition_ADHD_centile"].iloc[0][-2:])+int(traq["traq_inhibition_ADHD_centile"].iloc[0][0:2]))/2

        if type(traq["traq_total_Controls_centile"].iloc[0]) != str:
            traqTotC = traq["traq_total_Controls_centile"].iloc[0]
        else:
            traqTotC = (int(traq["traq_total_Controls_centile"].iloc[0][-2:])+int(traq["traq_total_Controls_centile"].iloc[0][0:2]))/2

        if type(traq["traq_attention_Controls_centile"].iloc[0]) != str:
            traqAttC = traq["traq_attention_Controls_centile"].iloc[0]
        else:
            traqAttC = (int(traq["traq_attention_Controls_centile"].iloc[0][-2:])+int(traq["traq_attention_Controls_centile"].iloc[0][0:2]))/2

        if type(traq["traq_inhibition_Controls_centile"].iloc[0]) != str:
            traqInhC = traq["traq_inhibition_Controls_centile"].iloc[0]
        else:
            traqInhC = (int(traq["traq_inhibition_Controls_centile"].iloc[0][-2:])+int(traq["traq_inhibition_Controls_centile"].iloc[0][0:2]))/2

        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(10, 5))
        # Adjust the spacing between subplots
        #fig.subplots_adjust(hspace=0.5, wspace=0.4)

        # Set the width of each bar
        bar_width = 0.35

        ### Analyse factorielle ADHD
        adhd_x = ["Total", "Inattention", "Impulsivité"]
        #adhd_y1 = [1, 20, 1]
        adhd_y1 = [traqTotC,
                   traqAttC,
                   traqInhC]


        adhd_y2 = [traqTotAD,
                   traqAttAD,
                   traqInhAD]


        ax1 = plt.subplot2grid((1, 1), (0, 0))
        ax1.set_title('Analyse factorielle')
        ax1.set_xlabel("Trognon & Richard ADHD Questionnaire")
        ax1.set_ylabel("Rang percentile (Pop.cliniques)")
        ax1.set_ylim(0, 110)
        ax1.bar(
            np.arange(len(adhd_x)),
            adhd_y1,
            width=bar_width,
            color = "Blue"
        )
        ax1.bar(
            np.arange(len(adhd_x)) + bar_width,
            adhd_y2,
            width=bar_width,
            color = "Red"
        )
        ax1.set_xticks(np.arange(len(adhd_x)) + bar_width / 2)
        ax1.set_xticklabels(adhd_x)

        # Add text to the bars
        for i in range(len(adhd_x)):
            ax1.text(i, adhd_y1[i] + 1, adhd_y1[i], ha='center')
            ax1.text(i + bar_width, adhd_y2[i] + 1, adhd_y2[i], ha='center')

        st.pyplot(fig)

        fig1_buffer = BytesIO()
        fig.savefig(fig1_buffer, format='png')
        fig1_buffer.seek(0)


        # TRAQ plot

        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(9, 2), dpi=100)
        bar_width = 0.35

        ### Analyse factorielle ADHD
        adhd_x = ["Total", "Inattention", "Impulsivité"]
        #adhd_y1 = [1, 20, 1]
        adhd_y1 = [traqTotC,
                traqAttC,
                traqInhC]


        adhd_y2 = [traqTotAD,
                traqAttAD,
                traqInhAD]


        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax1.set_title('Analyse factorielle')
        ax1.set_xlabel("Trognon & Richard ADHD Questionnaire")
        ax1.set_ylabel("Rang percentile (Pop.cliniques)")
        ax1.set_ylim(0, 110)
        #ax1.set_aspect('auto')
        ax1.bar(
            np.arange(len(adhd_x)),
            adhd_y1,
            width=bar_width,
            color = "Blue"
        )
        ax1.bar(
        np.arange(len(adhd_x)) + bar_width,
        adhd_y2,
        width=bar_width,
        color = "Red"
        )
        ax1.set_xticks(np.arange(len(adhd_x)) + bar_width / 2)
        ax1.set_xticklabels(adhd_x)

        # Add text to the bars
        for i in range(len(adhd_x)):
            ax1.text(i, adhd_y1[i] + 1, adhd_y1[i], ha='center')
            ax1.text(i + bar_width, adhd_y2[i] + 1, adhd_y2[i], ha='center')

        # -------------------
        comp_x = ["TDAH"]
        comp_y1 = [y_pred_proba[0,0]]
        comp_y2 = [y_pred_proba[0,1]]

        ## Diagnostic computationnel
        ax3 = plt.subplot2grid((1, 2), (0, 1))
        ax3.set_title('Diagnostic computationnel')
        ax3.set_ylim(0, 1.2)
        #ax3.set_aspect('auto')
        ax3.set_ylabel("Probabilité(%)")

        ax3.bar(
            np.arange(len(comp_x)),
            comp_y1,
            width=bar_width,
            label='Psychotypique',
            color = "Blue"
        )

        ax3.bar(
            np.arange(len(comp_x)) + bar_width,
            comp_y2,
            width=bar_width,
            label='Clinique',
            color= 'Red'
        )


        # Add text to the bars
        for i, (x, y1, y2) in enumerate(zip(comp_x, comp_y1, comp_y2)):
            ax3.text(i, y1 + 0.01, f"{y1:.2f}", ha='center', color='black')
            ax3.text(i + bar_width, y2 + 0.01, f"{y2:.2f}", ha='center', color='black')

        ax3.set_xticks(np.arange(len(comp_x)) + bar_width / 2)

        # Add dotted threshold line
        ax3.axhline(y=0.5, color='gray', linestyle='--', label='Seuil de décidabilité')


        # Add legend and axis labels ( put in good order ['Seuil de décidabilité', 'Psychotypique', 'Clinique'])
        handles, labels = ax3.get_legend_handles_labels()
        handles = handles[1:] + [handles[0]]
        labels = labels[1:] + [labels[0]]

        #ax3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        ax3.legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.1, -0.3), ncol=3)  # Set loc='lower center'
        ax3.set_xticklabels(comp_x)

        st.pyplot(fig)

        fig2_buffer = BytesIO()
        fig.savefig(fig2_buffer, format='png')
        fig2_buffer.seek(0)


        percentile = pd.DataFrame(columns = ["model","parameter","total","percentileRankControl","percentileRankClinical"])
        percentile["model"] = ["traq", "traq", "traq"]
        percentile["parameter"] = ["attention", "impulsivity", "global"]
        percentile["total"] = [traq["traq_attention"].iloc[0], traq["traq_inhibition"].iloc[0], traq["traq_total"].iloc[0]]
        percentile["percentileRankControl"] = [traqAttC,
                                            traqInhC,
                                            traqTotC]
        percentile["percentileRankClinical"] = [traqAttAD,
                                            traqInhAD,
                                            traqTotAD]

        #percentile.to_excel("./Patients/"+ givenLastName + "_centile.xlsx")

        proba = pd.DataFrame(columns = ["model", "group", "probability"])
        proba["model"] = ["traq", "traq"]
        proba["group"] = ["control", "adhd"]
        proba["probability"] = [y_pred_proba[0,:][0], y_pred_proba[0,:][1]]

        excel_buffer = BytesIO()

        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            user_df.to_excel(writer, sheet_name='patient', index=False)
            traq.to_excel(writer, sheet_name='TRAQ', index=False)
            percentile.to_excel(writer, sheet_name='percentile', index=False)
            proba.to_excel(writer, sheet_name='probabilite', index=False)

        # Rewind the buffer
        excel_buffer.seek(0)

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('figure1.png', fig1_buffer.getvalue())
            zf.writestr('figure2.png', fig2_buffer.getvalue())
            zf.writestr(f"{user['lastName']}.xlsx", excel_buffer.getvalue())
        zip_buffer.seek(0)

        # Provide a download button
        st.download_button(
            label="Download ZIP file",
            data=zip_buffer,
            file_name=f"{user['lastName']}_data.zip",
            mime='application/zip'
        )

       