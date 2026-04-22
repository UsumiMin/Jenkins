import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/UsumiMin/lab_ML_1/refs/heads/main/StudentsPerformance.csv', delimiter = ',')
    df.to_csv("students.csv", index = False)
    print("df: ", df.shape)
    return df

def clear_data():
    df = pd.read_csv("students.csv")
    
    cat_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    num_columns = ['math score', 'reading score', 'writing score']
    
    question_score = df[(df['math score'] < 0) | (df['math score'] > 100)]
    df = df.drop(question_score.index)
    question_score = df[(df['reading score'] < 0) | (df['reading score'] > 100)]
    df = df.drop(question_score.index)
    question_score = df[(df['writing score'] < 0) | (df['writing score'] > 100)]
    df = df.drop(question_score.index)

    question_gender = df[~((df['gender'] == 'male') | (df['gender'] == 'female'))]
    df = df.drop(question_gender.index)
    
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns]);
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    df.to_csv('df_clear.csv')
    return True

download_data()
clear_data()
