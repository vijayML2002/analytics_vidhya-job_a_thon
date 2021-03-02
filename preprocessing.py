import pandas as pd

df_train = pd.read_csv('train_Df64byy.csv')
df_test = pd.read_csv('test_YCcRUnU.csv')

HI_mean = 2
HPD_mean = 8
HPT_mean = 2

city_hash = {
    'C1':1,'C2':2,'C3':3,'C4':4,'C5':5,'C6':6,
    'C7':7,'C8':8,'C9':9,'C10':10,'C11':11,'C12':12,
    'C13':13,'C14':14,'C15':15,'C16':16,'C17':17,'C18':18,
    'C19':19,'C20':20,'C21':21,'C22':22,'C23':23,'C24':24,
    'C25':25,'C26':26,'C27':27,'C28':28,'C29':29,'C30':30,
    'C31':31,'C32':32,'C33':33,'C34':34,'C35':35,'C36':36
    }

acc_hash = {
    'Owned':1,
    'Rented':0
    }

insurance_type_hash = {
    'Individual':1,
    'Joint':0
    }

spouse_hash = {
    'No':0,
    'Yes':1
    }

indication_hash = {
    'X1':1,'X2':2,'X3':3,'X4':4,'X5':5,
    'X6':6,'X7':7,'X8':8,'X9':9,float("nan"):2
    }

policy_duration_hash = {
    '1.0':1.0,'2.0':2.0,'3.0':3.0,'4.0':4.0,'5.0':5.0,
    '6.0':6.0,'7.0':7.0,'8.0':8.0,'9.0':9.0,'10.0':10.0,
    '11.0':11.0,'12.0':12.0,'13.0':13.0,'14.0':14.0,'14+':30.0,
    float("nan"):8.0
    }

holding_type_hash = {float("nan"):2}

traindf_replaced = df_train.replace({
    'City_Code':city_hash,
    'Accomodation_Type':acc_hash,
    'Reco_Insurance_Type':insurance_type_hash,
    'Is_Spouse':spouse_hash,
    'Health Indicator':indication_hash,
    'Holding_Policy_Duration':policy_duration_hash,
    'Holding_Policy_Type':holding_type_hash
    })


testdf_replaced = df_test.replace({
    'City_Code':city_hash,
    'Accomodation_Type':acc_hash,
    'Reco_Insurance_Type':insurance_type_hash,
    'Is_Spouse':spouse_hash,
    'Health Indicator':indication_hash,
    'Holding_Policy_Duration':policy_duration_hash,
    'Holding_Policy_Type':holding_type_hash
    })

traindf_replaced.to_csv('train.csv',index=False)
testdf_replaced.to_csv('test.csv',index=False)


