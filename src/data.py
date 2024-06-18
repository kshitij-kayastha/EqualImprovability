import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSIncome


def df_to_array(*dfs):
    return tuple(map(lambda df: df.to_numpy(dtype=np.float64), dfs))

def df_to_tensor(*dfs):
    return tuple(map(torch.FloatTensor, dfs))


class FairnessDataset(Dataset):
    def __init__(self, X, Y, Z, imp_feats) -> None:
        self.X = X
        self.Y = Y
        self.Z = Z
        self.U_index = imp_feats['U_index']
        self.C_index = imp_feats['C_index']
        self.C_min = imp_feats['C_min']
        self.C_max = imp_feats['C_max']
        self.sensitive_attrs = sorted(list(set(self.Z.numpy())))
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        x, y, z = self.X[index], self.Y[index], self.Z[index]
        return x, y, z


class SyntheticDataset():
    def __init__(self, num_samples=20000, z1_mean=0.3, z2_mean=0.5, seed=None):
        self.num_samples = num_samples
        self.z1_mean = z1_mean
        self.z2_mean = z2_mean
        
        rng = np.random.default_rng(seed)
        
        xs, ys, zs = [], [], []

        x_dist = {
            (0,0): {'mean':(-0.1,-0.2), 'cov': np.array([[0.4,0.0], [0.0,0.4]])},
            (0,1): {'mean': (-0.2,-0.3), 'cov': np.array([[0.2,0.0], [0.0,0.2]])},
            (1,0): {'mean': (0.1,0.4), 'cov': np.array([[0.2,0.0], [0.0,0.2]])},
            (1,1): {'mean': (0.4,0.3), 'cov': np.array([[0.1,0.0], [0.0,0.1]])},
            }
        y_means = [self.z1_mean, self.z2_mean]
        
        for _ in range(self.num_samples):
            z = rng.binomial(n = 1, p = 0.4, size = 1)[0]
            y = rng.binomial(n = 1, p = y_means[z], size = 1)[0]
            x = rng.multivariate_normal(mean = x_dist[(y,z)]['mean'], cov = x_dist[(y,z)]['cov'], size = 1)[0]
            xs.append(x)
            ys.append(y)
            zs.append(z)

        data = pd.DataFrame(zip(np.array(xs).T[0], np.array(xs).T[1], ys, zs), columns = ['x1', 'x2', 'y', 'z'])
        rng.shuffle(data.values)
        
        self.X = data[['x1', 'x2']]
        self.Y = data['y']
        self.Z = data['z']
        self.XZ = data[['x1', 'x2', 'z']]
    
        self.sensitive_attrs = sorted(list(set(self.Z)))

        self.set_improvable_features()
    
    def set_improvable_features(self):
        self.imp_feats = {
            'U_index': [2],
            'C_index': [],
            'C_min': [],
            'C_max': []
        }
    
    def split_data(self, fold):
        n, m = 5, self.num_samples
        fold = fold % 5
        x_chunks, y_chunks, z_chunks, xz_chunks = [], [], [], []
        
        for i in range(n):
            start = int(i/n * m)
            end = int((i+1)/n * m)
            x_chunks.append(self.X.iloc[start:end])
            y_chunks.append(self.Y.iloc[start:end])
            z_chunks.append(self.Z.iloc[start:end])
            xz_chunks.append(self.XZ.iloc[start:end])
            
        self.X_test, self.Y_test, self.Z_test, self.XZ_test = x_chunks.pop(fold), y_chunks.pop(fold), z_chunks.pop(fold), xz_chunks.pop(fold)
        train_dataset = pd.concat(x_chunks), pd.concat(y_chunks), pd.concat(z_chunks), pd.concat(xz_chunks)
        
        self.X_train, self.X_val, self.Y_train, self.Y_val, self.Z_train, self.Z_val, self.XZ_train, self.XZ_val = train_test_split(*train_dataset, train_size=0.8, random_state=fold)
        
        return self
               
    def numpy(self, fold=0, z_blind=True):
        self.split_data(fold)
        if z_blind:
            train_arrays = df_to_array(self.X_train, self.Y_train, self.Z_train)
            val_arrays = df_to_array(self.X_val, self.Y_val, self.Z_val)
            test_arrays = df_to_array(self.X_test, self.Y_test, self.Z_test)
        else:
            train_arrays = df_to_array(self.XZ_train, self.Y_train, self.Z_train)
            val_arrays = df_to_array(self.XZ_val, self.Y_val, self.Z_val)
            test_arrays = df_to_array(self.XZ_test, self.Y_test, self.Z_test)
        return train_arrays, val_arrays, test_arrays
               
    def tensor(self, fold=0, z_blind=True):
        self.split_data(fold)
        if z_blind:
            train_tensors = df_to_tensor(self.X_train.values, self.Y_train.values, self.Z_train.values)
            val_tensors = df_to_tensor(self.X_val.values, self.Y_val.values, self.Z_val.values)
            test_tensors = df_to_tensor(self.X_test.values, self.Y_test.values, self.Z_test.values)
        else:
            train_tensors = df_to_tensor(self.XZ_train.values, self.Y_train.values, self.Z_train.values)
            val_tensors = df_to_tensor(self.XZ_val.values, self.Y_val.values, self.Z_val.values)
            test_tensors = df_to_tensor(self.XZ_test.values, self.Y_test.values, self.Z_test.values)
        return train_tensors, val_tensors, test_tensors

    
class GermanDataset():
    def __init__(self, seed: int | None = None):

        # data = pd.read_csv('../data/german.data', header = None, delim_whitespace = True)
        data = pd.read_csv('../data/german.data', header = None, sep = '\s+').sample(frac=1, random_state=seed)
        self.num_samples = len(data)
        
        data.columns=['Existing-Account-Status','Month-Duration','Credit-History','Purpose','Credit-Amount','Saving-Account','Present-Employment','Instalment-Rate','Sex','Guarantors','Residence','Property','Age','Installment','Housing','Existing-Credits','Job','Num-People','Telephone','Foreign-Worker','Status']
        self.cat_feats=['Credit-History','Purpose','Present-Employment', 'Sex','Guarantors','Property','Installment','Telephone','Foreign-Worker','Existing-Account-Status','Saving-Account','Housing','Job']
        self.num_feats =['Month-Duration','Credit-Amount']
        
        label_encoder = LabelEncoder()
        for x in self.cat_feats:
            data[x]=label_encoder.fit_transform(data[x])
            data[x].unique()

        data.loc[data['Age']<=30, 'Age'] = 0
        data.loc[data['Age']>30, 'Age'] = 1
        data=data.rename(columns = {'Age':'z'})

        data.loc[data['Status']==2, 'Status'] = 0
        data=data.rename(columns = {'Status':'y'})
        
        data[self.num_feats] = data[self.num_feats].astype(float)
        
        self.Z = data['z']
        self.Y = data['y']
        self.X = data.drop(labels=['z','y'], axis=1)
        self.XZ = pd.concat([self.X, self.Z], axis=1)

        self.sensitive_attrs = sorted(list(set(self.Z)))

        self.set_improvable_features()
        
    def set_improvable_features(self):
        self.imp_feats = {
            'U_index': np.setdiff1d(np.arange(20),[0,5,14,16]),
            'C_index': [0,3,7,9],
            'C_min': [0,0,0,0],
            'C_max': [3,4,2,3]
        }

    def split_data(self, fold):
        n, m = 5, self.num_samples
        fold = fold % 5
        x_chunks, y_chunks, z_chunks, xz_chunks = [], [], [], []
        
        for i in range(n):
            start = int(i/n * m)
            end = int((i+1)/n * m)
            x_chunks.append(self.X.copy().iloc[start:end])
            y_chunks.append(self.Y.copy().iloc[start:end])
            z_chunks.append(self.Z.copy().iloc[start:end])
            xz_chunks.append(self.XZ.copy().iloc[start:end])
            
        self.X_test, self.Y_test, self.Z_test, self.XZ_test = x_chunks.pop(fold), y_chunks.pop(fold), z_chunks.pop(fold), xz_chunks.pop(fold)
        train_dataset = pd.concat(x_chunks), pd.concat(y_chunks), pd.concat(z_chunks), pd.concat(xz_chunks)
        
        scaler = StandardScaler()
        train_dataset[0].loc[:, self.num_feats] = scaler.fit_transform(train_dataset[0][self.num_feats])
        train_dataset[-1].loc[:, self.num_feats] = scaler.transform(train_dataset[-1][self.num_feats])
        self.X_test.loc[:, self.num_feats] = scaler.transform(self.X_test[self.num_feats])
        self.XZ_test.loc[:, self.num_feats] = scaler.transform(self.XZ_test[self.num_feats])
        
        self.X_train, self.X_val, self.Y_train, self.Y_val, self.Z_train, self.Z_val, self.XZ_train, self.XZ_val = train_test_split(*train_dataset, train_size=0.8, random_state=fold)
        
        return self
               
    def numpy(self, fold=0, z_blind=True):
        self.split_data(fold)
        if z_blind:
            train_arrays = df_to_array(self.X_train, self.Y_train, self.Z_train)
            val_arrays = df_to_array(self.X_val, self.Y_val, self.Z_val)
            test_arrays = df_to_array(self.X_test, self.Y_test, self.Z_test)
        else:
            train_arrays = df_to_array(self.XZ_train, self.Y_train, self.Z_train)
            val_arrays = df_to_array(self.XZ_val, self.Y_val, self.Z_val)
            test_arrays = df_to_array(self.XZ_test, self.Y_test, self.Z_test)
        return train_arrays, val_arrays, test_arrays
               
    def tensor(self, fold=0, z_blind=True):
        self.split_data(fold)
        if z_blind:
            train_tensors = df_to_tensor(self.X_train.values, self.Y_train.values, self.Z_train.values)
            val_tensors = df_to_tensor(self.X_val.values, self.Y_val.values, self.Z_val.values)
            test_tensors = df_to_tensor(self.X_test.values, self.Y_test.values, self.Z_test.values)
        else:
            train_tensors = df_to_tensor(self.XZ_train.values, self.Y_train.values, self.Z_train.values)
            val_tensors = df_to_tensor(self.XZ_val.values, self.Y_val.values, self.Z_val.values)
            test_tensors = df_to_tensor(self.XZ_test.values, self.Y_test.values, self.Z_test.values)
        return train_tensors, val_tensors, test_tensors
    
    
class IncomeDataset():
    def __init__(self, seed: int | None = None) -> None:
        datasource = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        ca_data = datasource.get_data(states=['CA'], download=True)
        
        ca_features, ca_labels, _ = ACSIncome.df_to_pandas(ca_data)
        ca_features['SEX'] = ca_features['SEX'].map({2.0: 1, 1.0: 0}).astype(int)
        data = pd.concat([ca_features, ca_labels], axis=1)
        self.num_samples = len(data)
        
        data['PINCP'] = data['PINCP'].map({True: 1, False:0}).astype(int)
        data = data.rename(columns = {'SEX':'z'})
        
        self.cat_feats = ['COW','MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P']
        self.num_feats = ['AGEP', 'WKHP']
        data = pd.get_dummies(data, columns=self.cat_feats)
        data = data.rename(columns = {'PINCP':'y'})
        
        self.Z = data['z'].astype(float)
        self.Y = data['y'].astype(float)
        self.X = data.drop(labels=['z','y'], axis=1).astype(float)
        self.XZ = pd.concat([self.X, self.Z], axis=1).astype(float)

        self.sensitive_attrs = sorted(list(set(self.Z)))

        self.set_improvable_features()
    
    def set_improvable_features(self):
        self.imp_feats = {
            'U_index': np.setdiff1d(np.arange(785),[1]),
            'C_index': [1],
            'C_min': [1],
            'C_max': [24]
        }

    def split_data(self, fold):
        n, m = 5, self.num_samples
        fold = fold % 5
        x_chunks, y_chunks, z_chunks, xz_chunks = [], [], [], []
        
        for i in range(n):
            start = int(i/n * m)
            end = int((i+1)/n * m)
            x_chunks.append(self.X.iloc[start:end])
            y_chunks.append(self.Y.iloc[start:end])
            z_chunks.append(self.Z.iloc[start:end])
            xz_chunks.append(self.XZ.iloc[start:end])
            
        self.X_test, self.Y_test, self.Z_test, self.XZ_test = x_chunks.pop(fold), y_chunks.pop(fold), z_chunks.pop(fold), xz_chunks.pop(fold)
        train_dataset = pd.concat(x_chunks), pd.concat(y_chunks), pd.concat(z_chunks), pd.concat(xz_chunks)
        
        scaler = StandardScaler()
        train_dataset[0].loc[:, self.num_feats] = scaler.fit_transform(train_dataset[0][self.num_feats])
        train_dataset[-1].loc[:, self.num_feats] = scaler.transform(train_dataset[-1][self.num_feats])
        self.X_test.loc[:, self.num_feats] = scaler.transform(self.X_test[self.num_feats])
        self.XZ_test.loc[:, self.num_feats] = scaler.transform(self.XZ_test[self.num_feats])
        
        self.X_train, self.X_val, self.Y_train, self.Y_val, self.Z_train, self.Z_val, self.XZ_train, self.XZ_val = train_test_split(*train_dataset, train_size=0.8, random_state=fold)
        
        return self
               
    def numpy(self, fold=0, z_blind=True):
        self.split_data(fold)
        if z_blind:
            train_arrays = df_to_array(self.X_train, self.Y_train, self.Z_train)
            val_arrays = df_to_array(self.X_val, self.Y_val, self.Z_val)
            test_arrays = df_to_array(self.X_test, self.Y_test, self.Z_test)
        else:
            train_arrays = df_to_array(self.XZ_train, self.Y_train, self.Z_train)
            val_arrays = df_to_array(self.XZ_val, self.Y_val, self.Z_val)
            test_arrays = df_to_array(self.XZ_test, self.Y_test, self.Z_test)
        return train_arrays, val_arrays, test_arrays
               
    def tensor(self, fold=0, z_blind=True):
        self.split_data(fold)
        if z_blind:
            train_tensors = df_to_tensor(self.X_train.values, self.Y_train.values, self.Z_train.values)
            val_tensors = df_to_tensor(self.X_val.values, self.Y_val.values, self.Z_val.values)
            test_tensors = df_to_tensor(self.X_test.values, self.Y_test.values, self.Z_test.values)
        else:
            train_tensors = df_to_tensor(self.XZ_train.values, self.Y_train.values, self.Z_train.values)
            val_tensors = df_to_tensor(self.XZ_val.values, self.Y_val.values, self.Z_val.values)
            test_tensors = df_to_tensor(self.XZ_test.values, self.Y_test.values, self.Z_test.values)
        return train_tensors, val_tensors, test_tensors
        
        