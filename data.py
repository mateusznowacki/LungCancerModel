import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


class DatasetPreparer:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = None
        self.df_prepared = None
        self.df_smote = None

    def load_dataset(self):
        self.df = pd.read_csv(self.input_file)

    def clean_dataset(self):
        # Usunięcie duplikatów kolumn
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]

        # Uzupełnianie braków
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                if self.df[column].dtype in [np.float64, np.int64]:
                    mean_value = self.df[column].mean()
                    self.df[column] = self.df[column].fillna(mean_value)
                else:
                    mode_value = self.df[column].mode()[0]
                    self.df[column] = self.df[column].fillna(mode_value)

        # Zamiana wartości tekstowych na liczby
        if 'Gender' in self.df.columns:
            self.df['Gender'] = self.df['Gender'].map({'Male': 1, 'Female': 0})

        self.df_prepared = self.df.copy()

    def save_prepared_dataset(self, output_file):
        self.df_prepared.to_csv(output_file, index=False)

    def apply_smote(self):
        features = self.df_prepared.drop(columns=['Patient Id', 'Result', 'Level'])
        labels = self.df_prepared['Result']

        # Zamiana danych na tablice
        X = features.values
        y = labels.values

        # Tworzenie SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Odtworzenie DataFrame
        df_features = pd.DataFrame(X_resampled, columns=features.columns)
        df_labels = pd.DataFrame(y_resampled, columns=['Result'])

        # Dodanie sztucznych Patient Id i Level
        df_features['Patient Id'] = ['Generated_' + str(i) for i in range(len(df_features))]
        df_features['Level'] = ['Unknown' for _ in range(len(df_features))]

        # Kolejność kolumn
        self.df_smote = pd.concat([
            df_features[['Patient Id']],
            df_features.drop(columns=['Patient Id', 'Level']),
            df_features[['Level']],
            df_labels
        ], axis=1)

    def save_smote_dataset(self, output_file):
        self.df_smote.to_csv(output_file, index=False)

    def prepare_all(self):
        self.load_dataset()
        self.clean_dataset()
        self.save_prepared_dataset('dataset_prepared.csv')
        self.apply_smote()
        self.save_smote_dataset('dataset_smote.csv')


if __name__ == "__main__":
    preparer = DatasetPreparer('dataset.csv')
    preparer.prepare_all()
