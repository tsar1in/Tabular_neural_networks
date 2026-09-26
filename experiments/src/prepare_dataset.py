import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from .num_encoders.periodic_encoder import PeriodicEncoder
from .num_encoders.ple_encoder import PiecewiseLinearEncoder
from .num_encoders.ewme_encoder import ElementWiseMultiplEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


cat_encoders_dict = {
    "ohe": OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
    "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
}

num_encoders_dict = {
    "ple": PiecewiseLinearEncoder(20),
    "periodic": PeriodicEncoder(n_frequencies=3, initialization='log-linear', sigma=10.0),
    "ewme": ElementWiseMultiplEncoder(8),
    None: None
}


class TabDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = {
            'num': data['num'],
            'cat': data['cat']
        }
        self.n_features_num = self.data['num'].shape[1]
        self.n_features_all = self.n_features_num + self.data['cat'].shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x_item = (self.data['num'][index], self.data['cat'][index])
        y_item = self.labels[index]

        return x_item, y_item


class PrepareTabularData:
    def __init__(self, os_path, test_size, val_size, target_column, type_model="NN",
                 num_encoder="ple", cat_encoder="ohe", scaler=StandardScaler()):
        self.os_path = os_path
        self.test_size = test_size
        self.val_size = val_size

        self.scaler = scaler
        self.num_encoder = num_encoders_dict[num_encoder]
        self.cat_encoder = cat_encoders_dict[cat_encoder]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.type_model = type_model

        self.target_column = target_column
        self.num_features = None
        self.cat_features = None
        self.cat_cardinalities = None

        self.df = None
        self.dataset = None

        self.lookup_layers = None

    def shape(self):
        return self.df.shape
    
    def head(self, n=5):
        return self.df.head(n)

    def load_data(self):
        self.df = pd.read_csv(self.os_path)
        if self.os_path == "data/first_dataset/application_train.csv":
            self.df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "SK_ID_CURR"], inplace=True)

        print(f"Data downloaded successfully in the amount of {self.df.shape[0]}")

    def columns_type(self):
        self.num_features = sorted(
            [col for col in self.df.select_dtypes(include=['float64']).columns
             if col != self.target_column]
        )
        self.cat_features = sorted(
            [col for col in self.df.select_dtypes(exclude=['float64']).columns
             if col != self.target_column]
        )

        return self.num_features, self.cat_features

    def fill_nones(self):
        for col in self.num_features:
            self.df[col] = self.df[col].fillna(self.df[col].mean())

        for col in self.cat_features:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    def split_to_train_test(self):
        data_train, data_test = train_test_split(self.df, test_size=self.test_size, random_state=42, stratify=self.df[self.target_column])
        data_test, data_val = train_test_split(data_test, test_size=self.val_size, random_state=42, stratify=data_test[self.target_column])

        print(f"Train shape:    {data_train.shape}",
              f"Test shape:     {data_test.shape}",
              f"Validate shape: {data_val.shape}", sep="\n")

        data = {
            'train': {
                'cat': data_train.drop(columns=self.target_column)[self.cat_features],
                'num': data_train.drop(columns=self.target_column)[self.num_features],
            },
            'test': {
                'cat': data_test.drop(columns=self.target_column)[self.cat_features],
                'num': data_test.drop(columns=self.target_column)[self.num_features],
            },
            'val': {
                'cat': data_val.drop(columns=self.target_column)[self.cat_features],
                'num': data_val.drop(columns=self.target_column)[self.num_features],
            }
        }

        labels = {
            'train': data_train[self.target_column].values,
            'test': data_test[self.target_column].values,
            'val': data_val[self.target_column].values
        }

        return data, labels

    def prepare_num_features(self, data):
        self.scaler.fit(data['train']['num'])

        for data_type in ('train', 'test', 'val'):
            data[data_type]['num'] = pd.DataFrame(
                data=self.scaler.transform(data[data_type]['num']),
                columns=data[data_type]['num'].columns
            )

        if self.num_encoder is not None: # для проверки что дважды энкодер не применяется
            self.num_encoder.fit(data['train']['num'])

            for data_type in ('train', 'test', 'val'):
                data[data_type]['num'] = self.num_encoder.transform(data[data_type]['num'])

        return data

    def prepare_cat_features(self, data):
        cat_features = list(data['train']['cat'].columns)
        self.cat_cardinalities = [data['train']['cat'][col].nunique() for col in cat_features]

        for col in cat_features:
            known_values = set(data['train']['cat'][col].dropna().unique())
            most_common_val = data['train']['cat'][col].value_counts().idxmax()

            for data_type in ('test', 'val'):
                mask = ~data[data_type]['cat'][col].isin(known_values)
                data[data_type]['cat'].loc[mask, col] = most_common_val

        column_transformer = make_column_transformer(
            (self.cat_encoder, cat_features),
            remainder='passthrough'
        )

        column_transformer = column_transformer.fit(data['train']['cat'])

        for data_type in ('train', 'test', 'val'):
            data[data_type]['cat'] = pd.DataFrame(
                column_transformer.transform(data[data_type]['cat']),
                columns=column_transformer.get_feature_names_out()
            )

        return data
    
    def make_lookup_layers(self):
        lookup_layers = {}

        for c in self.cat_features:
            unique_values = self.df[c].unique()
            lookup_layers[c] = {
                'vocab': {v: i for i, v in enumerate(unique_values)},
                'vocab_size': len(unique_values)
            }

        for n in self.num_features:
            mean = self.df[n].mean()
            std = self.df[n].std()
            lookup_layers[n] = {
                'mean': mean,
                'std': std
            }

        return lookup_layers

    def prepare_torch_tensors(self, data, labels):
        for data_type in ('train', 'test', 'val'):
            for feature_type in ('cat', 'num'):
                type = (torch.int64 if feature_type == 'cat' and self.type_model == "Transformer" else torch.float32)
                if feature_type == 'cat' and self.type_model == "Transformer":
                    type = torch.int64
                elif feature_type == 'cat' and self.type_model == "MLP-Transformer":
                    type = torch.long
                else:
                    type = torch.float32

                data[data_type][feature_type] = torch.tensor(
                    data[data_type][feature_type].values,
                    device=self.device,
                    dtype=type
                )

            labels[data_type] = torch.tensor(
                labels[data_type],
                device=self.device,
                dtype=torch.float32
            )

        return data, labels
    
    def torch_tensors_dimensions(self):
        return self.dataset["train"].data['num'].shape[1], self.dataset["train"].data['cat'].shape[1]

    def prepare_dataset(self):
        self.columns_type()

        self.fill_nones()

        data, labels = self.split_to_train_test()

        if self.type_model != "Boosting":
            data = self.prepare_num_features(data)
            data = self.prepare_cat_features(data)

            self.lookup_layers = self.make_lookup_layers()

            data, labels = self.prepare_torch_tensors(data, labels)

        self.dataset = {
            df_part: TabDataset(data[df_part], labels[df_part])
            for df_part in ("train", "test", "val")
        }

        return self.dataset