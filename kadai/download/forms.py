from django import forms
from .kadai import predict
from django.core.files.storage import default_storage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class UploadFileForm(forms.Form):
    file = forms.FileField(label='CSV file')

    def save(self):
        upload_file = self.cleaned_data['file']
        predict_file = predict(self.file)
        file_name = default_storage.save(upload_file.name, predict_file)
        return default_storage.url(file_name)
