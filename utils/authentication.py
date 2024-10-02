# authentication.py
import io
import os
import base64
import hashlib
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

def authenticate_gdrive():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["google_drive"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=credentials)

def load_user_data(drive_service):
    file_id = get_user_data_file_id(drive_service)
    if file_id:
        request = drive_service.files().get_media(fileId=file_id)
        file_data = io.BytesIO()
        downloader = MediaIoBaseDownload(file_data, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        file_data.seek(0)
        return pd.read_csv(file_data)
    else:
        return pd.DataFrame(columns=["username", "password"])

def get_user_data_file_id(drive_service):
    results = drive_service.files().list(q="name='user_data.csv'", fields="files(id, name)").execute()
    items = results.get("files", [])
    if not items:
        return None
    return items[0]["id"]

def save_user_data(drive_service, df):
    file_id = get_user_data_file_id(drive_service)
    file_metadata = {"name": "user_data.csv"}
    file_data = io.BytesIO()
    df.to_csv(file_data, index=False)
    file_data.seek(0)

    media = MediaIoBaseUpload(file_data, mimetype="text/csv", resumable=True)
    if file_id:
        drive_service.files().update(fileId=file_id, media_body=media).execute()
    else:
        drive_service.files().create(body=file_metadata, media_body=media).execute()

def rerun():
    st.rerun()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_account(username, password, drive_service, user_data):
    if username in user_data["username"].values:
        return False
    new_entry = pd.DataFrame({"username": [username], "password": [hash_password(password)]})
    user_data = pd.concat([user_data, new_entry], ignore_index=True)
    save_user_data(drive_service, user_data)
    return True

def login(username, password, user_data):
    if username not in user_data["username"].values:
        return False, "Username does not exist. Please sign up first."
    stored_password = user_data.loc[user_data["username"] == username, "password"].values[0]
    if stored_password == hash_password(password):
        return True, "Logged in successfully!"
    else:
        return False, "Incorrect password."
    
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")