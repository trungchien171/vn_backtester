# alpha_db.py
import io
import os
import pandas as pd
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

def get_user_alpha_file_id(drive_service, username):
    """Retrieve the file ID of the user's alpha data file from Google Drive."""
    results = drive_service.files().list(q=f"name='{username}_alphas.csv'", fields="files(id, name)").execute()
    items = results.get("files", [])
    if not items:
        return None
    return items[0]["id"]

def load_user_alphas(drive_service, username):
    file_id = get_user_alpha_file_id(drive_service, username)
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
        return pd.DataFrame(columns=["alpha", "settings"])
    
def save_user_alphas(drive_service, username, alpha_df):
    file_id = get_user_alpha_file_id(drive_service, username)
    file_metadata = {"name": f"{username}_alphas.csv"}
    file_data = io.BytesIO()
    alpha_df.to_csv(file_data, index=False)
    file_data.seek(0)
    media = MediaIoBaseUpload(file_data, mimetype="text/csv", resumable=True)
    
    if file_id:
        drive_service.files().update(fileId=file_id, media_body=media).execute()
    else:
        drive_service.files().create(body=file_metadata, media_body=media).execute()

def submit_alpha(drive_service, username, alpha_formula, alpha_settings, metrics):
    alpha_df = load_user_alphas(drive_service, username)
    new_alpha = pd.DataFrame({
        "alpha": [alpha_formula],
        "settings": [alpha_settings],
        "Sharpe": [metrics["Sharpe"]],
        "Turnover (%)": [metrics["Turnover (%)"]],
        "Returns (%)": [metrics["Returns (%)"]],
        "Fitness": [metrics["Fitness"]],
        "Drawdown (%)": [metrics["Drawdown (%)"]],
        "Margin (%)": [metrics["Margin (%)"]]
    })
    alpha_df = pd.concat([alpha_df, new_alpha], ignore_index=True)
    save_user_alphas(drive_service, username, alpha_df)