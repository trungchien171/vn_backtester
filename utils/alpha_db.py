# alpha_db.py
import io
import os
import pandas as pd
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

def get_user_alpha_file_id(drive_service, username):
    """Retrieve the file ID of the user's alpha data file from Google Drive."""
    results = drive_service.files().list(q=f"name='{username}_submitted_alphas.csv'", fields="files(id, name)").execute()
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

def submit_alpha(drive_service, username, alpha_formula, alpha_settings, metrics, weight_metrics):
    alpha_df = load_user_alphas(drive_service, username)
    weights_metrics_flat = weight_metrics.to_numpy().flatten()
    new_alpha = pd.DataFrame({
        "alpha": [alpha_formula],
        "settings": [alpha_settings],
        "Sharpe": [f"{metrics['Sharpe']:.2f}"],
        "Turnover (%)": [f"{metrics['Turnover (%)']:.2f}"],
        "Returns (%)": [f"{metrics['Returns (%)']:.2f}"],
        "Fitness": [f"{metrics['Fitness']:.2f}"],
        "Drawdown (%)": [f"{metrics['Drawdown (%)']:.2f}"],
        "Margin (%)": [f"{metrics['Margin (%)']:.2f}"],
        "Weight Metrics": [weights_metrics_flat]
    })
    alpha_df = pd.concat([alpha_df, new_alpha], ignore_index=True)
    save_user_alphas(drive_service, username, alpha_df)

def load_all_submitted_alphas(drive_service):
    results = drive_service.files().list(q="name contains '_submitted_alphas.csv'", fields="files(id, name)").execute()
    items = results.get("files", [])

    if not items:
        return pd.DataFrame()
    
    all_alphas = []

    for item in items:
        file_id = item["id"]
        request = drive_service.files().get_media(fileId=file_id)
        file_data = io.BytesIO()
        downloader = MediaIoBaseDownload(file_data, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        file_data.seek(0)
        alpha_df = pd.read_csv(file_data)
        all_alphas.append(alpha_df)
    
    return pd.concat(all_alphas, ignore_index=True) if all_alphas else pd.DataFrame()