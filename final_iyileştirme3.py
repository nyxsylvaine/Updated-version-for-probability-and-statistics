import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path
from prophet import Prophet
import plotly.graph_objs as go

# Bugünün tarihi
end_date = datetime.now()
# 4 yıl önceki tarih
start_date = end_date - timedelta(days=4 * 365)

# Popüler hisse senetleri listesi
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'BRK-B', 'NVDA',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'DIS', 'VZ', 'NFLX',
    'PYPL', 'INTC', 'CMCSA'
]

# Çalışma dizinini al
base_directory = Path(__file__).parent

# Veriler ve grafikler için göreceli dizinler oluştur
save_directory = base_directory / "Veriler"
graph_directory = base_directory / "Grafikler"

# Hisse verilerini saklamak için boş bir liste oluştur
all_data = []

# Her bir hisse senedi için verileri çek
print("Veriler çekilmeye başlandı...")
for ticker in tickers:
    print(f"{ticker} için veri çekiliyor...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        print(f"{ticker} için veri bulunamadı.")
    else:
        data['Ticker'] = ticker  # Ticker bilgisini ekle
        data.reset_index(inplace=True)  # Tarih sütununu indeks yerine sütun olarak ekle
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
        all_data.append(data)

if not all_data:
    print("Hiçbir veri çekilemedi.")
else:
    all_data = pd.concat(all_data, ignore_index=True)
    all_data = all_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    all_data = all_data.infer_objects()
    all_data.interpolate(method='linear', inplace=True)
    all_data.ffill(inplace=True)
    all_data.bfill(inplace=True)

    current_date = datetime.now().strftime("%Y%m%d_%H%M")
    data_directory = save_directory / f"{current_date}_veriler"
    os.makedirs(data_directory, exist_ok=True)
    file_path = data_directory / f"{current_date}_hisse_verileri.csv"
    all_data.to_csv(file_path, index=False, sep=';')

    # İstatistikleri saklamak için bir liste oluştur
    stats_list = []

    predictions = []
    for ticker in tickers:
        print(f"{ticker} için tahmin yapılıyor...")
        ticker_data = all_data[all_data['Ticker'] == ticker][['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        ticker_data = ticker_data.dropna()
        ticker_data['ds'] = ticker_data['ds'].dt.tz_localize(None)

        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_prior_scale=10.0,
            changepoint_prior_scale=0.5
        )
        model.fit(ticker_data)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        forecast['Real Price'] = ticker_data['y'].tolist() + [None] * 30
        forecast['Ticker'] = ticker
        predictions.append(forecast[['ds', 'Ticker', 'yhat', 'Real Price']])

    predictions = pd.concat(predictions, ignore_index=True)
    graph_folder = graph_directory / f"{current_date}_grafikler"
    os.makedirs(graph_folder, exist_ok=True)

    print("\nÇekilen veriler : ")
    print(all_data)  # Tüm verileri terminalde göster

    print("\nGrafikler oluşturuluyor...")
    for ticker in tickers:
        ticker_data = predictions[predictions['Ticker'] == ticker]

        valid_data = ticker_data.dropna(subset=['Real Price', 'yhat'])
        accuracy = 100 - ((abs(valid_data['Real Price'] - valid_data['yhat']) / valid_data['Real Price']).mean() * 100)

        fiyat_ort = valid_data['Real Price'].mean()  # Ortalama
        fiyat_mod = valid_data['Real Price'].mode()[0]  # Mod
        fiyat_std = valid_data['Real Price'].std()  # Standart Sapma
        risk_ratio = (fiyat_std / fiyat_ort) * 100  # Risk Oranı
        fiyat_medyan = valid_data['Real Price'].median()  # Medyan

        # İstatistikleri tabloya ekle
        stats_list.append({
            'Hisse': ticker,
            'Doğruluk Oranı (%)': round(accuracy, 2),
            'Risk Oranı (%)': round(risk_ratio, 2),
            'Ortalama Fiyat (USD)': round(fiyat_ort, 2),
            'Mod Fiyat (USD)': round(fiyat_mod, 2),
            'Medyan Fiyat (USD)': round(fiyat_medyan, 2)
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_data['ds'], y=ticker_data['Real Price'], mode='lines', name='Gerçek Fiyat'))
        fig.add_trace(go.Scatter(x=ticker_data['ds'], y=ticker_data['yhat'], mode='lines', name='Tahmin Edilen Fiyat', line=dict(dash='dot')))

        fig.update_layout(
            title=f"{ticker} Fiyat Tahmini",
            xaxis_title="Tarih",
            yaxis_title="Fiyat",
            annotations=[  
                dict(  
                    x=0.01,  
                    y=0.99,  
                    xref="paper",  
                    yref="paper",  
                    showarrow=False,  
                    text=(  
                        f"<b>Doğruluk Oranı (%):</b> {accuracy:.2f} - Tahminlerin gerçek fiyatlara uyum oranı.<br>"  
                        f"<b>Risk Oranı (%):</b> {risk_ratio:.2f} - Fiyat dalgalanma oranı (volatilite).<br>"  
                        f"<b>Ortalama Fiyat:</b> {fiyat_ort:.2f} USD<br>"  
                        f"<b>Mod Fiyat:</b> {fiyat_mod:.2f} USD<br>"  
                        f"<b>Medyan Fiyat:</b> {fiyat_medyan:.2f} USD"  # Medyan fiyatı burada göster
                    ),  
                    align="left",  
                    font=dict(size=12, color="black"),  
                    bgcolor="rgba(255, 255, 255, 0.7)",  
                    bordercolor="black",  
                    borderwidth=1  
                )  
            ],
            xaxis_rangeslider_visible=True
        )

        graph_path = graph_folder / f"{ticker}_fiyat_tahmini_{current_date}.html"
        fig.write_html(graph_path)

    print("\nGrafikler oluşturuldu.")

    # İstatistikleri tablo olarak terminalde göster
    stats_df = pd.DataFrame(stats_list)
    stats_df.reset_index(drop=True, inplace=True)
    stats_df.index = stats_df.index + 1  # İndeks numarasını 1'den başlat

    print("\nİstatistikler:")
    print(stats_df)
