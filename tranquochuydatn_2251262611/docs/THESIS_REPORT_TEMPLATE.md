# De Cuong Bao Cao Do An Tot Nghiep (Mau)

## Bia va thong tin
- Ten de tai: Du doan muc nuoc ngam bang hoc may va danh gia theo chuoi thoi gian
- Sinh vien: ...
- GVHD: ...
- Bo mon/Khoa: ...
- Thoi gian: ...

## Loi cam on
(Viet ngan gon, 0.5 trang)

## Tom tat de tai (Abstract)
- Bai toan
- Du lieu
- Phuong phap
- Ket qua chinh (RMSE/MAE/R2)
- Tu khoa

## Chuong 1. Gioi thieu
1.1 Dat van de
1.2 Muc tieu nghien cuu
1.3 Doi tuong, pham vi
1.4 Dong gop cua de tai
1.5 Cau truc bao cao

## Chuong 2. Co so ly thuyet va cong trinh lien quan
2.1 Tong quan du bao chuoi thoi gian
2.2 Cac chi so danh gia (MAE, RMSE, MAPE, R2)
2.3 Cac huong du bao (recursive/direct)
2.4 Nghien cuu lien quan
2.5 Khoang trong nghien cuu

## Chuong 3. Du lieu va tien xu ly
3.1 Nguon du lieu
- Muc nuoc ngam: USGS NWIS Daily Values, tram 323527117050002, California.
- Tham so 72019: do sau den muc nuoc, don vi feet duoi mat dat.
- Chuoi thong ke 00002 (minimum) duoc chon de moi ngay chi co mot gia tri nhat quan.
- Mua va nhiet do: NASA POWER Daily API/MERRA-2 tai toa do tram 32.59100556, -117.083475.
- PRECTOTCORR: luong mua hieu chinh (mm/ngay); T2M: nhiet do 2 m (do C).
- Link API chinh xac duoc luu trong data/real/groundwater_weather_real.meta.json.
3.2 Mo ta cot du lieu
- date
- groundwater_level
- rainfall_mm
- temperature_c
3.3 Lam sach va ghep du lieu
- Khong tron cac chuoi maximum/minimum/median cua USGS.
- Sap xep va kiem tra trung ngay.
- Inner join one-to-one theo ngay; ket qua 2.914 ngay tu 2018-01-01 den 2025-12-31.
- Khong con ngay trung va khong thieu rainfall/temperature trong tap da ghep.
3.4 Tao dac trung:
- Lag features
- Rolling mean/std
- Bien thoi tiet rainfall_mm va temperature_c
3.5 Chia tap train/test theo thoi gian

## Chuong 4. De xuat mo hinh va quy trinh
4.1 Kien truc he thong
4.2 VAR/VECM (quan he tuyen tinh, dong lien ket)
4.3 LSTM (phi tuyen, phu thuoc dai han)
4.4 Hybrid VAR/VECM + LSTM residual
4.5 Benchmark voi naive_last
4.6 Rolling-origin backtesting nhieu horizon
4.7 API va giao dien demo

## Chuong 5. Thuc nghiem va danh gia
5.1 Cau hinh thuc nghiem (phan cung, thu vien)
5.2 Ket qua holdout
- Bang tu `artifacts/tuned_metrics.json`
5.3 Ket qua backtest nhieu horizon
- Bang tu `artifacts/backtest_metrics.csv`
- Hinh `artifacts/backtest_rmse.png`
- Hinh `artifacts/backtest_curve_last_origin.png`
5.4 Nhan xet
- Uu/nhuoc diem
- Horizon nao mo hinh yeu
5.5 Ablation bien thoi tiet
- So sanh cung time split: groundwater-only voi groundwater + rainfall + temperature.
- Khong dien giai chenhlech metric thanh quan he nhan qua.
5.6 De xuat cai tien

## Chuong 6. Ket luan va huong phat trien
6.1 Ket luan
6.2 Han che
6.3 Huong phat trien
- Them XGBoost/LightGBM
- Mo rong GRU/Transformer
- Thu them bom hut, do am dat va nhieu tram quan trac
- Mo rong multi-site

## Tai lieu tham khao
- USGS Water Services: https://waterservices.usgs.gov/
- NASA POWER Data Access Viewer: https://power.larc.nasa.gov/data-access-viewer/

## Phu luc
- Link source code
- Huong dan chay
- Mau payload API
