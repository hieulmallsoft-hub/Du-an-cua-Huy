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
3.1 Nguon du lieu (USGS/public)
3.2 Mo ta cot du lieu
3.3 Lam sach du lieu
3.4 Tao dac trung:
- Lag features
- Rolling mean/std
- Bien ngoai sinh (neu co)
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
5.5 De xuat cai tien

## Chuong 6. Ket luan va huong phat trien
6.1 Ket luan
6.2 Han che
6.3 Huong phat trien
- Them XGBoost/LightGBM
- Mo rong GRU/Transformer
- Them bien ngoai sinh thuc te
- Mo rong multi-site

## Tai lieu tham khao
- Papers, docs, nguon du lieu

## Phu luc
- Link source code
- Huong dan chay
- Mau payload API
