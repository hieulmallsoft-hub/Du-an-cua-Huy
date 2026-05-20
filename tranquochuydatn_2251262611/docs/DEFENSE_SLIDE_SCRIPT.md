# Script Trinh Bay Bao Ve (10-12 phut)

## Slide 1 - De tai
- Em xin trinh bay de tai du doan muc nuoc ngam theo chuoi thoi gian.

## Slide 2 - Bai toan
- Dau vao: lich su groundwater_level va bien ngoai sinh (neu co).
- Dau ra: du doan t+1 va du doan N buoc.

## Slide 3 - Du lieu
- Nguon USGS/public.
- So dong du lieu, tram du lieu, khoang thoi gian.

## Slide 4 - Tien xu ly
- Sap xep theo ngay.
- Tao lag/rolling.
- Chia train/test theo thoi gian.

## Slide 5 - Mo hinh
- VAR, VECM (tuyen tinh, dong lien ket).
- LSTM (phi tuyen, phu thuoc dai han).
- Hybrid VAR/VECM + LSTM residual.
- Baseline doi chieu: naive_last.

## Slide 6 - Ket qua holdout
- Trinh bay bang MAE/RMSE/R2.
- Giai thich vi sao model duoc chon.

## Slide 7 - Ket qua backtest nhieu horizon
- Trinh bay bieu do RMSE theo horizon.
- Neu horizon dai giam chat luong, neu ro huong cai tien.

## Slide 8 - Demo he thong
- Mo UI: nhap lich su, an du doan t+1 va N buoc.
- Cho thay xu huong tang/giam va ket qua JSON.

## Slide 9 - Dong gop
- Quy trinh train/tune/backtest/day du.
- API + giao dien demo.
- Bao cao tu dong.

## Slide 10 - Han che va huong phat trien
- Them du lieu ngoai sinh thuc te.
- Thu mo hinh nang cao.
- Mo rong nhieu tram.

## Q&A
- Chuan bi san 4 y:
1) Tai sao khong random split?
2) Tai sao can benchmark naive?
3) Vi sao horizon dai kho hon?
4) He thong co the trien khai that khong?
