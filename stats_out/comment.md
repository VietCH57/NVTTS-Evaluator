1. Dev dùng được cho Track A nhưng không có speaker lạ cho Track B. Cả 46 speaker dev đều có trong train, và không có câu dev nào trùng text với train (0/316), nên dev là held-out đúng nghĩa cho Track A. Nhưng không có tập speaker unseen nào để thử zero-shot. Nếu muốn đánh giá Track B cục bộ, ta phải tự giữ lại một số speaker khỏi quá trình huấn luyện.

Phương án khả thi nhất là nhóm đuôi dài: 173 speaker có ≤5 câu train, chỉ chiếm 0.95h trên 5.9h (16%). Trong đó 30 speaker có mặt ở dev (31 câu, 63 NV), đủ để làm tập thử Track B nhỏ. Cái giá là mất 16% dữ liệu train, và khi nộp bài thật thì huấn luyện lại với đầy đủ dữ liệu.

2. Phân bố speaker rất lệch. 6 speaker chiếm 76% câu dev, riêng spk_0000 chiếm 37%. Trong khi đó 101 speaker train chỉ có 1 câu, và 30/46 speaker dev chỉ có 1 câu. Vì vậy mọi metric cần báo cả micro (theo câu) và macro (trung bình theo speaker), kèm số mẫu n. Ngay cả "speaker đã thấy" ở nhóm đuôi cũng chỉ có 3 đến 78 giây audio, gần như zero-shot.

3. Transcript và tag là nhãn tự động, nên có trần chất lượng. Ở spk_0000 có các cụm như "ipo phan taxi premier league", "jorgen clob", "san di chín mốt", là lỗi nhận dạng tên riêng. Khớp với việc đề bài nói corpus dựng bằng pipeline tự động. Hệ quả: chạy Zipformer trên audio thật của dev vẫn cho WER > 0, và NVPA trên audio thật cũng sẽ < 1. Vì vậy cần một ground-truth calibration run (coi audio dev thật là "output của model") để lấy trần cho WER, NVPA, pMOS, SS. Điểm của model chỉ có ý nghĩa khi so với trần này.

4. Điều này ảnh hưởng thiết kế NVPA:

Breathing chiếm 80% sự kiện (3733/4646), laughter 14.7%, sniff và throatclearing mỗi loại khoảng 2.5%. Ở dev, sniff chỉ có 14 và throatclearing 19 sự kiện, nên mỗi sự kiện sai kéo NVPA loại đó xuống 5 đến 7 điểm. Cần luôn báo kèm n và khoảng tin cậy bootstrap, và báo cả NVPA micro lẫn macro theo loại.
NV chủ yếu nằm giữa câu (95%). Đầu câu chỉ 2.2% (13 sự kiện ở dev), cuối câu 3.1% (27 ở dev), nên breakdown theo vị trí đầu/cuối sẽ rất nhiễu.
Khoảng cách trung vị giữa hai NV liền kề là 12 từ (tương đương khoảng 3 giây). Với cửa sổ dung sai ±2 từ, một NV đặt ngẫu nhiên đã có khả năng trúng cỡ 40% (ước lượng thô). Vì vậy cần thêm phép thử shuffle baseline: xáo vị trí NV rồi tính lại NVPA, để biết dung sai đang quá lỏng hay không.
85 cặp NV nằm cùng một khe giữa hai từ (51 cặp là sniff cùng breathing), và 5.5% cặp liền kề cách nhau ≤1 từ. Matcher phải có cấu hình rõ cho trường hợp thứ tự trong cùng khe không xác định.
Từ tiếng Việt tách theo khoảng trắng nên 1 token xấp xỉ 0.25 giây (4.07 token/giây). Dung sai nên cho cấu hình được cả theo token lẫn theo giây.

5. Không câu nào thiếu NV (0/2055). Do đó diagnostic "có NV vs không NV" không lấy được từ dữ liệu. Muốn trả lời "NV có làm tăng WER không?" ta cần một lần synthesize đối chứng cùng text nhưng bỏ tag. Manifest có thể có trường tùy chọn control_audio, vẫn là đánh giá một model.

6. Định dạng sạch, parser đơn giản. Tag luôn cách bằng khoảng trắng, không dính chữ, không dính dấu câu, không lệch ngoặc, không có chữ số (số đã được đọc thành chữ), language_id toàn vi. Text gần như toàn chữ thường (116/1739 câu có chữ hoa, có vẻ ở đầu câu), nên chuẩn hóa WER chỉ cần NFC, lowercase, bỏ dấu câu.

7. Audio và độ dài. Toàn bộ 24 kHz mono, nhưng 25% là PCM_16 và 75% là PCM_24, có thể đến từ các nguồn khác nhau, không ảnh hưởng đánh giá. Câu dài: trung vị 12.7 giây, tối đa 25.8 giây. Phân bố dev giống train (4.00 so với 4.07 từ/giây), nên dev đại diện tốt. Các model chấm điểm (DNSMOS, ECAPA, Zipformer) cần được kiểm tra cách xử lý audio dài hơn 10 giây.