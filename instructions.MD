# HƯỚNG DẪN CÁC THÔNG SỐ TINH CHỈNH DÀNH CHO NGƯỜI DÙNG
### (Hệ thống Khảo sát Shopee Smart Locker - TAM & TPB)

Tài liệu này được thiết kế để giải thích **dễ hiểu nhất** mọi nút bấm, thanh trượt và con số trên bảng điều khiển, ngay cả khi bạn **không có nền tảng về công nghệ hay lập trình**.

---

## 📌 MỤC LỤC
1. [Bộ Điều Khiển Trung Tâm (Nút Bấm Chính)](#1-bộ-điều-khiển-trung-tâm)
2. [Nhóm 1: Số Lượng Mẫu & Luồng Thu Thập](#2-nhóm-1-số-lượng-mẫu--luồng-thu-thập)
3. [Nhóm 2: Khoảng Trễ & Kỹ Thuật "Giả Người Thật" (Anti-Bot)](#3-nhóm-2-khoảng-trễ--kỹ-thuật-giả-người-thật-anti-bot)
4. [Nhóm 3: Lên Lịch Tự Động Theo Khung Giờ (Scheduler)](#4-nhóm-3-lên-lịch-tự-động-theo-khung-giờ-scheduler)
5. [Nhóm 4: Điều Chỉnh Cơ Cấu Nhân Khẩu Học (Tuổi, Giới tính, Vùng miền)](#5-nhóm-4-điều-chỉnh-cơ-cấu-nhân-khẩu-học)
6. [Nhóm 5: Tinh Chỉnh 11 Thang Đo Khảo Sát (Điểm Kỳ Vọng & Độ Tin Cậy)](#6-nhóm-5-tinh-chỉnh-11-thang-đo-khảo-sát)
7. [Các Thẻ Báo Cáo Trên Màn Hình Chính (KPI Cards)](#7-các-thẻ-báo-cáo-trên-màn-hình-chính-kpi-cards)
8. [Xuất Dữ Liệu Nghiên Cứu (SPSS & Excel)](#8-xuất-dữ-liệu-nghiên-cứu-spss--excel)

---

## 1. BỘ ĐIỀU KHIỂN TRUNG TÂM

Nằm ngay trên thanh bên trái (Sidebar). Đây là công tắc vận hành của toàn bộ hệ thống:

| Tên Nút / Chức Năng | Ý nghĩa thực tế | Nên chỉnh thế nào? |
| :--- | :--- | :--- |
| **Chế độ Dry-Run An Toàn** | **Chế độ chạy thử nghiệm (Không gửi thật)**.<br>Khi BẬT: Hệ thống sẽ tạo mẫu, tính điểm, vẽ biểu đồ nhưng **không** gửi lên máy chủ của Tally. | • **BẬT (Mặc định)**: Khi bạn muốn kiểm tra trước xem biểu đồ có đẹp không, số liệu ra sao.<br>• **TẮT (Chế độ Live)**: Chỉ tắt khi bạn muốn nộp khảo sát thật! |
| **▶ Chạy (Start)** | Bắt đầu chạy hệ thống để thu thập mẫu khảo sát theo đúng thông số bạn đã cài. | Nhấn khi đã cài đặt xong các thông số bên dưới. |
| **⏸ Dừng / ⏯ Tiếp (Pause/Resume)** | Tạm dừng quá trình làm việc giữa chừng (ví dụ khi bạn muốn xem lại số liệu) và cho phép chạy tiếp sau đó. | Không làm mất dữ liệu đã làm xong. |
| **⏹ Ngắt (Stop)** | Dừng hẳn toàn bộ tiến trình thu mẫu. | Nhấn khi đã đạt yêu cầu hoặc muốn kết thúc phiên làm việc. |
| **🔄 Đặt lại (Reset)** | Xóa trắng dữ liệu của phiên hiện tại để làm lại từ đầu. | Chỉ dùng khi muốn reset lại toàn bộ bảng thống kê. |
| **💾 Lưu Cấu Hình** | Lưu lại toàn bộ các con số bạn vừa điều chỉnh vào bộ nhớ để lần sau mở web không phải chỉnh lại từ đầu. | Nhấn sau khi bạn đã chỉnh xong thông số ưng ý. |
| **📂 Tải Cấu Hình Cũ** | Nạp lại bộ thông số bạn đã lưu trước đó chỉ với 1 cú click. | Dùng khi vừa vào lại trang web. |

---

## 2. NHÓM 1: SỐ LƯỢNG MẪU & LUỒNG THU THẬP
*(Mục "🎯 Số Lượng Mẫu & Luồng Thực Thi")*

* **Form URL**: Đường link của bài khảo sát (đã được điền sẵn biểu mẫu Shopee Locker). Bạn không cần thay đổi trừ khi có link mới.
* **Số mẫu mục tiêu (N)**:
  * **Giải thích**: Tổng số lượng phiếu khảo sát bạn muốn hoàn thành cho bài nghiên cứu.
  * **Ví dụ**: Đặt là `150` nghĩa là khi hệ thống làm đủ 150 phiếu hợp lệ thì sẽ tự động dừng.
* **Số lượng Worker Threads (Số người ảo làm đồng thời)**:
  * **Giải thích**: Số "trợ lý ảo" cùng làm bài khảo sát song song.
  * **Gợi ý**: 
    * Nên để từ **4 đến 6** worker. 
    * Không nên đặt quá cao (trên 12) vì có thể làm máy bị lag hoặc gửi yêu cầu quá dồn dập trông bất thường.
* **PRNG Master Seed (Mã số tạo ngẫu nhiên)**:
  * **Giải thích**: Một con số bất kỳ (ví dụ: `42`). Giữ nguyên con số này sẽ giúp bạn tạo lại được bộ dữ liệu giống hệt lần chạy trước (tính tái lập trong nghiên cứu khoa học). Nếu không có yêu cầu đặc biệt, bạn cứ để mặc định `42`.

---

## 3. NHÓM 2: KHOẢNG TRỄ & KỸ THUẬT "GIẢ NGƯỜI THẬT" (ANTI-BOT)
*(Mục "⏳ Pacing Delay & Jitter Chống Bot")*

Hệ thống có cơ chế thông minh giúp hành vi gửi khảo sát trông hoàn toàn tự nhiên như người thật bấm tay, không bị hệ thống chống Spam của Tally phát hiện.

* **Kiểu phân phối Jitter**:
  * **Truncated Gaussian (Người thật - Khuyên dùng)**: Thời gian chờ giữa các lần nộp sẽ dao động tự nhiên theo hình chuông, giống như tâm lý người thật (lúc nhanh, lúc chậm nhưng tập trung vào khoảng giữa).
  * **Uniform (Đều)**: Các khoảng chờ được chọn ngẫu nhiên rải đều.
* **Khoảng trễ nộp form (giây)**:
  * **Giải thích**: Hai mốc thời gian (Tối thiểu - Tối đa) mà trợ lý ảo sẽ "nghỉ ngơi" giữa 2 lần nộp phiếu liên tiếp.
  * **Ví dụ**:
    * Đang để `1800s - 3600s`: Tức là sau khi nộp 1 phiếu, worker sẽ chờ từ **30 phút đến 1 tiếng** mới nộp phiếu tiếp theo. Đây là tốc độ siêu an toàn và cực kỳ tự nhiên.
    * Khi cần test nhanh trong chế độ **Dry-Run**: Bạn có thể kéo về `1s - 5s` để quan sát biểu đồ nhảy số tức thì.
* **Mô phỏng giải lao ngắt quãng (Micro-breaks)**:
  * Giả lập thói quen sinh hoạt của con người: Cứ sau một số lượng bài làm nhất định (ví dụ làm xong 15–20 phiếu), worker sẽ tạm "uống nước/nghỉ giải lao" vài chục giây rồi mới làm tiếp.

---

## 4. NHÓM 3: LÊN LỊCH TỰ ĐỘNG THEO KHUNG GIỜ (SCHEDULER)
*(Mục "⏰ Lên Lịch Thực Thi")*

Nếu bạn muốn khảo sát chỉ nộp vào những khung giờ hợp lý trong ngày (ví dụ chỉ nộp vào ban ngày từ 8h sáng đến 10h tối, tránh nộp lúc 2-3h sáng vì bất thường):

* **Hẹn giờ chạy sau (giây)**: Đếm ngược một khoảng thời gian rồi mới bắt đầu chạy.
* **Khung giờ chạy (HH:MM-HH:MM)**:
  * **Ví dụ điền**: `08:00-11:30, 13:30-22:00`
  * Hệ thống sẽ chỉ gửi phiếu vào khung giờ trên. Khi đến giờ nghỉ trưa hoặc ban đêm, hệ thống tự động tạm ngưng chờ đến khung giờ tiếp theo.
* **Giới hạn theo đợt (Batch size)** & **Thời gian hạ nhiệt (Cooldown)**:
  * Cho phép chia đợt: Ví dụ gửi mỗi đợt 30 phiếu, sau đó hệ thống nghỉ "hạ nhiệt" 10 phút rồi mới chạy đợt tiếp theo.

---

## 5. NHÓM 4: ĐIỀU CHỈNH CƠ CẤU NHÂN KHẨU HỌC
*(Mục "👥 Tùy Biến Trọng Số Nhân Khẩu Học Mặc Định")*

Giúp bạn kiểm soát tập khách hàng mẫu (Gen Z) phản ánh đúng thực tế nghiên cứu của bạn:

* **Độ tuổi trung bình**: Mặc định là `21.2` tuổi (Đặc trưng của sinh viên/người mới đi làm thuộc Gen Z từ 18 - 27 tuổi).
* **Độ lệch chuẩn tuổi**: Độ phân tán tuổi xung quanh tuổi trung bình. Để mặc định `1.8` giúp tuổi rải đều đẹp từ năm nhất đến năm cuối đại học.
* **Tỷ lệ Nữ (%)**: Mặc định khoảng `65%` (Phản ánh đúng thực tế nhóm đối tượng mua sắm online trên Shopee thường là nữ chiếm đa số). Bạn có thể kéo tăng/giảm theo giả định đề tài.
* **Tỷ lệ địa bàn (TP.HCM / Hà Nội / Đà Nẵng)**:
  * Hệ thống chia tỷ lệ tập trung tại 3 đô thị lớn có triển khai Shopee Smart Locker (TP.HCM ~50%, Hà Nội ~40%, Đà Nẵng ~10%).
  * Hệ thống tự động chuẩn hóa 100% mẫu chỉ thuộc 3 thành phố này (0% tỉnh ngoài để tránh câu trả lời bị loại do chưa từng tiếp cận tủ locker).

---

## 6. NHÓM 5: TINH CHỈNH 11 THANG ĐO KHẢO SÁT
*(Mục "📐 Tùy Biến Điểm Kỳ Vọng 11 Thang Đo TAM & TPB")*

Bài nghiên cứu đo lường bằng thang đo **Likert 6 điểm** (từ 1: Hoàn toàn không đồng ý đến 6: Hoàn toàn đồng ý). Bạn có thể điều chỉnh điểm trung bình kỳ vọng ($\mu$) cho từng yếu tố:

| Ký Hiệu | Tên Thang Đo | Ý Nghĩa Thực Tế | Điểm Mặc Định & Gợi Ý |
| :---: | :--- | :--- | :---: |
| **PU** | Nhận thức Hữu ích (Perceived Usefulness) | Khách thấy tủ locker Shopee tiện lợi, tiết kiệm thời gian nhận hàng. | `4.65 / 6` (Tích cực) |
| **PEOU** | Dễ sử dụng (Perceived Ease of Use) | Thao tác nhập mã/quét QR mở tủ rất đơn giản, dễ thao tác. | `4.70 / 6` (Rất tốt) |
| **PES** | Bền vững môi trường (Perceived Environmental Sustainability) | Giảm số lần giao hàng thất bại, giảm khí thải xe máy. | `4.75 / 6` (Rất cao) |
| **PC** | Mức độ phức tạp (Perceived Complexity) | *Lưu ý: Thang đo nghịch.* Càng thấp nghĩa là người dùng thấy tủ **ít phức tạp** (dễ dùng). Khi phân tích SPSS, hệ thống đã tự động đảo chiều để đồng bộ. | `2.35 / 6` (Thấp = Tốt) |
| **PI** | Đổi mới cá nhân (Personal Innovativeness) | Tính cách thích thử công nghệ mới của giới trẻ Gen Z. | `4.25 / 6` (Khá tốt) |
| **SE** | Tự tin công nghệ (Self-Efficacy) | Gen Z tự tin có đủ kỹ năng để dùng tủ thông minh mà không cần ai chỉ dẫn. | `4.75 / 6` (Rất cao) |
| **LC** | Vị trí tiện lợi (Location Convenience) | Tủ đặt gần ký túc xá, chung cư, trạm xe bus, tiện ghé lấy. | `4.70 / 6` (Cao) |
| **ATT** | Thái độ đối với dịch vụ (Attitude) | Cảm xúc thích thú và ủng hộ hình thức nhận hàng qua tủ. | `4.65 / 6` (Tích cực) |
| **SN** | Chuẩn chủ quan (Subjective Norms) | Bạn bè, người quen khuyên nên thử nhận qua tủ locker. | `4.30 / 6` (Khá cao) |
| **PBC** | Kiểm soát hành vi nhận thức (Perceived Behavioral Control) | Người dùng hoàn toàn chủ động được thời gian ra lấy hàng bất cứ lúc nào rảnh. | `4.70 / 6` (Rất cao) |
| **ITU** | Ý định sử dụng tương lai (Intention to Use) | Mong muốn tiếp tục chọn hình thức này trong các đơn hàng tới. | `4.65 / 6` (Rất cao) |

* **Ngưỡng Cronbach's Alpha NCKH ($\ge$ chuẩn)**:
  * **Khái niệm đơn giản**: Đây là "chứng chỉ chất lượng" của bảng số liệu. Trong nghiên cứu khoa học, bảng câu hỏi phải đạt hệ số $\alpha \ge 0.70$ hoặc $\ge 0.80$ thì mới được coi là dữ liệu tin cậy, nhất quán.
  * Mặc định hệ thống cài đặt chuẩn khắt khe: **$\ge 0.820$** (Đảm bảo số liệu xuất ra chạy mô hình SmartPLS/SPSS luôn đẹp và đạt chuẩn công bố khoa học).
* **Nút "↺ Khôi Phục Điểm Chuẩn Lý Thuyết"**: Nếu lỡ tay chỉnh lung tung, chỉ cần bấm nút này thì tất cả 11 thang đo sẽ quay về đúng chuẩn ban đầu.

---

## 7. CÁC THẺ BÁO CÁO TRÊN MÀN HÌNH CHÍNH (KPI CARDS)

Ngay phía trên giao diện là 5 thẻ đo đạc theo thời gian thực (cập nhật từng giây):

1. **Mục tiêu Khảo sát**: Đang có bao nhiêu phiếu / Mục tiêu (ví dụ: `150 / 150` là đã xong 100%).
2. **Kết quả Gửi form**: Số phiếu thành công (màu xanh lá ✓) và số phiếu bị lỗi mạng (màu đỏ ✗).
3. **Tốc độ & Thông lượng**: Tốc độ gửi trung bình (bao nhiêu mẫu/phút) và tổng thời gian đã chạy.
4. **Độ tin cậy Cronbach ($\alpha$)**: 
   * Nếu hiện viên nhãn màu **Xanh lá**: Số liệu cực kỳ tốt, đạt chuẩn NCKH.
   * Nếu hiện viên nhãn màu **Vàng/Xám**: Đang trong giai đoạn thu thập ban đầu (chưa đủ 10 mẫu để tính độ tin cậy ổn định).
5. **Tự động hóa Scheduler**: Cho biết hệ thống đang chạy tự do hay đang chờ khung giờ hẹn giờ.

---

## 8. XUẤT DỮ LIỆU NGHIÊN CỨU (SPSS & EXCEL)
*(Chuyển sang Tab 3: "💾 Xem Dữ Liệu & Xuất File")*

Khi đã thu thập xong mẫu, bạn có 2 nút tải file tiện lợi:

1. **📥 Tải Tệp CSV Số Hóa (SPSS, SmartPLS, AMOS)**:
   * **Công dụng**: Tệp dữ liệu dạng số thuần túy (Pure Numeric Matrix).
   * **Ưu điểm**: Mọi câu hỏi chữ (như "Hoàn toàn đồng ý", "Nữ", "TP.HCM") đã được tự động chuyển thành số (`1, 2, 3, 4, 5, 6`) và thang đo ngược `PC` đã được đảo chiều toán học sẵn. Bạn chỉ việc import thẳng vào phần mềm SPSS hoặc SmartPLS để chạy hồi quy / mô hình SEM mà không cần mất công clean data.
2. **📥 Tải Sổ Tay Mã Hóa 3-Sheet Excel (.xlsx)**:
   * **Công dụng**: File báo cáo thẩm mỹ đầy đủ gồm 3 trang:
     * *Sheet 1 (Data Matrix)*: Toàn bộ bảng dữ liệu số.
     * *Sheet 2 (Variable Codebook)*: Bảng đối chiếu mã câu hỏi, giải thích chi tiết biến nào ứng với câu hỏi nào trong form.
     * *Sheet 3 (Summary Stats)*: Bảng tổng hợp Mean (Trung bình), Std (Độ lệch chuẩn) và Cronbach's Alpha của từng thang đo.

---

> 💡 **MẸO DÀNH CHO BẠN KHI LÀM THỬ NGHIỆM**:
> 1. Luôn để **Chế độ Dry-Run: BẬT**.
> 2. Kéo khoảng trễ về ngắn (`1s - 5s`) rồi bấm **▶ Chạy**.
> 3. Xem tab 2 (Biểu đồ) và tab 3 (Dữ liệu) xuất hiện sau vài giây để kiểm tra giao diện.
> 4. Khi cần nộp thật: Bật chế độ trễ khuyến nghị (`1800s - 3600s`), tắt Dry-Run và bấm **▶ Chạy**.
