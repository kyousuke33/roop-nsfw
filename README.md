# 🚀 Roop ULTIMATE - Face Swap Không Giới Hạn!  
🔥 **ROOP phiên bản tối ưu!** Không còn bộ lọc NSFW, hiệu suất tối đa, hoạt động trơn tru trên cả CPU & GPU!  

## 📌 Thông Tin Dự Án  

[![Build Status](https://img.shields.io/github/actions/workflow/status/kyousuke33/roop-nsfw/ci.yml.svg?branch=main)](https://github.com/kyousuke33/roop-nsfw/actions?query=workflow:ci)
- 🏷 **Phiên Bản:** ROOP Ultimate v1.3.4 – Tối ưu hóa tốc độ & cải thiện khả năng phát hiện khuôn mặt.  
- 🤖 **Mô tả:** Roop là công cụ thay đổi khuôn mặt trong video & ảnh một cách dễ dàng. Chỉ cần một ảnh khuôn mặt!  
- 📜 **Các Tính Năng Mới:**  
  - 🚀 **Resume Progress** - Tiếp tục từ frame đã xử lý khi gặp sự cố.  
  - 🎭 **Face Enhancer** - Cải thiện độ sắc nét cho khuôn mặt.  
  - 🏆 **Multi-Face Detection** - Nhận diện nhiều khuôn mặt trên ảnh/video.  
  - 🔍 **Logging Chi Tiết** - Hiển thị tiến trình từng frame!  
  - 📖 **Hướng Dẫn:** Chạy ROOP trực tiếp với Google Colab hoặc trên máy cá nhân.  

## 📥 Cài Đặt Nhanh Trên Google Colab  
📌 **Sao chép & chạy lệnh sau:**  
```bash
git clone https://github.com/kyousuke33/roop-nsfw.git
cd roop-nsfw
pip install -r requirements.txt
```
📌 **Chạy ROOP từ Terminal / Command Line:**  
```bash
python run.py -s face.jpg -t input.mp4 -o output.mp4 --frame-processor face_swapper --keep-fps
```

## ⚙ Các Tùy Chọn Quan Trọng  
| Tùy Chọn | Chức Năng |
|----------|----------|
| `-s, --source` | Chọn ảnh khuôn mặt để thay thế (JPG, PNG). |
| `-t, --target` | Chọn ảnh hoặc video đầu vào. |
| `-o, --output` | Chọn file hoặc thư mục đầu ra. |
| `--keep-fps` | Giữ nguyên FPS của video đầu vào. |
| `--keep-frames` | Lưu các frame tạm để debug. |
| `--many-faces` | Xử lý tất cả khuôn mặt trong ảnh/video. |
| `--resume-frame` | Tiếp tục từ frame đã xử lý trước đó nếu bị gián đoạn. |
| `--output-video-encoder` | Bộ mã hóa video (libx264, libx265, h264_nvenc). |
| `--execution-provider` | Chạy trên CPU hoặc CUDA (nếu có GPU). |
| `--execution-threads` | Số luồng CPU/GPU sử dụng. |

---

## 🏆 Các Tính Năng Mới & Cải Tiến  
✅ **Resume Face Swap**: Nếu bị gián đoạn, có thể tiếp tục từ frame đã swap.  
🎭 **Face Enhancer**: Tăng cường chi tiết khuôn mặt, mịn màng & nét hơn so với ảnh/video gốc.  
📊 **Tối ưu hiệu suất**: Giảm tải bộ nhớ RAM, cải thiện tốc độ xử lý.  
📖 **Logging Chi Tiết**: Hiển thị thông tin tiến trình theo thời gian thực.  

---

## 📜 Giấy Phép & Điều Khoản  
⚠ **Lưu ý:** Dự án này sử dụng nhiều thư viện và mô hình của bên thứ ba. Vui lòng kiểm tra giấy phép và tuân thủ các điều khoản sử dụng.  
📜 **Giấy phép:** MIT License  

---

## 🌟 Liên Hệ & Góp Ý  
📬 Nếu bạn có bất kỳ câu hỏi hoặc góp ý nào, hãy mở Issue hoặc tham gia cộng đồng của chúng tôi trên GitHub!  
⭐ **Thích dự án này?** Đừng quên Star 🌟 trên GitHub để giúp dự án phát triển hơn!  

---

## 🔗 **Bắt Đầu Trải Nghiệm Ngay Trên Colab!**  
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)  
📖 **[Hướng Dẫn Sử Dụng](https://github.com/kyousuke33/roop-nsfw/wiki/Wiki%E2%80%90Roop%E2%80%90Ultimate)**  

Cùng trải nghiệm công nghệ AI mạnh mẽ nhất với **ROOP Ultimate** ngay hôm nay! 🚀
