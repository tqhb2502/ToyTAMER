# TAMER
TAMER (Training an Agent Manually via Evaluative Reinforcement) là một framework cho phép con người đưa ra feedback cho agent trong học tăng cường, đề xuất bởi [Knox + Stone](http://www.cs.utexas.edu/~sniekum/classes/RLFD-F16/papers/Knox09.pdf) vào năm 2009.

Đây là mã nguồn của một TAMER agent được chuyển đổi từ Q-learning agent bằng các bước được tác giả cung cấp tại [đây](http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html). Mã nguồn này cũng được chỉnh sửa dựa trên mã nguồn gốc có thể tìm thấy tại [đây](https://github.com/benibienz/TAMER).

## Chuẩn bị
- Phiên bản Python: `3.8.20`
- Cài đặt gói cần thiết: `pip install -r requirements.txt`

## Chạy chương trình và huấn luyện
- Chạy chương trình: `python run.py`
- Huấn luyện: nhấn phím W để thưởng cho agent, nhấn phím A để phạt agent
- Bạn có thể chọn trò chơi khác hoặc thay đổi tham số huấn luyện trong `run.py`